# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply

from .dextra_amp_env_cfg import DextraAmpEnvCfg
from .motions import MotionLoader

def compute_fk_observations(joint_pos: torch.Tensor) -> torch.Tensor:
    """
    Compute FK-based observations from joint positions
    
    Args:
        joint_pos: [N, 12] joint positions
        
    Returns:
        fk_obs: [N, 7] (base_height, left_foot_xyz, right_foot_xyz)
    """
    # Dextra link lengths (from URDF)
    THIGH_LENGTH = 0.095807  # meters
    CALF_LENGTH = 0.093      # meters
    BASE_OFFSET = 0.2865     # initial base height
    
    # Joint indices (Dextra ordering)
    # [L_HipYaw, R_HipYaw, L_HipRoll, R_HipRoll, 
    #  L_Thigh, R_Thigh, L_Calf, R_Calf,
    #  L_AnklePitch, R_AnklePitch, L_AnkleRoll, R_AnkleRoll]
    L_THIGH = 4
    L_CALF = 6
    R_THIGH = 5
    R_CALF = 7
    
    # Extract joints
    l_thigh = joint_pos[:, L_THIGH]
    l_calf = joint_pos[:, L_CALF]
    r_thigh = joint_pos[:, R_THIGH]
    r_calf = joint_pos[:, R_CALF]
    
    # FK: Leg heights
    l_leg_height = (THIGH_LENGTH * torch.cos(l_thigh) + 
                    CALF_LENGTH * torch.cos(l_thigh + l_calf))
    r_leg_height = (THIGH_LENGTH * torch.cos(r_thigh) + 
                    CALF_LENGTH * torch.cos(r_thigh + r_calf))
    
    # Base height (average of both legs)
    base_height = (l_leg_height + r_leg_height) / 2.0 + BASE_OFFSET
    
    # Foot positions (sagittal plane)
    l_foot_x = (THIGH_LENGTH * torch.sin(l_thigh) + 
                CALF_LENGTH * torch.sin(l_thigh + l_calf))
    r_foot_x = (THIGH_LENGTH * torch.sin(r_thigh) + 
                CALF_LENGTH * torch.sin(r_thigh + r_calf))
    
    l_foot_z = l_leg_height
    r_foot_z = r_leg_height
    
    # Y-axis approximated as 0 (lateral motion small)
    zeros = torch.zeros_like(l_foot_x)
    
    fk_obs = torch.stack([
        base_height,
        l_foot_x, zeros, l_foot_z,  # Left foot
        r_foot_x, zeros, r_foot_z,  # Right foot
    ], dim=-1)
    
    return fk_obs


class DextraAmpEnv(DirectRLEnv):
    cfg: DextraAmpEnvCfg

    def __init__(self, cfg: DextraAmpEnvCfg, render_mode: str | None = None, **kwargs):

        if cfg.use_fk_observations:
            # Policy obs: 31D (encoder + FK)
            cfg.num_observations = 31
            cfg.observation_space = 31
            print(f"\n{'='*70}")
            print(f"🔧 FK Mode Enabled!")
            print(f"{'='*70}")
            print(f"📊 Policy obs:  31D (24D encoder + 7D FK)")
            print(f"📊 AMP obs:     43D (ground truth - 항상!)")
            print(f"{'='*70}\n")
        else:
            # Default: Full privileged
            cfg.num_observations = 43
            cfg.observation_space = 43
            print(f"\n{'='*70}")
            print(f"📊 Full Privileged Mode")
            print(f"{'='*70}")
            print(f"📊 Policy obs:  43D (ground truth)")
            print(f"📊 AMP obs:     43D (ground truth)")
            print(f"{'='*70}\n")

        super().__init__(cfg, render_mode, **kwargs)

        # Action offset and scale
        dof_lower_limits = torch.full((12,), -2.6180, device=self.device)
        dof_upper_limits = torch.full((12,), 2.6180, device=self.device)
        # dof_lower_limits = torch.full((12,), -1.5708, device=self.device)
        # dof_upper_limits = torch.full((12,), 1.5708, device=self.device)
        self.action_offset = 0.5 * (dof_upper_limits + dof_lower_limits)
        self.action_scale = (dof_upper_limits - dof_lower_limits)/2

        # Load motion
        self._motion_loader = MotionLoader(
            motion_file=self.cfg.motion_file,
            device=self.device,
            speed_scale=self.cfg.motion_speed_scale,
        )

        print(f"\n{'='*60}")
        print(f"Motion: {cfg.motion_file}")
        print(f"Duration: {self._motion_loader.duration:.2f}s")
        print(f"Frames: {self._motion_loader.num_frames}")
        print(f"{'='*60}\n")

        # DOF and key body indexes
        key_body_names = ["L_AnkleRoll_Link_1", "R_AnkleRoll_Link_1"]
        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in key_body_names]
        self.motion_dof_indexes = self._motion_loader.get_dof_index(self.robot.data.joint_names)
        self.motion_ref_body_index = self._motion_loader.get_body_index([self.cfg.reference_body])[0]
        self.motion_key_body_indexes = self._motion_loader.get_body_index(key_body_names)

        # AMP observation space
        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device
        )

        # Rolling vx buffer for windowed-average termination
        # Shape: (num_envs, vel_window_steps); filled with a large value initially so
        # the grace period does not falsely trigger before the buffer is full.
        ws = self.cfg.vel_window_steps
        self._vel_window_buf = torch.full(
            (self.num_envs, ws), fill_value=1e3, device=self.device
        )
        self._vel_window_ptr = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._vel_window_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Action buffers for action-rate penalty (anti-jitter)
        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.prev_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

        # Rolling statistics over the most recent ``num_envs`` completed
        # episodes. Keeping one entry per episode avoids bias when many
        # environments time out together but deaths happen in smaller batches.
        self._episode_stats_capacity = self.num_envs
        self._episode_stats_count = 0
        self._episode_stats_write_index = 0
        self._completed_episode_lengths = torch.zeros(
            self._episode_stats_capacity, dtype=torch.float32, device=self.device
        )
        self._completed_episode_timeouts = torch.zeros_like(self._completed_episode_lengths)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        # Ground plane
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="max",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )
        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        # Add robot to scene
        self.scene.articulations["robot"] = self.robot
        
        # Lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions[:] = self.actions
        self.actions = actions.clone()

    def _apply_action(self):
        actions = torch.clamp(self.actions, -1.0, 1.0)
        target = self.action_offset + self.action_scale * actions
        self.robot.set_joint_position_target(target)

    def _compute_fk_policy_obs(self, joint_vel_obs: torch.Tensor) -> torch.Tensor:
        """
        Compute policy observations for FK mode (31D)
        Encoder (24D) + FK (7D)
        """
        
        # Encoder observations (24D)
        encoder_obs = torch.cat([
            self.robot.data.joint_pos,              # 12D
            joint_vel_obs,                          # 12D, sensor-noised during DR
        ], dim=-1)
        
        # FK observations (7D)
        fk_obs = compute_fk_observations(self.robot.data.joint_pos)
        
        # Total: 31D
        return torch.cat([encoder_obs, fk_obs], dim=-1)


    def _get_observations(self) -> dict:
        # Build task observation
        # obs = compute_obs(
        #     self.robot.data.joint_pos,
        #     self.robot.data.joint_vel,
        #     self.robot.data.body_pos_w[:, self.ref_body_index],
        #     self.robot.data.body_quat_w[:, self.ref_body_index],
        #     self.robot.data.body_lin_vel_w[:, self.ref_body_index],
        #     self.robot.data.body_ang_vel_w[:, self.ref_body_index],
        #     self.robot.data.body_pos_w[:, self.key_body_indexes],
        # )

        joint_vel_obs = self.robot.data.joint_vel
        noise_cfg = getattr(self, "_joint_velocity_observation_noise_cfg", None)
        if noise_cfg is not None:
            joint_vel_obs = noise_cfg.func(joint_vel_obs, noise_cfg)

        if self.cfg.use_fk_observations:
            # FK mode: 31D policy observations
            policy_obs = self._compute_fk_policy_obs(joint_vel_obs)
        else:
            # Default: 43D full privileged
            policy_obs = compute_obs(
                self.robot.data.joint_pos,
                joint_vel_obs,
                self.robot.data.body_pos_w[:, self.ref_body_index],
                self.robot.data.body_quat_w[:, self.ref_body_index],
                self.robot.data.body_lin_vel_w[:, self.ref_body_index],
                self.robot.data.body_ang_vel_w[:, self.ref_body_index],
                self.robot.data.body_pos_w[:, self.key_body_indexes],
            )

        # ========================================
        # ✅ 중요: AMP obs는 항상 ground truth (43D)!
        # ========================================
        amp_obs = compute_obs(
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            self.robot.data.body_pos_w[:, self.ref_body_index],
            self.robot.data.body_quat_w[:, self.ref_body_index],
            self.robot.data.body_lin_vel_w[:, self.ref_body_index],
            self.robot.data.body_ang_vel_w[:, self.ref_body_index],
            self.robot.data.body_pos_w[:, self.key_body_indexes],
        )

        # Update AMP observation history
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        
        # Build AMP observation
        #self.amp_observation_buffer[:, 0] = obs.clone()
        self.amp_observation_buffer[:, 0] = amp_obs.clone()
        # Do not replace `self.extras` entirely: SKRL logs `infos["log"]` from extras, and
        # `_get_rewards` / `_get_dones` run before this in `DirectRLEnv.step`.
        self.extras["amp_obs"] = self.amp_observation_buffer.view(-1, self.amp_observation_size)

        # ✅ Discriminator reward 확인
        # if self.common_step_counter % 100 == 0 and hasattr(self, 'extras'):
        #     if 'amp_obs' in self.extras:
        #         # Student obs
        #         student_amp = self.extras['amp_obs'][:4]  # 첫 4개 env
                
        #         # Reference obs (motion file)
        #         ref_amp = self.collect_reference_motions(4, None)
                
        #         # Discriminator score
        #         with torch.no_grad():
        #             student_score = self.discriminator(student_amp).mean()
        #             ref_score = self.discriminator(ref_amp).mean()
                
        #         print(f"\n[Discriminator @ {self.common_step_counter}]")
        #         print(f"  Student score: {student_score:.3f}")
        #         print(f"  Reference score: {ref_score:.3f}")
        #         print(f"  Gap: {(ref_score - student_score):.3f}")

        #return {"policy": obs}
        return {"policy": policy_obs}

    def _get_rewards(self) -> torch.Tensor:
        # --- velocity tracking reward (world +X) ---
        vx = self.robot.data.body_lin_vel_w[:, self.ref_body_index, 0]
        vel_reward = torch.exp(-self.cfg.target_vel_tracking_coeff * (vx - self.cfg.target_vel_x_world) ** 2)

        # --- foot-flat reward ---
        foot_quats = self.robot.data.body_quat_w[:, self.key_body_indexes]  # (N, 2, 4)
        world_z = torch.zeros(self.num_envs, 2, 3, device=self.device)
        world_z[..., 2] = 1.0
        foot_z_world = quat_apply(foot_quats, world_z)           # (N, 2, 3)
        dot = foot_z_world[..., 2].clamp(-1.0, 1.0)              # (N, 2)
        foot_flat_reward = torch.exp(-self.cfg.foot_flat_coeff * (1.0 - dot) ** 2).mean(dim=-1)  # (N,)

        combined = self.cfg.vel_reward_weight * vel_reward + self.cfg.foot_flat_reward_weight * foot_flat_reward

        # --- action-rate penalty (anti-jitter) ---
        if self.cfg.action_rate_penalty_weight > 0.0:
            action_rate_penalty = ((self.actions - self.prev_actions) ** 2).mean(dim=-1)  # (N,)
            combined = combined - self.cfg.action_rate_penalty_weight * action_rate_penalty
        else:
            action_rate_penalty = torch.zeros(self.num_envs, device=self.device)

        # --- TensorBoard (SKRL): trainer only logs infos["log"] values that are scalar tensors ---
        prev_log = self.extras.get("log") if isinstance(self.extras.get("log"), dict) else {}
        self.extras["log"] = {
            **prev_log,
            "reward/vel_tracking": vel_reward.mean().detach(),
            "reward/foot_flat": foot_flat_reward.mean().detach(),
            "reward/task_combined": combined.mean().detach(),
            "reward/action_rate_penalty": action_rate_penalty.mean().detach(),
            "metric/base_vel_x": vx.mean().detach(),
            "metric/foot_dot_z": dot.mean().detach(),
            "metric/base_height": self.robot.data.body_pos_w[:, self.ref_body_index, 2].mean().detach(),
        }

        return combined

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.early_termination:
            too_low = self.robot.data.body_pos_w[:, self.ref_body_index, 2] < self.cfg.termination_height

            vx = self.robot.data.body_lin_vel_w[:, self.ref_body_index, 0]

            # --- instantaneous vx check (legacy) ---
            if self.cfg.termination_min_vel_x > 0.0:
                too_slow_instant = vx < self.cfg.termination_min_vel_x
            else:
                too_slow_instant = torch.zeros_like(too_low)

            # --- windowed-average vx check ---
            if self.cfg.vel_window_min_vx > 0.0:
                ws = self.cfg.vel_window_steps
                # Write current vx into rolling buffer
                idx = self.episode_length_buf % ws  # (num_envs,)
                self._vel_window_buf[torch.arange(self.num_envs, device=self.device), idx] = vx
                self._vel_window_count = torch.clamp(self._vel_window_count + 1, max=ws)
                # Only trigger after the buffer is full (grace period)
                buffer_full = self._vel_window_count >= ws
                mean_vx = self._vel_window_buf.mean(dim=1)
                too_slow_window = buffer_full & (mean_vx < self.cfg.vel_window_min_vx)
            else:
                too_slow_window = torch.zeros_like(too_low)

            died = too_low | too_slow_instant | too_slow_window
        else:
            died = torch.zeros_like(time_out)

        if "log" not in self.extras or not isinstance(self.extras["log"], dict):
            self.extras["log"] = {}
        log = self.extras["log"]
        log["episode/deaths"] = died.sum().to(dtype=torch.float32).detach()
        log["episode/timeouts"] = time_out.sum().to(dtype=torch.float32).detach()

        self._log_completed_episode_metrics(died, time_out, log)

        return died, time_out

    def _log_completed_episode_metrics(
        self, died: torch.Tensor, time_out: torch.Tensor, log: dict[str, torch.Tensor]
    ) -> None:
        """Update and log statistics over recently completed episodes.

        The rolling window contains one sample per completed episode and has a
        capacity equal to the number of parallel environments. An episode that
        is both terminal and at the horizon is classified as a death, not as a
        successful timeout, so the timeout and death fractions sum to one.
        """
        completed_ids = (died | time_out).nonzero(as_tuple=False).squeeze(-1)
        num_completed = completed_ids.numel()

        if num_completed > 0:
            completed_lengths = self.episode_length_buf[completed_ids].to(dtype=torch.float32)
            completed_timeouts = (time_out[completed_ids] & ~died[completed_ids]).to(dtype=torch.float32)

            capacity = self._episode_stats_capacity
            if num_completed >= capacity:
                self._completed_episode_lengths[:] = completed_lengths[-capacity:]
                self._completed_episode_timeouts[:] = completed_timeouts[-capacity:]
                self._episode_stats_write_index = 0
                self._episode_stats_count = capacity
            else:
                write_index = self._episode_stats_write_index
                first_count = min(num_completed, capacity - write_index)
                first_slice = slice(write_index, write_index + first_count)
                self._completed_episode_lengths[first_slice] = completed_lengths[:first_count]
                self._completed_episode_timeouts[first_slice] = completed_timeouts[:first_count]

                remaining = num_completed - first_count
                if remaining > 0:
                    self._completed_episode_lengths[:remaining] = completed_lengths[first_count:]
                    self._completed_episode_timeouts[:remaining] = completed_timeouts[first_count:]

                self._episode_stats_write_index = (write_index + num_completed) % capacity
                self._episode_stats_count = min(capacity, self._episode_stats_count + num_completed)

        completed_metric_keys = (
            "episode/mean_length",
            "episode/mean_length_s",
            "episode/timeout_fraction",
            "episode/death_fraction",
            "episode/stats_window_count",
        )
        if self._episode_stats_count == 0:
            for key in completed_metric_keys:
                log.pop(key, None)
            return

        valid_slice = slice(None) if self._episode_stats_count == self._episode_stats_capacity else slice(
            0, self._episode_stats_count
        )
        mean_length = self._completed_episode_lengths[valid_slice].mean()
        timeout_fraction = self._completed_episode_timeouts[valid_slice].mean()
        log["episode/mean_length"] = mean_length.detach()
        log["episode/mean_length_s"] = (mean_length * self.step_dt).detach()
        log["episode/timeout_fraction"] = timeout_fraction.detach()
        log["episode/death_fraction"] = (1.0 - timeout_fraction).detach()
        log["episode/stats_window_count"] = torch.tensor(
            self._episode_stats_count, dtype=torch.float32, device=self.device
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # Reset rolling vx buffer for terminated envs (refill with large value for grace period)
        self._vel_window_buf[env_ids] = 1e3
        self._vel_window_count[env_ids] = 0

        # Reset action buffers so the first step after reset has no spurious penalty
        self.prev_actions[env_ids] = 0.0
        self.actions[env_ids] = 0.0

        if self.cfg.reset_strategy == "default":
            root_state, joint_pos, joint_vel = self._reset_strategy_default(env_ids)
        elif self.cfg.reset_strategy.startswith("random"):
            start = "start" in self.cfg.reset_strategy
            root_state, joint_pos, joint_vel = self._reset_strategy_random(env_ids, start)
        else:
            raise ValueError(f"Unknown reset strategy: {self.cfg.reset_strategy}")

        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # Reset strategies

    def _reset_strategy_default(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        return root_state, joint_pos, joint_vel

    def _reset_strategy_random(
        self, env_ids: torch.Tensor, start: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Sample random motion times
        num_samples = env_ids.shape[0]
        times = np.zeros(num_samples) if start else self._motion_loader.sample_times(num_samples)
        
        # Sample random motions
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)

        # Get root transforms (base_link)
        motion_base_index = self._motion_loader.get_body_index(["base_link"])[0]
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = body_positions[:, motion_base_index] + self.scene.env_origins[env_ids]
        #root_state[:, 2] += 0.3  # ✅ 추가 offset!
        
        # ✅ 디버깅: 높이 확인
        if self.common_step_counter % 1000 == 0 and len(env_ids) > 0:
            print(f"\n[Reset Debug]")
            print(f"  Motion height: {body_positions[:, motion_base_index, 2].mean():.3f}m")
            print(f"  Final height: {root_state[:, 2].mean():.3f}m")
            print(f"  Termination threshold: {self.cfg.termination_height:.3f}m")
        
        root_state[:, 2] += 0.0  # No additional offset
        root_state[:, 3:7] = body_rotations[:, motion_base_index]
        root_state[:, 7:10] = body_linear_velocities[:, motion_base_index]
        root_state[:, 10:13] = body_angular_velocities[:, motion_base_index]
        
        # Get DOFs state
        dof_pos = dof_positions[:, self.motion_dof_indexes]
        dof_vel = dof_velocities[:, self.motion_dof_indexes]

        # Update AMP observation
        amp_observations = self.collect_reference_motions(num_samples, times)
        self.amp_observation_buffer[env_ids] = amp_observations.view(num_samples, self.cfg.num_amp_observations, -1)

        return root_state, dof_pos, dof_vel

    # Env methods

    def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None) -> torch.Tensor:
        # Sample random motion times
        if current_times is None:
            current_times = self._motion_loader.sample_times(num_samples)

        # ✅ Use POLICY dt, not motion dt!
        policy_dt = self.cfg.sim.dt * self.cfg.decimation

        
        # times = (
        #     np.expand_dims(current_times, axis=-1)
        #     - self._motion_loader.dt * np.arange(0, self.cfg.num_amp_observations)
        # ).flatten()
        
        times = (
            np.expand_dims(current_times, axis=-1)
            - policy_dt * np.arange(0, self.cfg.num_amp_observations)
        ).flatten()


        # Get motions
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)
        
        # Compute AMP observation
        amp_observation = compute_obs(
            dof_positions[:, self.motion_dof_indexes],
            dof_velocities[:, self.motion_dof_indexes],
            body_positions[:, self.motion_ref_body_index],
            body_rotations[:, self.motion_ref_body_index],
            body_linear_velocities[:, self.motion_ref_body_index],
            body_angular_velocities[:, self.motion_ref_body_index],
            body_positions[:, self.motion_key_body_indexes],
        )
        return amp_observation.view(-1, self.amp_observation_size)


@torch.jit.script
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_apply(q, ref_tangent)
    normal = quat_apply(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)


@torch.jit.script
def compute_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_positions: torch.Tensor,
    root_rotations: torch.Tensor,
    root_linear_velocities: torch.Tensor,
    root_angular_velocities: torch.Tensor,
    key_body_positions: torch.Tensor,
) -> torch.Tensor:
    """
    Compute observation for Dextra AMP.
    
    Observation (43D):
    - dof_positions: 12
    - dof_velocities: 12
    - root_height: 1
    - root_tangent_normal: 6
    - root_linear_velocities: 3
    - root_angular_velocities: 3
    - key_body_positions (2 feet): 6
    
    Total: 12+12+1+6+3+3+6 = 43
    """
    obs = torch.cat(
        (
            dof_positions,
            dof_velocities,
            root_positions[:, 2:3],  # Root body height
            quaternion_to_tangent_and_normal(root_rotations),
            root_linear_velocities,
            root_angular_velocities,
            (key_body_positions - root_positions.unsqueeze(-2)).view(key_body_positions.shape[0], -1),
        ),
        dim=-1,
    )
    return obs
