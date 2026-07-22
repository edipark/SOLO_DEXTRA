# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import MISSING

import torch

from .dextra_robot_cfg import DEXTRA_CFG

from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from .actuators import AX18AActuator, AX18AActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")


def configure_joint_velocity_observation_noise(env, env_ids, noise_cfg: UniformNoiseCfg):
    """Register joint-velocity sensor noise without changing the simulated state."""
    del env_ids
    env._joint_velocity_observation_noise_cfg = noise_cfg


def randomize_ax18a_effort_limit(env, env_ids, asset_cfg: SceneEntityCfg):
    """Randomize one AX-18A effort-limit ratio per environment.

    Every AX-18A joint in a given environment receives the same sampled ratio,
    while different environments receive independent ratios. The cached punch
    torque is updated together with the effort limit so the compliance profile
    remains consistent with the hardware register model.
    """
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    else:
        env_ids = torch.as_tensor(env_ids, device=asset.device, dtype=torch.long)

    ratio_min, ratio_max = env.cfg.effort_limit_ratio_range
    if ratio_min <= 0.0 or ratio_max < ratio_min or ratio_max > 1.0:
        raise ValueError(
            "effort_limit_ratio_range must satisfy 0 < min <= max <= 1; "
            f"got {env.cfg.effort_limit_ratio_range}"
        )

    ratios = torch.empty((env_ids.numel(), 1), device=asset.device).uniform_(ratio_min, ratio_max)
    effort_limits = env.cfg.ax18a_stall_torque * ratios

    for actuator in asset.actuators.values():
        if not isinstance(actuator, AX18AActuator):
            continue
        actuator_effort_limits = effort_limits.expand(-1, actuator.num_joints)
        actuator.effort_limit[env_ids] = actuator_effort_limits
        actuator._punch_torque[env_ids] = (
            float(actuator.cfg.punch) / 1023.0 * actuator_effort_limits
        )

    # Keep the sampled ratios available for diagnostics and effort-bin eval.
    if not hasattr(env, "_ax18a_effort_limit_ratio"):
        env._ax18a_effort_limit_ratio = torch.full(
            (env.scene.num_envs,),
            float(env.cfg.effort_limit_ratio),
            device=asset.device,
        )
    env._ax18a_effort_limit_ratio[env_ids] = ratios.squeeze(-1)


@configclass
class DextraEventCfg:
    """Domain randomization events for the Dextra robot."""

    # AX-18A position resolution is about 0.0051 rad/tick. At the 30 Hz policy
    # rate, differentiating quantized positions produces velocity steps of about
    # 0.154 rad/s. Corrupt only the policy joint-velocity observation; the
    # simulated joint state and discriminator AMP observations stay noise-free.
    joint_velocity_observation_noise = EventTerm(
        func=configure_joint_velocity_observation_noise,
        mode="startup",
        params={
            "noise_cfg": UniformNoiseCfg(
                n_min=-0.15,
                n_max=0.15,
                operation="add",
            ),
        },
    )

    ax18a_effort_limit = EventTerm(
        func=randomize_ax18a_effort_limit,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.3),
            "dynamic_friction_range": (1.0, 1.2),
            "restitution_range": (0.0, 0.01),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    add_thigh_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*Thigh.*"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    add_distal_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*Calf.*|.*Ankle.*"),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (0.9, 1.3),
            "operation": "scale",
        },
    )

    joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "armature_distribution_params": (0.0005, 0.0006),
            "operation": "abs",
        },
    )

    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            # stiffness is a dummy field for AX18AActuator — skip randomization
            "damping_distribution_params": (0.8, 1.2),  # ±20% back-EMF variation
            "operation": "scale",
        },
    )


@configclass
class DextraAmpEnvCfg(DirectRLEnvCfg):
    """Dextra AMP environment config (base class)."""
    use_fk_observations: bool = False  # --fk flag로 활성화

    # AX-18A torque-limit model. The nominal ratio configures the actuator
    # before startup DR; the range samples one shared ratio per environment.
    ax18a_stall_torque: float = 1.8
    effort_limit_ratio: float = 0.3
    effort_limit_ratio_range: tuple[float, float] = (0.25, 0.35)

    # Episode
    #episode_length_s = 10.0
    episode_length_s = 20
    #decimation = 2  # 60Hz control (120Hz physics / 2)
    decimation = 4 # For stable control, conservative projections regarding control loop
    motion_speed_scale: float = 1.0            # Scale factor for motion playback speed (default: 1.0)


    # Spaces
    observation_space = 43  # 12 dof + 12 vel + 1 height + 6 quat + 3 lin_vel + 3 ang_vel + 6 feet
    #def observation_space(self):
    #    return 31 if self.use_fk_observations else 43
    action_space = 12       # 12 DOFs
    state_space = 0
    num_amp_observations = 4
    #num_amp_observations = 18 # Need much more due to 20hz observation
    amp_observation_space = 43  # Same as observation_space

    # Termination
    early_termination = True
    termination_height = 0.15   # Base link below 15cm → die
    termination_min_vel_x: float = 0.0  # Instantaneous vx threshold (0.0 = disabled)
    # termination_min_vel_x: float = 0.0  # Instantaneous vx threshold (disabled; relying on windowed velocity termination instead)

    vel_window_min_vx: float = 0.0   # m/s  (set 0.0 to disable)
    vel_window_steps: int = 4        # number of control steps to average over (~2s @ 30Hz)

    # Task reward: world +X linear velocity tracking (see `_get_rewards` in dextra_amp_env.py).
    # Requires `task_reward_weight > 0` in `agents/skrl_amp_cfg.yaml` to affect learning.
    target_vel_x_world: float = 0.1 / motion_speed_scale           # m/s desired along world +X
    target_vel_tracking_coeff: float = 200.0   # exp(-coeff * (vx - target)^2); larger = sharper peak
    vel_reward_weight: float = 0.5            # weight within combined task reward

    # Foot-flat reward: penalizes feet tilting away from ground-parallel.
    # Foot local Z-axis (URDF up-axis) vs world Z dot product; 1.0 = perfectly flat.
    foot_flat_reward_weight: float = 0.0     # weight within combined task reward
    foot_flat_coeff: float = 10.0              # exp(-coeff * (1 - dot)^2)

    # Action-rate penalty: penalizes rapid target changes between consecutive steps.
    # Reduces joint jittering. Penalty = mean(||a_t - a_{t-1}||^2) across joints.
    # Set > 0 to activate; start small (e.g. 0.01) and tune up.
    action_rate_penalty_weight: float = 0.5
    # Motion
    motion_file: str = os.path.join(MOTIONS_DIR, "dextra_walk_flat_pitch_fk_30hz_stride0p5_vel0p5.npz")  # FK motion file (see `motions/create_motion_variant.py`)
    reference_body = "base_link"
    reset_strategy = "default"  # default, random, random-start

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,  # 120Hz physics
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=3.0,
        replicate_physics=True
    )

    # Robot
    # -------------------------------------------------------------------------
    # AX18AActuator: Dynamixel AX-18A compliance model.
    # Compliance slope=64 → slope width ≈ 0.324 rad and effective stiffness ≈ 5.3 N·m/rad.
    # -------------------------------------------------------------------------
    robot: ArticulationCfg = DEXTRA_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        actuators={
            "legs": AX18AActuatorCfg(
                joint_names_expr=[".*HipYaw.*", ".*HipRoll.*", ".*Thigh.*", ".*Calf.*", ".*Ankle.*"],
                stall_torque=ax18a_stall_torque,  # AX-18A physical motor limit [N·m] (fixed, spec)
                effort_limit=ax18a_stall_torque * effort_limit_ratio,
                velocity_limit=10.16,    # AX-18A no-load speed [rad/s]
                damping=0.177,           # back-EMF: τ_stall / ω_no_load = 1.8 / 10.16 or 0.035
                armature=0.00054,        # rotor inertia reflected to joint [kg·m²]
                friction=0.0,           # PhysX static friction coefficient. set 0.0 to prevent discontinuities with the AX-18A compliance model
                compliance_slope=64.0,   # AX-18A register slope; deploy writes the same value on connect
            ),
        },
    )
    # -------------------------------------------------------------------------
    # [ImplicitActuator] PD handled by PhysX (continuous-time).
    # Faster training convergence; less accurate sim-to-real.
    # -------------------------------------------------------------------------
    # robot: ArticulationCfg = DEXTRA_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
    #     actuators={
    #         "legs": ImplicitActuatorCfg(
    #             joint_names_expr=[".*HipYaw.*", ".*HipRoll.*", ".*Thigh.*",
    #                               ".*Calf.*", ".*AnklePitch.*", ".*AnkleRoll.*"],
    #             stiffness=11.1,          # N·m/rad (AX-18A k at slope=32)
    #             damping=0.8,             # N·m·s/rad
    #             effort_limit=1.8,        # AX-18A stall torque [N·m]
    #             velocity_limit=10.16,    # AX-18A no-load speed [rad/s]
    #             armature=0.00054,        # joint-space inertia [kg·m²]
    #             friction=0.05,           # PhysX static friction coefficient
    #         ),
    #     },
    # )
    # -------------------------------------------------------------------------
    # [DCMotor] explicit PD + 4-quadrant torque-speed saturation curve.
    # Stiffness is ramped up via curriculum (Phase B→C3) to close sim-to-real
    # gap with the AX-18A hardware (factory k ≈ 11.1 N·m/rad at slope=32).
    # -------------------------------------------------------------------------
    # robot: ArticulationCfg = DEXTRA_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
    #     actuators={
    #         "legs": DCMotorCfg(
    #             joint_names_expr=[".*HipYaw.*", ".*HipRoll.*", ".*Thigh.*", ".*Calf.*", ".*Ankle.*"],
    #             stiffness=4.5,           # N·m/rad  (Phase B baseline; ramped via curriculum)
    #             damping=0.45,            # N·m·s/rad
    #             # AX-18A @ 12 V
    #             effort_limit=1.8,        # continuous/peak torque [N·m]
    #             saturation_effort=1.8,   # stall torque [N·m]  (DC motor 4-quadrant curve)
    #             velocity_limit=10.16,    # no-load speed [rad/s]
    #             armature=0.00054,        # rotor inertia reflected to joint [kg·m²]
    #             friction=0.05,           # PhysX static friction coefficient
    #         ),
    #     },
    # )

    # Domain randomization
    events: DextraEventCfg = DextraEventCfg()


@configclass
class DextraAmpWalkEnvCfg(DextraAmpEnvCfg):
    """Dextra AMP Walk environment config."""
    motion_file = os.path.join(MOTIONS_DIR, "dextra_walk_flat_pitch_fk_30hz_stride0p4_vel0p4.npz")
    # motion_file = os.path.join(MOTIONS_DIR, "dextra_walk_flat_pitch_fk.npz")