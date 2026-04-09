# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import MISSING

from .dextra_robot_cfg import DEXTRA_CFG

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")


@configclass
class DextraEventCfg:
    """Domain randomization events for the Dextra robot."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (0.7, 1.3),
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
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )


@configclass
class DextraAmpEnvCfg(DirectRLEnvCfg):
    """Dextra AMP environment config (base class)."""
    use_fk_observations: bool = False  # --fk flag로 활성화


    # Episode
    #episode_length_s = 10.0
    episode_length_s = 10 # For much more stable reading, need to look at longer frames
    #decimation = 2  # 60Hz control (120Hz physics / 2)
    decimation = 2 # For stable control, conservative projections regarding control loop

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
    termination_height = 0.15  # Base link below 15cm

    # Task reward: world +X linear velocity tracking (see `_get_rewards` in dextra_amp_env.py).
    # Requires `task_reward_weight > 0` in `agents/skrl_amp_cfg.yaml` to affect learning.
    target_vel_x_world: float = 0.3           # m/s desired along world +X
    target_vel_tracking_coeff: float = 0.25   # exp(-coeff * (vx - target)^2); larger = sharper peak
    vel_reward_weight: float = 0.3            # weight within combined task reward

    # Foot-flat reward: penalizes feet tilting away from ground-parallel.
    # Foot local Z-axis (URDF up-axis) vs world Z dot product; 1.0 = perfectly flat.
    foot_flat_reward_weight: float = 0.7      # weight within combined task reward
    foot_flat_coeff: float = 10.0              # exp(-coeff * (1 - dot)^2)

    # Motion
    motion_file: str = os.path.join(MOTIONS_DIR, "dextra_walk_flat_pitch_fk.npz")
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
    robot: ArticulationCfg = DEXTRA_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*HipYaw.*", ".*HipRoll.*", ".*Thigh.*", ".*Calf.*", ".*Ankle.*"],
                stiffness=6.0,
                damping=0.6,
                effort_limit=1.8,
                velocity_limit=10.16,
                armature=0.00054,      # joint-space inertia에 추가되어 안정성 향상
                friction=0.05,         # 관절 정적 마찰 계수
                ),
        },
    )

    # Domain randomization
    events: DextraEventCfg = DextraEventCfg()


@configclass
class DextraAmpWalkEnvCfg(DextraAmpEnvCfg):
    """Dextra AMP Walk environment config."""
    motion_file = os.path.join(MOTIONS_DIR, "dextra_walk_flat_pitch_fk.npz")
    # motion_file = os.path.join(MOTIONS_DIR, "dextra_walk_flat_pitch_fk.npz")