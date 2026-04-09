"""Replay Dextra motion NPZ files on the robot in Isaac Lab simulator.

Usage (from IsaacLab root):
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/SOLO_DEXTRA/replay_motion.py \
        --file source/isaaclab_tasks/isaaclab_tasks/direct/SOLO_DEXTRA/motions/dextra_walk_flat_pitch_fk.npz

    # Slow-motion (0.5x speed):
    ./isaaclab.sh -p ... --file ... --speed 0.5

    # Side-by-side with matplotlib skeleton:
    ./isaaclab.sh -p ... --file ... --matplotlib

    # Print base velocity (motion command vs sim) every 60 physics steps:
    ./isaaclab.sh -p ... --file ... --print-base-velocity --print-base-velocity-interval 60
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay Dextra motion in Isaac Lab.")
parser.add_argument("--file", type=str, required=True, help="Path to motion .npz file")
parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
parser.add_argument("--matplotlib", action="store_true", help="Also show matplotlib skeleton viewer")
parser.add_argument(
    "--print-base-velocity",
    action="store_true",
    default=False,
    help="Print base linear/angular velocity (world frame) each N sim steps.",
)
parser.add_argument(
    "--print-base-velocity-interval",
    type=int,
    default=30,
    help="With --print-base-velocity, print every N physics steps (default: 30).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Isaac Lab imports (must come after AppLauncher) ---

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext

# Robot config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(SCRIPT_DIR, "assets")
sys.path.insert(0, os.path.join(SCRIPT_DIR, "motions"))
from motion_loader import MotionLoader


DEXTRA_REPLAY_CFG = ArticulationCfg(
    prim_path="/World/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=os.path.join(ASSETS_DIR, "Dextra_lowerbody.urdf"),
        fix_base=False,
        merge_fixed_joints=False,
        self_collision=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=200.0,
                damping=20.0,
            ),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.2865),
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=200.0,
            damping=20.0,
            effort_limit=1.8,
            velocity_limit=10.0,
        ),
    },
)


def design_scene() -> dict:
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    robot = Articulation(cfg=DEXTRA_REPLAY_CFG)
    return {"robot": robot}


def run_replay(sim: SimulationContext, robot: Articulation, motion: MotionLoader, speed: float):
    sim_dt = sim.get_physics_dt()
    print_vel = args_cli.print_base_velocity
    print_vel_every = max(1, int(args_cli.print_base_velocity_interval))
    step_idx = 0

    # Map motion DOFs to robot DOFs
    motion_dof_indexes = motion.get_dof_index(robot.data.joint_names)
    motion_base_index = motion.get_body_index(["base_link"])[0]

    print(f"\n{'='*60}")
    print(f"  Motion Replay")
    print(f"  File: {args_cli.file}")
    print(f"  Duration: {motion.duration:.2f}s  |  Frames: {motion.num_frames}")
    print(f"  Speed: {speed:.1f}x  |  Sim dt: {sim_dt:.4f}s")
    print(f"  Robot joints: {robot.data.joint_names}")
    print(f"  Motion DOFs:  {motion.dof_names}")
    print(f"  Bodies: {motion.body_names}")
    print(f"{'='*60}\n")

    current_time = 0.0

    while simulation_app.is_running():
        # Wrap around
        if current_time > motion.duration:
            current_time = 0.0

        times = np.array([current_time])
        dof_pos, dof_vel, body_pos, body_rot, body_lin_vel, body_ang_vel = motion.sample(
            num_samples=1, times=times
        )

        # Set root pose (base_link world position + orientation)
        root_state = robot.data.default_root_state.clone()
        root_state[:, 0:3] = body_pos[:, motion_base_index]
        root_state[:, 3:7] = body_rot[:, motion_base_index]
        root_state[:, 7:10] = body_lin_vel[:, motion_base_index] * speed
        root_state[:, 10:13] = body_ang_vel[:, motion_base_index] * speed

        robot.write_root_link_pose_to_sim(root_state[:, :7])
        robot.write_root_com_velocity_to_sim(root_state[:, 7:])

        # Set joint positions
        joint_pos = dof_pos[:, motion_dof_indexes]
        joint_vel = dof_vel[:, motion_dof_indexes] * speed
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)

        if print_vel and step_idx % print_vel_every == 0:
            v_cmd = root_state[0, 7:10].cpu().numpy()
            w_cmd = root_state[0, 10:13].cpu().numpy()
            v_sim = robot.data.root_lin_vel_w[0].cpu().numpy()
            w_sim = robot.data.root_ang_vel_w[0].cpu().numpy()
            print(
                f"[t={current_time:.3f}s step={step_idx}] "
                f"motion→sim v=({v_cmd[0]:+.4f}, {v_cmd[1]:+.4f}, {v_cmd[2]:+.4f}) m/s  "
                f"ω=({w_cmd[0]:+.4f}, {w_cmd[1]:+.4f}, {w_cmd[2]:+.4f}) rad/s  |  "
                f"sim v=({v_sim[0]:+.4f}, {v_sim[1]:+.4f}, {v_sim[2]:+.4f}) m/s  "
                f"ω=({w_sim[0]:+.4f}, {w_sim[1]:+.4f}, {w_sim[2]:+.4f}) rad/s"
            )

        step_idx += 1
        current_time += sim_dt * speed


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.5, 1.5, 1.0], target=[0.0, 0.0, 0.25])

    scene = design_scene()
    robot = scene["robot"]

    sim.reset()
    print("[INFO] Scene ready, loading motion...")

    motion = MotionLoader(motion_file=args_cli.file, device=sim.device)

    if args_cli.matplotlib:
        import threading
        import matplotlib
        matplotlib.use("TkAgg")
        sys.path.insert(0, os.path.join(SCRIPT_DIR, "motions"))
        from motion_viewer import MotionViewer
        viewer = MotionViewer(args_cli.file, render_scene=True)
        t = threading.Thread(target=viewer.show, daemon=True)
        t.start()

    run_replay(sim, robot, motion, args_cli.speed)


if __name__ == "__main__":
    main()
    simulation_app.close()
