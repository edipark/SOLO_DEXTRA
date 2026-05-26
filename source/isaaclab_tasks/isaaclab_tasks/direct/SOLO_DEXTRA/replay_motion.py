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

    # Record video (saves to videos/ next to the motion file):
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/SOLO_DEXTRA/replay_motion.py \
        --file source/isaaclab_tasks/isaaclab_tasks/direct/SOLO_DEXTRA/motions/dextra_walk_flat_pitch_fk_30hz_2x_slower_stride0p6_vel0p8_symright_periodic.npz --headless --video --video-length 600
    ./isaaclab.sh -p ... --file ... --headless --video --video-length 600 --video-dir /tmp/replay_out
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay Dextra motion in Isaac Lab.")
parser.add_argument("--file", type=str, required=True, help="Path to motion .npz file")
parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
parser.add_argument("--video", action="store_true", default=False, help="Record video (mp4).")
parser.add_argument("--video-length", type=int, default=600, help="Video length in physics steps.")
parser.add_argument("--video-dir", type=str, default=None, help="Output folder for video (auto if None).")
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

# Enable cameras before AppLauncher (matches play_teacher_with_estimator pattern)
if args_cli.video:
    args_cli.enable_cameras = True

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


def _get_rgb_frame(rgb_annotator) -> "np.ndarray | None":
    """Extract H×W×3 uint8 RGB array from annotator.

    Isaac Sim 4.x: get_data() returns a dict {"data": ndarray, ...}.
    Older versions: returns ndarray directly.
    """
    raw = rgb_annotator.get_data()
    if raw is None:
        return None
    frame = raw.get("data") if isinstance(raw, dict) else raw
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return None
    # Keep only RGB channels (drop alpha if RGBA)
    if frame.ndim == 3 and frame.shape[2] >= 3:
        return np.ascontiguousarray(frame[:, :, :3])
    return None


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


def run_replay(
    sim: SimulationContext,
    robot: Articulation,
    motion: MotionLoader,
    speed: float,
    video_writer=None,
    rgb_annotator=None,
):
    sim_dt = sim.get_physics_dt()
    print_vel = args_cli.print_base_velocity
    print_vel_every = max(1, int(args_cli.print_base_velocity_interval))
    step_idx = 0

    # Map motion DOFs to robot DOFs
    motion_dof_indexes = motion.get_dof_index(robot.data.joint_names)
    motion_base_index = motion.get_body_index(["base_link"])[0]

    # Capture one video frame every N physics steps to match 30 fps
    VIDEO_FPS = 30
    capture_every = max(1, round(1.0 / (sim_dt * VIDEO_FPS)))

    print(f"\n{'='*60}")
    print(f"  Motion Replay")
    print(f"  File: {args_cli.file}")
    print(f"  Duration: {motion.duration:.2f}s  |  Frames: {motion.num_frames}")
    print(f"  Speed: {speed:.1f}x  |  Sim dt: {sim_dt:.4f}s")
    print(f"  Robot joints: {robot.data.joint_names}")
    print(f"  Motion DOFs:  {motion.dof_names}")
    print(f"  Bodies: {motion.body_names}")
    if args_cli.video:
        print(f"  Video: {args_cli.video_length} steps  (capture every {capture_every} steps)")
    print(f"{'='*60}\n")

    current_time = 0.0

    while simulation_app.is_running():
        # Stop when video length is reached
        if args_cli.video and step_idx >= args_cli.video_length:
            print(f"[INFO] Reached video-length ({args_cli.video_length} steps). Stopping.")
            break

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

        # Capture video frame at VIDEO_FPS rate
        if video_writer is not None and rgb_annotator is not None:
            if step_idx % capture_every == 0:
                frame = _get_rgb_frame(rgb_annotator)
                if frame is not None:
                    video_writer.append_data(frame)

        if print_vel and step_idx % print_vel_every == 0:
            v_cmd = root_state[0, 7:10].cpu().numpy()
            w_cmd = root_state[0, 10:13].cpu().numpy()
            v_sim = robot.data.root_lin_vel_w[0].cpu().numpy()
            w_sim = robot.data.root_ang_vel_w[0].cpu().numpy()
            print(
                f"[t={current_time:.3f}s step={step_idx}] "
                f"motion->sim v=({v_cmd[0]:+.4f}, {v_cmd[1]:+.4f}, {v_cmd[2]:+.4f}) m/s  "
                f"w=({w_cmd[0]:+.4f}, {w_cmd[1]:+.4f}, {w_cmd[2]:+.4f}) rad/s  |  "
                f"sim v=({v_sim[0]:+.4f}, {v_sim[1]:+.4f}, {v_sim[2]:+.4f}) m/s  "
                f"w=({w_sim[0]:+.4f}, {w_sim[1]:+.4f}, {w_sim[2]:+.4f}) rad/s"
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

    # --- Video setup (matches play_teacher_with_estimator pattern) ---
    video_writer = None
    rgb_annotator = None
    render_product = None
    rep_mod = None
    if args_cli.video:
        import imageio
        import pathlib
        import omni.replicator.core as rep_mod

        render_product = rep_mod.create.render_product("/OmniverseKit_Persp", (1280, 720))
        rgb_annotator = rep_mod.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        rgb_annotator.attach([render_product])

        video_dir = args_cli.video_dir
        if video_dir is None:
            video_dir = str(pathlib.Path(args_cli.file).parent / "videos")
        os.makedirs(video_dir, exist_ok=True)

        stem = pathlib.Path(args_cli.file).stem
        video_path = os.path.join(video_dir, f"{stem}.mp4")
        video_writer = imageio.get_writer(video_path, fps=30, codec="libx264", quality=8)
        print(f"[INFO] Recording video to {video_dir}  ({args_cli.video_length} steps)")
    # ---

    if args_cli.matplotlib:
        import threading
        import matplotlib
        matplotlib.use("TkAgg")
        sys.path.insert(0, os.path.join(SCRIPT_DIR, "motions"))
        from motion_viewer import MotionViewer
        viewer = MotionViewer(args_cli.file, render_scene=True)
        t = threading.Thread(target=viewer.show, daemon=True)
        t.start()

    run_replay(sim, robot, motion, args_cli.speed, video_writer=video_writer, rgb_annotator=rgb_annotator)

    # --- Cleanup: replicator must be destroyed before video writer is closed ---
    if rgb_annotator is not None and render_product is not None:
        try:
            rgb_annotator.detach([render_product])
        except Exception:
            pass
    if rep_mod is not None and render_product is not None:
        try:
            rep_mod.destroy.render_product(render_product)
        except Exception:
            pass

    if video_writer is not None:
        video_writer.close()
        print(f"[INFO] Video saved to: {video_path}")
    # ---


if __name__ == "__main__":
    import threading
    main()
    # simulation_app.close() can hang indefinitely on Omniverse render threads.
    # Run it in a daemon thread with a timeout, then force-exit regardless.
    _close_thread = threading.Thread(target=simulation_app.close, daemon=True)
    _close_thread.start()
    _close_thread.join(timeout=15.0)
    os._exit(0)
