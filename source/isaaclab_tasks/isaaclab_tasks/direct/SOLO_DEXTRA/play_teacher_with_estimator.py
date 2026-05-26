"""Play AMP teacher policy with a trained state estimator.

Loads an AMP teacher checkpoint and a state estimator checkpoint, then
runs the teacher using the estimator's predicted privileged state instead
of ground truth.  With ``--video`` the run is saved as an mp4 file.

Launch via::

    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/SOLO_DEXTRA/play_teacher_with_estimator.py \
        --teacher_checkpoint logs/skrl/dextra_amp_walk/task+amp/checkpoints/best_agent.pt \
        --estimator_checkpoint logs/solo_estimator/LSTM_w50_seed42_.../best_estimator.pt \
        --num-envs 1 --headless --video --video-length 600

    # Debug: GT fallback during warm-up (not for deployment)
    ./isaaclab.sh -p ... --gt-warmup --headless --video --video-length 600
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Play AMP teacher + state estimator (no student)")
parser.add_argument("--teacher_checkpoint", type=str, required=True,
                    help="Path to SKRL AMP best_agent.pt")
parser.add_argument("--estimator_checkpoint", type=str, required=True,
                    help="Path to estimator checkpoint (best_estimator.pt)")
parser.add_argument("--num-envs", type=int, default=1,
                    help="Number of parallel environments")
parser.add_argument("--video", action="store_true", default=False,
                    help="Record video (mp4)")
parser.add_argument("--video-length", type=int, default=600,
                    help="Video length in env steps")
parser.add_argument("--video-dir", type=str, default=None,
                    help="Video output folder (auto if None)")
parser.add_argument("--real-time", action="store_true", default=False,
                    help="Run at real-time speed")
parser.add_argument("--gt-warmup", action="store_true", default=False,
                    help="Debug: use GT priv during warm-up (default: always estimator)")
parser.add_argument("--csv-output", type=str, default=None,
                    help="Optional CSV output path (deploy-compatible target_* format)")
parser.add_argument("--csv-log-env-id", type=int, default=0,
                    help="Env index to write to CSV when --csv-output is used")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=False)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- heavy imports (after AppLauncher) ---

import time

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401

from solo_models import (
    ENCODER_DIM, PRIV_DIM, OBS_DIM,
    TeacherPolicy, load_estimator,
)

POLICY_DT = (1.0 / 120.0) * 2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    teacher_path = os.path.abspath(args_cli.teacher_checkpoint)
    estimator_path = os.path.abspath(args_cli.estimator_checkpoint)
    csv_output = os.path.abspath(args_cli.csv_output) if args_cli.csv_output else None

    # --- environment ---
    from isaaclab_tasks.direct.SOLO_DEXTRA.dextra_amp_env_cfg import DextraAmpWalkEnvCfg

    env_cfg = DextraAmpWalkEnvCfg()
    env_cfg.scene.num_envs = getattr(args_cli, "num_envs", 1)
    env_cfg.use_fk_observations = False
    env_cfg.termination_min_vel_x = 0.0  # 속도 terminate 비활성화 (estimator warm-up 보호)
    env_cfg.vel_window_min_vx = 0.0 

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make("Isaac-Dextra-Amp-Walk-Direct-v0", cfg=env_cfg, render_mode=render_mode)

    try:
        dt = env.unwrapped.step_dt
    except AttributeError:
        dt = POLICY_DT

    # --- video wrapper ---
    if args_cli.video:
        video_dir = args_cli.video_dir
        if video_dir is None:
            video_dir = os.path.join(os.path.dirname(estimator_path), "videos")
        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[play] Recording video to {video_dir}  ({args_cli.video_length} steps)")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # --- reset once to discover the actual CUDA device ---
    def unwrap_obs(obs):
        return obs["policy"] if isinstance(obs, dict) else obs

    raw_obs, _ = env.reset()
    obs = unwrap_obs(raw_obs)
    device = obs.device
    num_envs = obs.shape[0]
    print(f"[play] Environment device: {device}, num_envs: {num_envs}")

    csv_writer = None
    csv_file = None
    if csv_output is not None:
        csv_dir = os.path.dirname(csv_output)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)

        csv_log_env_id = int(args_cli.csv_log_env_id)
        if csv_log_env_id < 0 or csv_log_env_id >= num_envs:
            raise ValueError(f"--csv-log-env-id must be in [0, {num_envs - 1}], got {csv_log_env_id}")

        csv_file = open(csv_output, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["# generated_by=play_teacher_with_estimator.py", "joint_names="])
        csv_writer.writerow(["step", "env_id"] + [f"target_{i}" for i in range(12)])
        print(f"[play] CSV logging enabled: {csv_output} (env_id={csv_log_env_id})")

    # --- load teacher ---
    teacher = TeacherPolicy(OBS_DIM, device=str(device))
    teacher.load_from_checkpoint(teacher_path, device=str(device))
    teacher.eval()
    print(f"[play] Teacher loaded: {teacher_path}")

    # --- load estimator ---
    estimator, est_ckpt = load_estimator(estimator_path, device=str(device))
    est_cfg = est_ckpt["estimator_config"]
    est_type = est_cfg["type"].upper()
    window = est_ckpt.get("window", 50)
    use_mlp = est_type == "MLP"
    print(f"[play] Estimator loaded: {est_type}, window={window}")
    if args_cli.gt_warmup:
        print("[play] Debug: GT priv fallback during warm-up")
    else:
        print("[play] Always using estimator (including warm-up)")

    # --- history buffer ---
    hist_buf = torch.zeros((num_envs, window, ENCODER_DIM), device=device)
    hist_valid = torch.zeros(num_envs, device=device, dtype=torch.long)

    # --- play loop ---
    print(f"[play] Running AMP teacher + estimator (num_envs={num_envs})…")
    timestep = 0

    while simulation_app.is_running():
        t0 = time.time()

        with torch.no_grad():
            enc = obs[:, :ENCODER_DIM]

            # update history
            hist_buf = torch.roll(hist_buf, -1, dims=1)
            hist_buf[:, -1] = enc
            hist_valid = hist_valid + 1

            # estimator prediction (always, including warm-up with zero-padded history)
            if use_mlp:
                priv_est = estimator.predict_denormalized(enc)
            else:
                priv_est = estimator.predict_denormalized(hist_buf)

            if args_cli.gt_warmup:
                # debug: GT fallback during warm-up
                priv_gt = obs[:, ENCODER_DIM:ENCODER_DIM + PRIV_DIM]
                ready = hist_valid >= window
                priv_used = torch.where(ready.unsqueeze(-1), priv_est, priv_gt)
            else:
                priv_used = priv_est

            obs_combined = torch.cat([enc, priv_used], dim=-1)
            action = teacher(obs_combined)

        if csv_writer is not None:
            action_cpu = action.detach().to(torch.float32).cpu()
            csv_writer.writerow([timestep, csv_log_env_id] + action_cpu[csv_log_env_id].tolist())

        raw_obs, _, terminated, truncated, _ = env.step(action)
        obs = unwrap_obs(raw_obs)

        reset_mask = (terminated | truncated).squeeze(-1) if terminated.dim() > 1 else (terminated | truncated)
        if reset_mask.any():
            with torch.inference_mode():
                hist_buf[reset_mask] = 0
                hist_valid[reset_mask] = 0

        timestep += 1

        if args_cli.video and timestep >= args_cli.video_length:
            break

        if args_cli.real_time:
            sleep = dt - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)

    env.close()
    if csv_file is not None:
        csv_file.close()
    print(f"[play] Done. {timestep} steps executed.")
    if args_cli.video:
        print(f"[play] Video saved to: {video_dir}")
    if csv_output is not None:
        print(f"[play] CSV saved to: {csv_output}")


if __name__ == "__main__":
    main()
    simulation_app.close()
