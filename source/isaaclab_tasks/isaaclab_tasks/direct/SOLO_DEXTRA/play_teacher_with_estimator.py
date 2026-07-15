"""Play AMP teacher policy with a trained state estimator.

Loads an AMP teacher checkpoint and a state estimator checkpoint, then
runs the teacher using the estimator's predicted privileged state instead
of ground truth.  With ``--video`` the run is saved as an mp4 file.

Supports two workflows:

1) Rollout + action logging:
    Save normalized policy actions to an ``.npz`` file.
2) Action replay:
    Skip teacher/estimator inference and replay actions from a logged file.

Launch via::

    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/SOLO_DEXTRA/play_teacher_with_estimator.py \
        --teacher_checkpoint logs/skrl/dextra_amp_walk/task+amp/checkpoints/best_agent.pt \
        --estimator_checkpoint logs/solo_estimator/LSTM_w50_seed42_.../best_estimator.pt \
        --num-envs 1 --headless --video --video-length 600

    # Debug: GT fallback during warm-up (not for deployment)
    ./isaaclab.sh -p ... --gt-warmup --headless --video --video-length 600

    # Rollout + action log
    ./isaaclab.sh -p ... \
        --teacher_checkpoint <teacher.pt> \
        --estimator_checkpoint <estimator.pt> \
        --action-log-output logs/rollout/actions_run01.npz \
        --video --video-length 600

    # Replay previously logged actions only
    ./isaaclab.sh -p ... \
        --replay-action-log logs/rollout/actions_run01.npz \
        --video --video-length 600
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Play AMP teacher + state estimator (no student)")
parser.add_argument("--teacher_checkpoint", type=str, default=None,
                    help="Path to SKRL AMP best_agent.pt")
parser.add_argument("--estimator_checkpoint", type=str, default=None,
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
                    help="Optional CSV output path (logs normalized action_* values)")
parser.add_argument("--csv-log-env-id", type=int, default=0,
                    help="Env index to write to CSV when --csv-output is used")
parser.add_argument("--action-log-output", type=str, default=None,
                    help="Optional .npz path to save normalized actions (recommended)")
parser.add_argument("--action-log-env-id", type=int, default=0,
                    help="Env index to save into --action-log-output")
parser.add_argument("--replay-action-log", type=str, default=None,
                    help="Replay mode: path to .npz action log created by --action-log-output")
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
    AX18A_RAD_PER_TICK, ENCODER_DIM, PRIV_DIM, OBS_DIM,
    TeacherPolicy, load_estimator,
)

POLICY_DT = (1.0 / 120.0) * 4  # sim dt * decimation = 1/30 s
EMA_ALPHA = 0.2  # Must match estimator training and SOLO_HW/config.yaml


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    replay_mode = args_cli.replay_action_log is not None
    if replay_mode:
        if args_cli.teacher_checkpoint or args_cli.estimator_checkpoint:
            print("[play] replay mode enabled: teacher/estimator checkpoints are ignored")
        teacher_path = None
        estimator_path = None
        replay_action_path = os.path.abspath(args_cli.replay_action_log)
    else:
        if args_cli.teacher_checkpoint is None or args_cli.estimator_checkpoint is None:
            raise ValueError(
                "Non-replay mode requires --teacher_checkpoint and --estimator_checkpoint. "
                "Or use --replay-action-log."
            )
        teacher_path = os.path.abspath(args_cli.teacher_checkpoint)
        estimator_path = os.path.abspath(args_cli.estimator_checkpoint)
        replay_action_path = None

    csv_output = os.path.abspath(args_cli.csv_output) if args_cli.csv_output else None
    action_log_output = os.path.abspath(args_cli.action_log_output) if args_cli.action_log_output else None

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
            if replay_mode:
                video_dir = os.path.join(os.path.dirname(replay_action_path), "videos")
            else:
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
        action_dim = int(env_cfg.action_space)
        csv_writer.writerow(["# generated_by=play_teacher_with_estimator.py", "joint_names="])
        csv_writer.writerow(["step", "env_id"] + [f"action_{i}" for i in range(action_dim)])
        print(f"[play] CSV logging enabled: {csv_output} (env_id={csv_log_env_id})")

    action_log_records = []
    action_log_env_id = int(args_cli.action_log_env_id)
    if action_log_output is not None:
        if action_log_env_id < 0 or action_log_env_id >= num_envs:
            raise ValueError(f"--action-log-env-id must be in [0, {num_envs - 1}], got {action_log_env_id}")
        action_log_dir = os.path.dirname(action_log_output)
        if action_log_dir:
            os.makedirs(action_log_dir, exist_ok=True)
        print(f"[play] Action log enabled: {action_log_output} (env_id={action_log_env_id})")

    if replay_mode:
        replay_data = np.load(replay_action_path)
        if "actions" not in replay_data:
            raise ValueError(f"Replay file missing 'actions' array: {replay_action_path}")
        replay_actions = replay_data["actions"].astype(np.float32)
        if replay_actions.ndim != 2:
            raise ValueError(
                f"Replay actions must have shape [T, action_dim], got {replay_actions.shape}"
            )
        replay_steps = replay_actions.shape[0]
        replay_action_dim = replay_actions.shape[1]
        action_dim = int(env_cfg.action_space)
        if replay_action_dim != action_dim:
            raise ValueError(
                f"Replay action dim mismatch: file={replay_action_dim}, env={action_dim}"
            )
        replay_actions_torch = torch.from_numpy(replay_actions).to(device=device)
        print(f"[play] Replay mode: {replay_action_path}")
        print(f"[play] Replay actions: steps={replay_steps}, action_dim={replay_action_dim}")
        if args_cli.gt_warmup:
            print("[play] note: --gt-warmup ignored in replay mode")
    else:
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

        # Hardware-accurate joint-velocity pipeline state. The estimator was
        # trained with AX-18A position quantization, finite differences at the
        # 30 Hz policy rate, and an EMA velocity filter.
        prev_pos_q = None
        ema_vel = torch.zeros((num_envs, 12), device=device)

    # --- play loop ---
    print(f"[play] Running AMP teacher + estimator (num_envs={num_envs})…")
    timestep = 0

    while simulation_app.is_running():
        t0 = time.time()

        with torch.no_grad():
            if replay_mode:
                if timestep >= replay_steps:
                    break
                action = replay_actions_torch[timestep].unsqueeze(0).repeat(num_envs, 1)
            else:
                joint_pos = obs[:, :12]
                pos_q = (joint_pos / AX18A_RAD_PER_TICK).round() * AX18A_RAD_PER_TICK
                if prev_pos_q is None:
                    raw_vel = torch.zeros_like(joint_pos)
                else:
                    raw_vel = (pos_q - prev_pos_q) / POLICY_DT
                ema_vel = EMA_ALPHA * raw_vel + (1.0 - EMA_ALPHA) * ema_vel
                prev_pos_q = pos_q
                enc = torch.cat([joint_pos, ema_vel], dim=-1)

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

        # Clamp to [-1, 1] — matches what env._apply_action() actually sends to physics.
        # Log and replay both use the clamped value so sim logs == HW replay targets.
        action_clamped = action.clamp(-1.0, 1.0)

        if csv_writer is not None:
            action_cpu = action_clamped.detach().to(torch.float32).cpu()
            csv_writer.writerow([timestep, csv_log_env_id] + action_cpu[csv_log_env_id].tolist())
        if action_log_output is not None:
            action_cpu = action_clamped.detach().to(torch.float32).cpu()
            action_log_records.append(action_cpu[action_log_env_id].numpy().copy())

        raw_obs, _, terminated, truncated, _ = env.step(action)
        obs = unwrap_obs(raw_obs)

        reset_mask = (terminated | truncated).squeeze(-1) if terminated.dim() > 1 else (terminated | truncated)
        if (not replay_mode) and reset_mask.any():
            with torch.inference_mode():
                hist_buf[reset_mask] = 0
                hist_valid[reset_mask] = 0
                new_pos = obs[reset_mask, :12]
                prev_pos_q[reset_mask] = (
                    new_pos / AX18A_RAD_PER_TICK
                ).round() * AX18A_RAD_PER_TICK
                ema_vel[reset_mask] = 0.0

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
    if action_log_output is not None:
        if len(action_log_records) == 0:
            print("[play] WARNING: action log requested but no actions were recorded")
        else:
            actions_np = np.asarray(action_log_records, dtype=np.float32)
            np.savez_compressed(
                action_log_output,
                actions=actions_np,
                step_dt=np.float32(dt),
                source="play_teacher_with_estimator.py",
                replay_mode=np.bool_(replay_mode),
                teacher_checkpoint="" if teacher_path is None else teacher_path,
                estimator_checkpoint="" if estimator_path is None else estimator_path,
                env_id=np.int32(action_log_env_id),
            )
            print(f"[play] Action log saved to: {action_log_output}  (steps={actions_np.shape[0]})")
    print(f"[play] Done. {timestep} steps executed.")
    if args_cli.video:
        print(f"[play] Video saved to: {video_dir}")
    if csv_output is not None:
        print(f"[play] CSV saved to: {csv_output}")


if __name__ == "__main__":
    main()
    simulation_app.close()
