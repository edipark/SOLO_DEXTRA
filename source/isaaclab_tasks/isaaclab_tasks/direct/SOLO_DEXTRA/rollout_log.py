"""Roll out AMP teacher policy with estimator and save deploy-ready CSV.

This script follows the same policy+estimator path as
`play_teacher_with_estimator.py`, but logs per-step targets in a CSV format
directly consumable by `SOLO_ws/replay_csv.py`.

Launch via::

    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/SOLO_DEXTRA/rollout_log.py \
        --teacher_checkpoint logs/skrl/dextra_amp_walk/task+amp/checkpoints/best_agent.pt \
        --estimator_checkpoint logs/solo_estimator/<run>/best_estimator.pt \
        --steps 600 \
        --output logs/rollout/rollout.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Roll out AMP teacher + estimator and save actions to CSV"
)
parser.add_argument(
    "--teacher_checkpoint", type=str, required=True, help="Path to SKRL AMP best_agent.pt"
)
parser.add_argument(
    "--estimator_checkpoint", type=str, required=True, help="Path to estimator checkpoint"
)
parser.add_argument("--steps", type=int, default=600, help="Number of env steps to roll out")
parser.add_argument("--output", type=str, required=True, help="Output CSV path")
parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument(
    "--log-env-id",
    type=int,
    default=0,
    help="Environment index to record into CSV (default: 0)",
)
parser.add_argument(
    "--gt-warmup",
    action="store_true",
    default=False,
    help="Debug: use GT privileged state during warm-up",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- heavy imports (after AppLauncher) ---

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401

from solo_models import ENCODER_DIM, OBS_DIM, PRIV_DIM, TeacherPolicy, load_estimator


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    teacher_path = os.path.abspath(args_cli.teacher_checkpoint)
    estimator_path = os.path.abspath(args_cli.estimator_checkpoint)
    output_csv = os.path.abspath(args_cli.output)

    from isaaclab_tasks.direct.SOLO_DEXTRA.dextra_amp_env_cfg import DextraAmpWalkEnvCfg

    env_cfg = DextraAmpWalkEnvCfg()
    env_cfg.scene.num_envs = max(1, int(args_cli.num_envs))
    env_cfg.use_fk_observations = False

    env = gym.make("Isaac-Dextra-Amp-Walk-Direct-v0", cfg=env_cfg)

    def unwrap_obs(obs):
        return obs["policy"] if isinstance(obs, dict) else obs

    raw_obs, _ = env.reset()
    obs = unwrap_obs(raw_obs)
    device = obs.device
    num_envs = obs.shape[0]

    teacher = TeacherPolicy(OBS_DIM, device=str(device))
    teacher.load_from_checkpoint(teacher_path, device=str(device))
    teacher.eval()

    estimator, est_ckpt = load_estimator(estimator_path, device=str(device))
    est_cfg = est_ckpt["estimator_config"]
    est_type = est_cfg["type"].upper()
    window = int(est_ckpt.get("window", 50))
    use_mlp = est_type == "MLP"

    hist_buf = torch.zeros((num_envs, window, ENCODER_DIM), device=device)
    hist_valid = torch.zeros(num_envs, device=device, dtype=torch.long)

    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    log_env_id = int(args_cli.log_env_id)
    if log_env_id < 0 or log_env_id >= num_envs:
        raise ValueError(
            f"--log-env-id must be in [0, {num_envs - 1}], got {log_env_id}"
        )

    action_dim = int(env_cfg.action_space)
    header = ["step", "env_id"] + [f"target_{i}" for i in range(action_dim)]

    rows_written = 0

    print(f"[rollout] device={device}, num_envs={num_envs}, steps={args_cli.steps}")
    print(f"[rollout] teacher={teacher_path}")
    print(f"[rollout] estimator={estimator_path} (type={est_type}, window={window})")
    print(f"[rollout] output={output_csv}")
    print(f"[rollout] logging only env_id={log_env_id} to CSV")

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["# generated_by=rollout_log.py", "joint_names="])
        writer.writerow(header)

        for step in range(int(args_cli.steps)):
            if not simulation_app.is_running():
                print("[rollout] simulation_app stopped early")
                break

            with torch.no_grad():
                enc = obs[:, :ENCODER_DIM]

                hist_buf = torch.roll(hist_buf, -1, dims=1)
                hist_buf[:, -1] = enc
                hist_valid = hist_valid + 1

                if use_mlp:
                    priv_est = estimator.predict_denormalized(enc)
                else:
                    priv_est = estimator.predict_denormalized(hist_buf)

                if args_cli.gt_warmup:
                    priv_gt = obs[:, ENCODER_DIM : ENCODER_DIM + PRIV_DIM]
                    ready = hist_valid >= window
                    priv_used = torch.where(ready.unsqueeze(-1), priv_est, priv_gt)
                else:
                    priv_used = priv_est

                obs_combined = torch.cat([enc, priv_used], dim=-1)
                action = teacher(obs_combined)

            action_cpu = action.detach().to(torch.float32).cpu()
            row = [step, log_env_id] + action_cpu[log_env_id].tolist()
            writer.writerow(row)
            rows_written += 1

            raw_obs, _, terminated, truncated, _ = env.step(action)
            obs = unwrap_obs(raw_obs)

            reset_mask = terminated | truncated
            if reset_mask.dim() > 1:
                reset_mask = reset_mask.squeeze(-1)
            if reset_mask.any():
                hist_buf[reset_mask] = 0
                hist_valid[reset_mask] = 0

    env.close()
    print(f"[rollout] done. wrote {rows_written} rows")


if __name__ == "__main__":
    main()
    simulation_app.close()
