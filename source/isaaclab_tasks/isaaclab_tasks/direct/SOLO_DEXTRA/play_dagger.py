"""Play (and optionally record video of) a DAgger student policy.

Loads a student checkpoint produced by ``train_dagger.py`` and runs it in
the simulator.  With ``--video`` the run is saved as an mp4 file.

Launch via::

    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/SOLO_DEXTRA/play_dagger.py \
        --checkpoint logs/dagger/dextra/<run>/checkpoints/student_latest.pt \
        --num-envs 1 --headless --video --video-length 600
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a DAgger student policy checkpoint")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to student_*.pt checkpoint")
parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--video", action="store_true", default=False, help="Record video (mp4)")
parser.add_argument("--video-length", type=int, default=600, help="Video length in env steps")
parser.add_argument("--video-dir", type=str, default=None, help="Video output folder (auto if None)")
parser.add_argument("--real-time", action="store_true", default=False, help="Run at real-time speed")
parser.add_argument("--vel-clip", type=float, default=10.0, help="Clip numerical velocity (rad/s)")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- heavy imports (after AppLauncher) ---

import os
import time

import gymnasium as gym
import torch
import torch.nn as nn

import isaaclab_tasks  # noqa: F401

STUDENT_OBS_DIM = 24
ACTION_DIM = 12
JOINT_DIM = 12
POLICY_DT = (1.0 / 120.0) * 2


# ---------------------------------------------------------------------------
# Duplicated from train_dagger.py so this script is self-contained.
# ---------------------------------------------------------------------------

class RunningNormalizer:
    def __init__(self, dim: int, device: str, clip: float = 5.0, eps: float = 1e-6) -> None:
        self.mean = torch.zeros(dim, device=device, dtype=torch.float64)
        self.var = torch.ones(dim, device=device, dtype=torch.float64)
        self.count = torch.tensor(1e-4, device=device, dtype=torch.float64)
        self.clip = clip
        self.eps = eps

    @torch.no_grad()
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.float()
        std = self.var.float().sqrt().clamp(min=self.eps)
        return ((x - mean) / std).clamp(-self.clip, self.clip)

    @torch.no_grad()
    def denormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        mean = self.mean.float()
        std = self.var.float().sqrt().clamp(min=self.eps)
        return x_norm * std + mean

    def load_state_dict(self, d: dict) -> None:
        self.mean = d["mean"].to(self.mean.device)
        self.var = d["var"].to(self.var.device)
        self.count = d["count"].to(self.count.device)


class StudentPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int = STUDENT_OBS_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dims: tuple[int, ...] = (256, 256, 128),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ELU())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ckpt_path = os.path.abspath(args_cli.checkpoint)

    # --- environment (create first to discover the actual device) ---
    from isaaclab_tasks.direct.SOLO_DEXTRA.dextra_amp_env_cfg import DextraAmpWalkEnvCfg

    env_cfg = DextraAmpWalkEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.use_fk_observations = False

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
            video_dir = os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), "videos")
        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[play] Recording video to {video_dir}  ({args_cli.video_length} steps)")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # --- reset once to discover the actual CUDA device the env lives on ---
    def unwrap_obs(obs):
        return obs["policy"] if isinstance(obs, dict) else obs

    raw_obs, _ = env.reset()
    obs = unwrap_obs(raw_obs)
    device = obs.device
    print(f"[play] Environment device: {device}")

    # --- load checkpoint onto the same device ---
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    print(f"[play] Loaded checkpoint: {ckpt_path}")
    print(f"[play]   iteration={ckpt.get('iteration', '?')}  beta={ckpt.get('beta', '?')}")

    student = StudentPolicy().to(device)
    student.load_state_dict(ckpt["student_state_dict"])
    student.eval()

    obs_normalizer = RunningNormalizer(STUDENT_OBS_DIM, str(device))
    if "obs_normalizer" in ckpt:
        obs_normalizer.load_state_dict(ckpt["obs_normalizer"])
        print("[play]   obs_normalizer loaded")
    else:
        print("[play]   WARNING: no obs_normalizer in checkpoint — using identity")

    action_normalizer = RunningNormalizer(ACTION_DIM, str(device), clip=10.0)
    if "action_normalizer" in ckpt:
        action_normalizer.load_state_dict(ckpt["action_normalizer"])
        print("[play]   action_normalizer loaded")
    else:
        print("[play]   WARNING: no action_normalizer in checkpoint — student outputs raw actions")

    # --- play loop ---
    prev_joint_pos = obs[:, :JOINT_DIM].clone()

    print(f"[play] Running student policy (num_envs={args_cli.num_envs})…")
    timestep = 0

    while simulation_app.is_running():
        t0 = time.time()

        with torch.inference_mode():
            obs_43d = obs.float()
            joint_pos = obs_43d[:, :JOINT_DIM]
            joint_vel_fd = ((joint_pos - prev_joint_pos) / POLICY_DT).clamp(
                -args_cli.vel_clip, args_cli.vel_clip
            )
            prev_joint_pos = joint_pos.clone()

            obs_24d = torch.cat([joint_pos, joint_vel_fd], dim=-1)
            obs_norm = obs_normalizer.normalize(obs_24d)
            action_norm = student(obs_norm)
            action = action_normalizer.denormalize(action_norm)

        raw_obs, _, terminated, truncated, _ = env.step(action)
        obs = unwrap_obs(raw_obs)

        reset_mask = terminated | truncated
        if reset_mask.any():
            # `prev_joint_pos` may be an inference tensor, so avoid in-place writes.
            keep = ~reset_mask.unsqueeze(-1)
            prev_joint_pos = torch.where(keep, prev_joint_pos, obs[:, :JOINT_DIM])

        timestep += 1

        if args_cli.video and timestep >= args_cli.video_length:
            break

        if args_cli.real_time:
            sleep = dt - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)

    env.close()
    print(f"[play] Done. {timestep} steps executed.")
    if args_cli.video:
        print(f"[play] Video saved to: {video_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
