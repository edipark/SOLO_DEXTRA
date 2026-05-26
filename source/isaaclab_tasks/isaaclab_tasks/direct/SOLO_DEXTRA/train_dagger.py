"""DAgger policy distillation for Dextra.

Trains a 24-D student policy (joint_pos + joint_vel) by imitating a 43-D
privileged teacher policy (SKRL AMP checkpoint) using the DAgger algorithm.

Joint velocity is computed via finite differencing of consecutive joint
positions, matching the real robot where only encoders are available.

Launch via::

    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/SOLO_DEXTRA/train_dagger.py \
        --teacher-checkpoint logs/skrl/dextra_amp_walk/<run>/checkpoints/best_agent.pt \
        --num-envs 2048 --num-iterations 300 --headless

Checkpoints: ``student_latest.pt`` (last iter), ``student_best_eval.pt`` (highest eval
``mean_episode_length`` seen during training). Defaults: ``--lr 1e-4``,
``--beta-decay 0.998``.
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="DAgger distillation: teacher (43D) -> student (24D)")
parser.add_argument("--teacher-checkpoint", type=str, required=True, help="Path to SKRL AMP best_agent.pt")
parser.add_argument("--num-envs", type=int, default=2048, help="Number of parallel environments")
parser.add_argument("--num-iterations", type=int, default=300, help="DAgger outer iterations")
parser.add_argument("--rollout-steps", type=int, default=500, help="Env steps per DAgger iteration")
parser.add_argument("--train-steps", type=int, default=0, help="Gradient updates per iter (0 = auto: 2x new data)")
parser.add_argument("--batch-size", type=int, default=1024, help="Mini-batch size for student training")
parser.add_argument("--lr", type=float, default=1e-4, help="Student learning rate")
parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay")
parser.add_argument("--beta-init", type=float, default=1.0, help="Initial teacher mixing ratio (1=pure teacher)")
parser.add_argument("--beta-decay", type=float, default=0.998, help="Multiplicative beta decay per iteration")
parser.add_argument("--beta-min", type=float, default=0.02, help="Minimum beta value")
parser.add_argument("--buffer-capacity", type=int, default=2_000_000, help="Max transitions in replay buffer")
parser.add_argument("--vel-clip", type=float, default=10.0, help="Clip numerical velocity (rad/s)")
parser.add_argument("--eval-interval", type=int, default=20, help="Evaluate pure-student every N iterations")
parser.add_argument("--eval-steps", type=int, default=300, help="Steps per evaluation rollout")
parser.add_argument("--save-interval", type=int, default=50, help="Save checkpoint every N iterations")
parser.add_argument("--log-dir", type=str, default=None, help="TensorBoard + checkpoint dir (auto if None)")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- heavy imports below (after AppLauncher) ---

import os
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import isaaclab_tasks  # noqa: F401 – triggers gym.register()

from isaaclab_tasks.direct.SOLO_DEXTRA.teacher_policy import TeacherPolicy

TEACHER_OBS_DIM = 43
STUDENT_OBS_DIM = 24
ACTION_DIM = 12
JOINT_DIM = 12
POLICY_DT = (1.0 / 120.0) * 2  # sim_dt * decimation = 1/60 s


# ---------------------------------------------------------------------------
# Running normalizer (no learnable params — stable for DAgger)
# ---------------------------------------------------------------------------

class RunningNormalizer:
    """Welford online normalizer on GPU. No learnable params, no gradient issues."""

    def __init__(self, dim: int, device: str, clip: float = 5.0, eps: float = 1e-6) -> None:
        self.mean = torch.zeros(dim, device=device, dtype=torch.float64)
        self.var = torch.ones(dim, device=device, dtype=torch.float64)
        self.count = torch.tensor(1e-4, device=device, dtype=torch.float64)
        self.clip = clip
        self.eps = eps

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        batch_mean = x.double().mean(dim=0)
        batch_var = x.double().var(dim=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

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

    def state_dict(self) -> dict:
        return {"mean": self.mean.cpu(), "var": self.var.cpu(), "count": self.count.cpu()}

    def load_state_dict(self, d: dict) -> None:
        self.mean = d["mean"].to(self.mean.device)
        self.var = d["var"].to(self.var.device)
        self.count = d["count"].to(self.count.device)


# ---------------------------------------------------------------------------
# Student policy (no BatchNorm — uses external RunningNormalizer)
# ---------------------------------------------------------------------------

class StudentPolicy(nn.Module):
    """Deterministic MLP: 24-D (normalised externally) -> 12-D actions."""

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
# Ring-buffer replay buffer (GPU)
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-capacity ring buffer storing (obs, action) pairs on GPU."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: str) -> None:
        self._capacity = capacity
        self._ptr = 0
        self._size = 0
        self._obs = torch.zeros(capacity, obs_dim, device=device)
        self._act = torch.zeros(capacity, action_dim, device=device)

    @property
    def size(self) -> int:
        return self._size

    def add(self, obs: torch.Tensor, actions: torch.Tensor) -> None:
        n = obs.shape[0]
        if n >= self._capacity:
            obs = obs[-self._capacity:]
            actions = actions[-self._capacity:]
            n = self._capacity
        end = self._ptr + n
        if end <= self._capacity:
            self._obs[self._ptr:end] = obs
            self._act[self._ptr:end] = actions
        else:
            first = self._capacity - self._ptr
            self._obs[self._ptr:] = obs[:first]
            self._act[self._ptr:] = actions[:first]
            rest = n - first
            self._obs[:rest] = obs[first:]
            self._act[:rest] = actions[first:]
        self._ptr = end % self._capacity
        self._size = min(self._size + n, self._capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, self._size, (batch_size,), device=self._obs.device)
        return self._obs[idx], self._act[idx]


# ---------------------------------------------------------------------------
# DAgger trainer
# ---------------------------------------------------------------------------

class DAggerTrainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- log directory ---
        if args.log_dir is None:
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.log_dir = os.path.abspath(os.path.join("logs", "dagger", "dextra", ts))
        else:
            self.log_dir = os.path.abspath(args.log_dir)
        os.makedirs(os.path.join(self.log_dir, "checkpoints"), exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        print(f"[DAgger] Logging to {self.log_dir}")

        # --- seed ---
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        # --- environment ---
        from isaaclab_tasks.direct.SOLO_DEXTRA.dextra_amp_env_cfg import DextraAmpWalkEnvCfg

        env_cfg = DextraAmpWalkEnvCfg()
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.use_fk_observations = False  # 43-D mode
        self.env = gym.make("Isaac-Dextra-Amp-Walk-Direct-v0", cfg=env_cfg)
        print(f"[DAgger] Environment created  num_envs={args.num_envs}  obs=43D")

        # --- teacher ---
        self.teacher = TeacherPolicy(
            checkpoint_path=os.path.abspath(args.teacher_checkpoint),
            obs_dim=TEACHER_OBS_DIM,
            action_dim=ACTION_DIM,
            hidden_dims=(512, 256),
            device=self.device,
        )

        # --- student ---
        self.student = StudentPolicy(
            obs_dim=STUDENT_OBS_DIM,
            action_dim=ACTION_DIM,
            hidden_dims=(256, 256, 128),
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.num_iterations, eta_min=args.lr * 0.1
        )
        total_params = sum(p.numel() for p in self.student.parameters())
        print(f"[DAgger] Student policy: {STUDENT_OBS_DIM}D -> (256,256,128) -> {ACTION_DIM}D  ({total_params:,} params)")

        # --- normalizers ---
        self.obs_normalizer = RunningNormalizer(STUDENT_OBS_DIM, self.device)
        self.action_normalizer = RunningNormalizer(ACTION_DIM, self.device, clip=10.0)

        # --- replay buffer (stores RAW obs and RAW actions — normalised on the fly) ---
        self.buffer = ReplayBuffer(
            capacity=args.buffer_capacity,
            obs_dim=STUDENT_OBS_DIM,
            action_dim=ACTION_DIM,
            device=self.device,
        )

        self.beta = args.beta_init
        self.global_step = 0
        self._prev_joint_pos: torch.Tensor | None = None
        self._best_eval_ep_len = float("-inf")
        self._best_eval_iter = 0

    @staticmethod
    def _unwrap_obs(obs) -> torch.Tensor:
        """Extract the raw tensor from gym obs (may be dict or tensor)."""
        if isinstance(obs, dict):
            return obs["policy"]
        return obs

    def _build_student_obs(
        self,
        obs_43d: torch.Tensor,
        reset_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the 24-D student observation with numerical joint velocity.

        Returns the raw (un-normalised) 24-D observation.  Normalisation
        happens separately via ``obs_normalizer``.
        """
        joint_pos = obs_43d[:, :JOINT_DIM]

        if self._prev_joint_pos is None:
            joint_vel_fd = torch.zeros_like(joint_pos)
        else:
            joint_vel_fd = (joint_pos - self._prev_joint_pos) / POLICY_DT

        if reset_mask is not None and reset_mask.any():
            joint_vel_fd[reset_mask] = 0.0

        joint_vel_fd = joint_vel_fd.clamp(-self.args.vel_clip, self.args.vel_clip)

        self._prev_joint_pos = joint_pos.clone()
        return torch.cat([joint_pos, joint_vel_fd], dim=-1)

    # ---------------------------------------------------------------
    # Core loop
    # ---------------------------------------------------------------

    def run(self) -> None:
        print(f"\n{'='*60}")
        print(f"[DAgger] Starting training: {self.args.num_iterations} iterations")
        print(f"  rollout_steps={self.args.rollout_steps}  train_steps={self.args.train_steps}")
        print(f"  beta={self.beta:.2f}  decay={self.args.beta_decay}  min={self.args.beta_min}")
        print(f"  lr={self.args.lr}  vel_clip={self.args.vel_clip}")
        print(f"  buffer_capacity={self.args.buffer_capacity:,}")
        print(f"{'='*60}\n")

        raw_obs, _ = self.env.reset()
        obs = self._unwrap_obs(raw_obs)
        self._prev_joint_pos = obs[:, :JOINT_DIM].clone()
        t0 = time.time()

        for iteration in range(1, self.args.num_iterations + 1):
            # 1) ROLLOUT with beta-mixed actions
            obs = self._rollout(obs)

            # 2) TRAIN student on accumulated buffer
            train_loss, grad_norm = self._train()

            # 3) DECAY beta + step LR scheduler
            self.beta = max(self.args.beta_min, self.beta * self.args.beta_decay)
            self.scheduler.step()

            # 4) LOG
            self.writer.add_scalar("Loss/mse", train_loss, iteration)
            self.writer.add_scalar("Metric/beta", self.beta, iteration)
            self.writer.add_scalar("Metric/buffer_size", self.buffer.size, iteration)
            self.writer.add_scalar("Metric/grad_norm", grad_norm, iteration)
            self.writer.add_scalar("Metric/lr", self.scheduler.get_last_lr()[0], iteration)

            elapsed = time.time() - t0
            print(
                f"[iter {iteration:4d}/{self.args.num_iterations}]  "
                f"loss={train_loss:.6f}  beta={self.beta:.4f}  "
                f"buf={self.buffer.size:,}  elapsed={elapsed:.0f}s"
            )

            # 5) EVAL (pure student)
            if iteration % self.args.eval_interval == 0:
                eval_mse, eval_len, eval_act_norm = self._evaluate()
                self.writer.add_scalar("Eval/action_mse", eval_mse, iteration)
                self.writer.add_scalar("Eval/mean_episode_length", eval_len, iteration)
                self.writer.add_scalar("Eval/student_action_norm", eval_act_norm, iteration)
                print(
                    f"  [eval] action_mse={eval_mse:.6f}  mean_ep_len={eval_len:.1f}  "
                    f"act_norm={eval_act_norm:.4f}"
                )
                if eval_len > self._best_eval_ep_len:
                    self._best_eval_ep_len = eval_len
                    self._best_eval_iter = iteration
                    self.writer.add_scalar("Eval/best_mean_episode_length", eval_len, iteration)
                    self._save_best_checkpoint(iteration, eval_len, eval_mse)
                self.writer.add_scalar("Eval/best_mean_episode_length_so_far", self._best_eval_ep_len, iteration)

            # 6) SAVE
            if iteration % self.args.save_interval == 0 or iteration == self.args.num_iterations:
                self._save_checkpoint(iteration)

        self.writer.close()
        self.env.close()
        print(f"\n[DAgger] Training complete. Total time: {time.time() - t0:.0f}s")
        print(f"[DAgger] Checkpoints saved to: {os.path.join(self.log_dir, 'checkpoints')}")
        if self._best_eval_ep_len > float("-inf"):
            print(
                f"[DAgger] Best eval mean_ep_len={self._best_eval_ep_len:.1f} at iter {self._best_eval_iter} "
                f"-> checkpoints/student_best_eval.pt"
            )

    # ---------------------------------------------------------------
    # Rollout
    # ---------------------------------------------------------------

    def _rollout(self, obs: torch.Tensor) -> torch.Tensor:
        """Collect data using beta-mixed policy, label with teacher."""
        self.student.eval()
        all_obs_raw: list[torch.Tensor] = []
        all_teacher_act: list[torch.Tensor] = []

        for _ in range(self.args.rollout_steps):
            obs_43d = obs.float()
            obs_24d_raw = self._build_student_obs(obs_43d, reset_mask=None)

            self.obs_normalizer.update(obs_24d_raw)
            obs_24d_norm = self.obs_normalizer.normalize(obs_24d_raw)

            with torch.no_grad():
                teacher_act = self.teacher(obs_43d)
                student_act_norm = self.student(obs_24d_norm)
                student_act = self.action_normalizer.denormalize(student_act_norm)

            self.action_normalizer.update(teacher_act)
            action = self.beta * teacher_act + (1.0 - self.beta) * student_act

            raw_obs, _, terminated, truncated, _ = self.env.step(action)
            obs = self._unwrap_obs(raw_obs)

            reset_mask = terminated | truncated
            if reset_mask.any():
                self._prev_joint_pos[reset_mask] = obs[reset_mask, :JOINT_DIM]

            all_obs_raw.append(obs_24d_raw)
            all_teacher_act.append(teacher_act)
            self.global_step += 1

        self.buffer.add(torch.cat(all_obs_raw, dim=0), torch.cat(all_teacher_act, dim=0))
        return obs

    # ---------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------

    def _train(self) -> tuple[float, float]:
        """Train student on buffered (raw_obs, teacher_action) data.

        Raw observations are normalised on-the-fly using the current
        normaliser statistics so that old and new data are treated
        consistently.

        Returns (avg_loss, avg_grad_norm).
        """
        self.student.train()
        if self.buffer.size < self.args.batch_size:
            return 0.0, 0.0

        new_data_per_iter = self.args.num_envs * self.args.rollout_steps
        if self.args.train_steps > 0:
            n_steps = self.args.train_steps
        else:
            n_steps = max(100, 2 * new_data_per_iter // self.args.batch_size)

        total_loss = 0.0
        total_grad = 0.0
        for _ in range(n_steps):
            obs_raw, act_raw = self.buffer.sample(self.args.batch_size)
            obs_norm = self.obs_normalizer.normalize(obs_raw)
            act_norm = self.action_normalizer.normalize(act_raw)
            pred_norm = self.student(obs_norm)
            loss = nn.functional.mse_loss(pred_norm, act_norm)

            self.optimizer.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_grad += gn.item()

        return total_loss / n_steps, total_grad / n_steps

    # ---------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------

    @torch.no_grad()
    def _evaluate(self) -> tuple[float, float, float]:
        """Roll out pure student and measure action MSE vs teacher & episode length.

        Returns (avg_action_mse, mean_episode_length, mean_student_action_norm).
        """
        self.student.eval()
        raw_obs, _ = self.env.reset()
        obs = self._unwrap_obs(raw_obs)
        self._prev_joint_pos = obs[:, :JOINT_DIM].clone()

        mse_sum = 0.0
        act_norm_sum = 0.0
        count = 0

        for _ in range(self.args.eval_steps):
            obs_43d = obs.float()
            obs_24d_raw = self._build_student_obs(obs_43d, reset_mask=None)
            obs_24d_norm = self.obs_normalizer.normalize(obs_24d_raw)

            student_act_norm = self.student(obs_24d_norm)
            student_act = self.action_normalizer.denormalize(student_act_norm)
            teacher_act = self.teacher(obs_43d)

            mse_sum += nn.functional.mse_loss(student_act, teacher_act).item()
            act_norm_sum += student_act.norm(dim=-1).mean().item()
            count += 1

            raw_obs, _, terminated, truncated, info = self.env.step(student_act)
            obs = self._unwrap_obs(raw_obs)

            reset_mask = terminated | truncated
            if reset_mask.any():
                self._prev_joint_pos[reset_mask] = obs[reset_mask, :JOINT_DIM]

        inner_env = self.env.unwrapped
        ep_len = 0.0
        if hasattr(inner_env, "episode_length_buf"):
            ep_len = float(inner_env.episode_length_buf.float().mean().item())

        avg_mse = mse_sum / max(count, 1)
        avg_act_norm = act_norm_sum / max(count, 1)
        return avg_mse, ep_len, avg_act_norm

    # ---------------------------------------------------------------
    # Checkpointing
    # ---------------------------------------------------------------

    def _checkpoint_payload(self, iteration: int) -> dict:
        return {
            "iteration": iteration,
            "student_state_dict": self.student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "obs_normalizer": self.obs_normalizer.state_dict(),
            "action_normalizer": self.action_normalizer.state_dict(),
            "beta": self.beta,
            "buffer_size": self.buffer.size,
        }

    def _save_checkpoint(self, iteration: int) -> None:
        payload = self._checkpoint_payload(iteration)
        path = os.path.join(self.log_dir, "checkpoints", f"student_iter_{iteration:05d}.pt")
        torch.save(payload, path)
        latest = os.path.join(self.log_dir, "checkpoints", "student_latest.pt")
        torch.save(payload, latest)
        print(f"  [save] {path}")

    def _save_best_checkpoint(self, iteration: int, eval_ep_len: float, eval_action_mse: float) -> None:
        payload = self._checkpoint_payload(iteration)
        payload["best_eval_mean_episode_length"] = eval_ep_len
        payload["best_eval_iteration"] = iteration
        payload["best_eval_action_mse"] = eval_action_mse
        best_path = os.path.join(self.log_dir, "checkpoints", "student_best_eval.pt")
        torch.save(payload, best_path)
        print(
            f"  [best] mean_ep_len={eval_ep_len:.1f} (iter {iteration})  "
            f"action_mse={eval_action_mse:.6f}  -> student_best_eval.pt"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    trainer = DAggerTrainer(args_cli)
    trainer.run()


if __name__ == "__main__":
    main()
    simulation_app.close()
