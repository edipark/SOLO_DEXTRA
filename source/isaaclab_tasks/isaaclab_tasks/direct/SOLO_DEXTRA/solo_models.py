"""
SOLO: Shared Models and Utilities
=================================
State estimator 모델 (LSTM, TCN, MLP), Teacher Policy, 데이터 수집기,
평가기 등 공통 컴포넌트를 정의한다.
"""

import torch
import torch.nn as nn
import numpy as np
import pickle

# ============================================================================
# CONSTANTS
# ============================================================================

ENCODER_DIM = 24   # 관절 각도 (12 pos + 12 vel)
PRIV_DIM = 19      # Teacher가 사용하는 privileged state
OBS_DIM = 43       # ENCODER_DIM + PRIV_DIM
ACTION_DIM = 12    # 관절 토크 명령

PRIV_NAMES = [
    "base_height", "tangent_x", "tangent_y", "tangent_z",
    "normal_x", "normal_y", "normal_z",
    "lin_vel_x", "lin_vel_y", "lin_vel_z",
    "ang_vel_x", "ang_vel_y", "ang_vel_z",
    "L_foot_x", "L_foot_y", "L_foot_z",
    "R_foot_x", "R_foot_y", "R_foot_z",
]


# ============================================================================
# HELPER
# ============================================================================

def force_skrl_isaaclab_reset(env) -> None:
    """SKRL IsaacLabWrapper는 reset을 한 번만 호출하므로, 매 eval/collect 전에 재활성화."""
    e = env
    for _ in range(32):
        if hasattr(e, "_reset_once"):
            e._reset_once = True
        nxt = getattr(e, "_env", None)
        if nxt is None or nxt is e:
            break
        e = nxt


# ============================================================================
# STATE ESTIMATOR MODELS
# ============================================================================

class LSTMStateEstimator(nn.Module):
    """LSTM 기반 상태 추정기: (B, T, encoder_dim) → (B, priv_dim)"""

    def __init__(self, encoder_dim=ENCODER_DIM, priv_dim=PRIV_DIM,
                 hidden_size=256, num_layers=2, dropout=0.0):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.priv_dim = priv_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=encoder_dim, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, priv_dim),
        )

        self.register_buffer("input_mean", torch.zeros(encoder_dim))
        self.register_buffer("input_std", torch.ones(encoder_dim))
        self.register_buffer("output_mean", torch.zeros(priv_dim))
        self.register_buffer("output_std", torch.ones(priv_dim))

    def forward(self, x):
        x_norm = (x - self.input_mean) / (self.input_std + 1e-8)
        _, (h_n, _) = self.lstm(x_norm)
        return self.fc(h_n[-1])

    def predict_denormalized(self, x):
        out = self.forward(x)
        return out * self.output_std + self.output_mean

    def set_input_normalization(self, mean, std):
        self.input_mean.data = mean.to(self.input_mean.device)
        self.input_std.data = std.to(self.input_std.device)

    def set_output_normalization(self, mean, std):
        self.output_mean.data = mean.to(self.output_mean.device)
        self.output_std.data = std.to(self.output_std.device)

    def get_config(self):
        return {
            "type": "LSTM",
            "encoder_dim": self.encoder_dim,
            "priv_dim": self.priv_dim,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
        }


class TCNBlock(nn.Module):
    """Causal dilated convolution block with residual connection."""

    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.trim1 = padding
        self.trim2 = padding
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        # x: (B, C, T)
        out = self.conv1(x)
        if self.trim1 > 0:
            out = out[:, :, :-self.trim1]
        out = self.relu(self.dropout(out))

        out = self.conv2(out)
        if self.trim2 > 0:
            out = out[:, :, :-self.trim2]
        out = self.relu(self.dropout(out))

        return self.relu(out + self.skip(x))


class TCNStateEstimator(nn.Module):
    """TCN 기반 상태 추정기: (B, T, encoder_dim) → (B, priv_dim)

    Temporal Convolutional Network으로 시계열 관절 각도 정보에서
    privileged state를 추정한다.
    """

    def __init__(self, encoder_dim=ENCODER_DIM, priv_dim=PRIV_DIM,
                 num_channels=(64, 128, 128), kernel_size=3, dropout=0.1):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.priv_dim = priv_dim
        self.num_channels = list(num_channels)
        self.kernel_size = kernel_size

        layers = []
        in_ch = encoder_dim
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)

        self.fc = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1] // 2),
            nn.ReLU(),
            nn.Linear(num_channels[-1] // 2, priv_dim),
        )

        self.register_buffer("input_mean", torch.zeros(encoder_dim))
        self.register_buffer("input_std", torch.ones(encoder_dim))
        self.register_buffer("output_mean", torch.zeros(priv_dim))
        self.register_buffer("output_std", torch.ones(priv_dim))

    def forward(self, x):
        # x: (B, T, encoder_dim)
        x_norm = (x - self.input_mean) / (self.input_std + 1e-8)
        # Conv expects (B, C, T)
        out = self.tcn(x_norm.transpose(1, 2))
        # 마지막 time step 사용
        return self.fc(out[:, :, -1])

    def predict_denormalized(self, x):
        out = self.forward(x)
        return out * self.output_std + self.output_mean

    def set_input_normalization(self, mean, std):
        self.input_mean.data = mean.to(self.input_mean.device)
        self.input_std.data = std.to(self.input_std.device)

    def set_output_normalization(self, mean, std):
        self.output_mean.data = mean.to(self.output_mean.device)
        self.output_std.data = std.to(self.output_std.device)

    def get_config(self):
        return {
            "type": "TCN",
            "encoder_dim": self.encoder_dim,
            "priv_dim": self.priv_dim,
            "num_channels": self.num_channels,
            "kernel_size": self.kernel_size,
        }


class MLPStateEstimator(nn.Module):
    """MLP 상태 추정기: 히스토리 없이 single frame (B, encoder_dim) → (B, priv_dim)"""

    def __init__(self, encoder_dim=ENCODER_DIM, priv_dim=PRIV_DIM, hidden_size=256):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.priv_dim = priv_dim
        self.hidden_size = hidden_size

        self.net = nn.Sequential(
            nn.Linear(encoder_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, priv_dim),
        )

        self.register_buffer("input_mean", torch.zeros(encoder_dim))
        self.register_buffer("input_std", torch.ones(encoder_dim))
        self.register_buffer("output_mean", torch.zeros(priv_dim))
        self.register_buffer("output_std", torch.ones(priv_dim))

    def forward(self, x):
        if x.dim() == 3:
            x = x[:, -1, :]
        x_norm = (x - self.input_mean) / (self.input_std + 1e-8)
        return self.net(x_norm)

    def predict_denormalized(self, x):
        out = self.forward(x)
        return out * self.output_std + self.output_mean

    def set_input_normalization(self, mean, std):
        self.input_mean.data = mean.to(self.input_mean.device)
        self.input_std.data = std.to(self.input_std.device)

    def set_output_normalization(self, mean, std):
        self.output_mean.data = mean.to(self.output_mean.device)
        self.output_std.data = std.to(self.output_std.device)

    def get_config(self):
        return {
            "type": "MLP",
            "encoder_dim": self.encoder_dim,
            "priv_dim": self.priv_dim,
            "hidden_size": self.hidden_size,
        }


# ============================================================================
# TEACHER POLICY
# ============================================================================

class TeacherPolicy(nn.Module):
    """SKRL AMP 체크포인트에서 로드한 Teacher Policy (43D obs → 12D action)"""

    def __init__(self, obs_dim=OBS_DIM, action_dim=ACTION_DIM, device="cuda"):
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )
        self.register_buffer("running_mean", torch.zeros(obs_dim))
        self.register_buffer("running_var", torch.ones(obs_dim))

    def forward(self, obs):
        if obs.device != self.running_mean.device:
            obs = obs.to(self.running_mean.device)
        obs_norm = (obs - self.running_mean) / (torch.sqrt(self.running_var) + 1e-8)
        obs_norm = torch.clamp(obs_norm, -5.0, 5.0)
        return self.net(obs_norm)

    def load_from_checkpoint(self, ckpt_path, device="cuda"):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        policy = ckpt["policy"]
        mapping = {"net_container.0": 0, "net_container.2": 2, "net_container.4": 4}
        for skrl_key, idx in mapping.items():
            self.net[idx].weight.data = policy[f"{skrl_key}.weight"].float().to(device)
            self.net[idx].bias.data = policy[f"{skrl_key}.bias"].float().to(device)
        self.running_mean.data = ckpt["state_preprocessor"]["running_mean"].float().to(device)
        self.running_var.data = ckpt["state_preprocessor"]["running_variance"].float().to(device)
        self.to(device)
        self.device = device
        return self


class SkrlAgentWrapper:
    """skrl Agent를 wrapping하여 deterministic action을 얻는다."""

    def __init__(self, agent):
        self.agent = agent
        self.agent.set_running_mode("eval")

    def get_action(self, obs):
        with torch.no_grad():
            out = self.agent.act(obs, 0, 0)
            return out[-1].get("mean_actions", out[0])


# ============================================================================
# DATA COLLECTION
# ============================================================================

class DataCollector:
    """Teacher rollout에서 (history, privileged, single_frame) 데이터를 수집."""

    def __init__(self, window, encoder_dim=ENCODER_DIM, priv_dim=PRIV_DIM, device="cuda"):
        self.window = window
        self.encoder_dim = encoder_dim
        self.priv_dim = priv_dim
        self.device = device

    def collect_with_teacher_gt(self, env, teacher, num_steps, noise=0.0):
        """Teacher + GT privileged info로 rollout하며 데이터 수집."""
        histories, privileged, single_frames = [], [], []
        force_skrl_isaaclab_reset(env)
        obs, _ = env.reset()
        num_envs = obs.shape[0]

        buf = torch.zeros((num_envs, self.window, self.encoder_dim), device=self.device)
        valid = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        stats = {"samples": 0, "ep_lengths": [], "deaths": 0,
                 "ep_len": torch.zeros(num_envs, device=self.device)}

        with torch.no_grad():
            for _ in range(num_steps):
                enc = obs[:, :self.encoder_dim]
                priv = obs[:, self.encoder_dim:self.encoder_dim + self.priv_dim]

                buf = torch.roll(buf, -1, dims=1)
                buf[:, -1] = enc
                valid += 1

                action = teacher.get_action(obs)

                # 모든 스텝 수집 (zero-padded warm-up 포함)
                histories.append(buf.cpu())
                privileged.append(priv.cpu())
                single_frames.append(enc.cpu())
                stats["samples"] += num_envs

                if noise > 0:
                    action = action + torch.randn_like(action) * noise

                obs, _, term, trunc, _ = env.step(action)
                stats["ep_len"] += 1

                done = (term | trunc).squeeze()
                if done.any():
                    stats["deaths"] += term.squeeze().sum().item()
                    stats["ep_lengths"].extend(stats["ep_len"][done].cpu().tolist())
                    stats["ep_len"][done] = 0
                    buf[done] = 0
                    valid[done] = 0

        if not histories:
            return None, None, None, {}

        ep_lens = stats["ep_lengths"]
        return (
            torch.cat(histories), torch.cat(privileged), torch.cat(single_frames),
            {
                "total_samples": stats["samples"],
                "avg_episode": np.mean(ep_lens) if ep_lens else num_steps,
                "std_episode": np.std(ep_lens) if ep_lens else 0.0,
                "death_rate": stats["deaths"] / len(ep_lens) * 100 if ep_lens else 0.0,
            },
        )

    def collect_with_estimator(self, env, teacher, estimator, num_steps,
                               est_ratio=0.8, noise=0.0, use_mlp=False):
        """Estimator로 privileged를 추정하며 rollout + 데이터 수집 (DAgger)."""
        histories, privileged, single_frames = [], [], []
        force_skrl_isaaclab_reset(env)
        obs, _ = env.reset()
        num_envs = obs.shape[0]

        buf = torch.zeros((num_envs, self.window, self.encoder_dim), device=self.device)
        valid = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        stats = {
            "samples": 0, "est_used": 0, "gt_used": 0,
            "ep_lengths": [], "deaths": 0,
            "ep_len": torch.zeros(num_envs, device=self.device),
        }

        teacher.eval()
        estimator.eval()

        with torch.no_grad():
            for _ in range(num_steps):
                enc = obs[:, :self.encoder_dim]
                priv_gt = obs[:, self.encoder_dim:self.encoder_dim + self.priv_dim]

                buf = torch.roll(buf, -1, dims=1)
                buf[:, -1] = enc
                valid += 1

                # warm-up 포함: 항상 estimator 사용 가능 (zero-padded history)
                use_est = torch.rand(num_envs, device=self.device) < est_ratio

                if use_mlp:
                    priv_est = estimator.predict_denormalized(enc)
                else:
                    priv_est = estimator.predict_denormalized(buf)

                priv_for_action = torch.where(use_est.unsqueeze(-1), priv_est, priv_gt)
                obs_combined = torch.cat([enc, priv_for_action], dim=-1)
                action = teacher(obs_combined)

                # 모든 스텝 수집 (zero-padded warm-up 포함)
                histories.append(buf.cpu())
                privileged.append(priv_gt.cpu())
                single_frames.append(enc.cpu())
                stats["samples"] += num_envs
                stats["est_used"] += use_est.sum().item()
                stats["gt_used"] += (~use_est).sum().item()

                if noise > 0:
                    action = action + torch.randn_like(action) * noise

                obs, _, term, trunc, _ = env.step(action)
                stats["ep_len"] += 1

                done = (term | trunc).squeeze()
                if done.any():
                    stats["deaths"] += term.squeeze().sum().item()
                    stats["ep_lengths"].extend(stats["ep_len"][done].cpu().tolist())
                    stats["ep_len"][done] = 0
                    buf[done] = 0
                    valid[done] = 0

        if not histories:
            return None, None, None, {}

        total = stats["est_used"] + stats["gt_used"]
        return (
            torch.cat(histories), torch.cat(privileged), torch.cat(single_frames),
            {
                "total_samples": stats["samples"],
                "est_usage": stats["est_used"] / total if total > 0 else 0,
                "avg_episode": np.mean(stats["ep_lengths"]) if stats["ep_lengths"] else num_steps,
                "death_rate": stats["deaths"] / len(stats["ep_lengths"]) if stats["ep_lengths"] else 0,
            },
        )


# ============================================================================
# EVALUATION (EPISODE-BASED)
# ============================================================================

class Evaluator:
    """Episode 단위 평가기: death/timeout 구분, 통계 산출."""

    def __init__(self, window, encoder_dim=ENCODER_DIM, priv_dim=PRIV_DIM,
                 max_episode_steps=1000, device="cuda"):
        self.window = window
        self.encoder_dim = encoder_dim
        self.priv_dim = priv_dim
        self.max_episode_steps = max_episode_steps
        self.device = device

    def evaluate_teacher_gt(self, env, teacher, num_episodes, seed=None):
        """Teacher + GT privileged info 기준 baseline 평가."""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        force_skrl_isaaclab_reset(env)
        obs, _ = env.reset()
        num_envs = obs.shape[0]

        completed_episodes = []
        cur_len = torch.zeros(num_envs, device=self.device)
        deaths, timeouts = 0, 0

        teacher.eval()
        with torch.no_grad():
            while len(completed_episodes) < num_episodes:
                action = teacher(obs)
                obs, _, term, trunc, _ = env.step(action)
                cur_len += 1

                done = (term | trunc).squeeze()
                timeout = (cur_len >= self.max_episode_steps) & (~done)
                any_done = done | timeout

                if any_done.any():
                    for idx in any_done.nonzero(as_tuple=True)[0]:
                        if len(completed_episodes) >= num_episodes:
                            break
                        episode_data = {
                            "length": int(cur_len[idx].item()),
                            "death": bool(term[idx].item()),
                            "timeout": bool(timeout[idx].item()),
                        }
                        completed_episodes.append(episode_data)
                        if episode_data["death"]:
                            deaths += 1
                        if episode_data["timeout"]:
                            timeouts += 1

                    cur_len[any_done] = 0

        ep_lengths = [e["length"] for e in completed_episodes]
        return {
            "avg_episode": np.mean(ep_lengths),
            "std_episode": np.std(ep_lengths),
            "death_rate": deaths / num_episodes * 100,
            "timeout_rate": timeouts / num_episodes * 100,
            "num_episodes": len(completed_episodes),
        }

    def evaluate_with_estimator(self, env, teacher, estimator, num_episodes,
                                seed=None, use_mlp=False):
        """State estimator로 privileged info를 추정하며 평가."""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        force_skrl_isaaclab_reset(env)
        obs, _ = env.reset()
        num_envs = obs.shape[0]

        buf = torch.zeros((num_envs, self.window, self.encoder_dim), device=self.device)
        valid = torch.zeros(num_envs, device=self.device, dtype=torch.long)

        completed_episodes = []
        cur_len = torch.zeros(num_envs, device=self.device)
        deaths, timeouts = 0, 0

        teacher.eval()
        estimator.eval()

        with torch.no_grad():
            while len(completed_episodes) < num_episodes:
                enc = obs[:, :self.encoder_dim]

                buf = torch.roll(buf, -1, dims=1)
                buf[:, -1] = enc
                valid += 1

                if use_mlp:
                    priv_est = estimator.predict_denormalized(enc)
                else:
                    priv_est = estimator.predict_denormalized(buf)

                # 항상 estimator 사용 (warm-up 포함, GT fallback 없음)
                obs_est = torch.cat([enc, priv_est], dim=-1)
                action = teacher(obs_est)

                obs, _, term, trunc, _ = env.step(action)
                cur_len += 1

                done = (term | trunc).squeeze()
                timeout = (cur_len >= self.max_episode_steps) & (~done)
                any_done = done | timeout

                if any_done.any():
                    for idx in any_done.nonzero(as_tuple=True)[0]:
                        if len(completed_episodes) >= num_episodes:
                            break
                        episode_data = {
                            "length": int(cur_len[idx].item()),
                            "death": bool(term[idx].item()),
                            "timeout": bool(timeout[idx].item()),
                        }
                        completed_episodes.append(episode_data)
                        if episode_data["death"]:
                            deaths += 1
                        if episode_data["timeout"]:
                            timeouts += 1

                    cur_len[any_done] = 0
                    buf[any_done] = 0
                    valid[any_done] = 0

        ep_lengths = [e["length"] for e in completed_episodes]
        return {
            "avg_episode": np.mean(ep_lengths),
            "std_episode": np.std(ep_lengths),
            "death_rate": deaths / num_episodes * 100,
            "timeout_rate": timeouts / num_episodes * 100,
            "num_episodes": len(completed_episodes),
        }


# ============================================================================
# ESTIMATOR TRAINER
# ============================================================================

class EstimatorTrainer:
    """State estimator 학습 로직 (MSE, early-stopping, normalization)."""

    def __init__(self, estimator, device="cuda", use_mlp=False):
        self.estimator = estimator.to(device)
        self.device = device
        self.use_mlp = use_mlp

    def train(self, histories, targets, single_frames=None,
              epochs=50, lr=1e-3, batch_size=1024, verbose=True):
        # Input normalization
        if self.use_mlp:
            input_data = single_frames
            input_mean = input_data.mean(dim=0)
            input_std = input_data.std(dim=0) + 1e-8
        else:
            input_mean = histories.mean(dim=(0, 1))
            input_std = histories.std(dim=(0, 1)) + 1e-8

        self.estimator.set_input_normalization(input_mean.to(self.device), input_std.to(self.device))

        # Output normalization
        output_mean = targets.mean(dim=0)
        output_std = targets.std(dim=0) + 1e-8
        targets_norm = (targets - output_mean) / output_std
        self.estimator.set_output_normalization(output_mean.to(self.device), output_std.to(self.device))

        # Train / Val split
        n = len(targets)
        idx = torch.randperm(n)
        val_n = min(int(n * 0.1), 10000)

        if self.use_mlp:
            train_x, val_x = single_frames[idx[val_n:]], single_frames[idx[:val_n]]
        else:
            train_x, val_x = histories[idx[val_n:]], histories[idx[:val_n]]
        train_t, val_t = targets_norm[idx[val_n:]], targets_norm[idx[:val_n]]

        if verbose:
            print(f"      Train: {len(train_x):,}, Val: {len(val_x):,}")

        opt = torch.optim.AdamW(self.estimator.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

        best_loss, best_state = float("inf"), None
        patience_counter, max_patience = 0, max(15, epochs // 4)

        for epoch in range(epochs):
            self.estimator.train()
            losses = []
            perm = torch.randperm(len(train_x))

            for i in range(0, len(train_x), batch_size):
                x = train_x[perm[i:i + batch_size]].to(self.device)
                t = train_t[perm[i:i + batch_size]].to(self.device)
                pred = self.estimator(x)
                loss = nn.functional.mse_loss(pred, t)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.estimator.parameters(), 1.0)
                opt.step()
                losses.append(loss.item())

            self.estimator.eval()
            val_losses = []
            with torch.no_grad():
                for i in range(0, len(val_x), batch_size):
                    x = val_x[i:i + batch_size].to(self.device)
                    t = val_t[i:i + batch_size].to(self.device)
                    val_losses.append(nn.functional.mse_loss(self.estimator(x), t).item())

            train_loss, val_loss = np.mean(losses), np.mean(val_losses)
            sched.step(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.estimator.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 10 == 0:
                print(f"        Epoch {epoch + 1:3d}: train={train_loss:.6f}, val={val_loss:.6f}")

            if patience_counter >= max_patience:
                if verbose:
                    print(f"        Early stop at epoch {epoch + 1}")
                break

        self.estimator.load_state_dict(best_state)
        self.estimator.to(self.device)

        return output_mean, output_std, {"best_val_loss": best_loss}


# ============================================================================
# MODEL SAVE / LOAD
# ============================================================================

class ModelSaver:
    """Estimator 모델을 seed / experiment / round 별로 저장."""

    def __init__(self, output_dir, teacher_checkpoint):
        import os
        self.output_dir = output_dir
        self.teacher_checkpoint = teacher_checkpoint
        self.models_dir = os.path.join(output_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)

    def save_estimator(self, estimator, exp_name, seed, round_num, window, extra_info=None):
        import os
        filename = f"{exp_name}_seed{seed}_round{round_num:02d}.pt"
        filepath = os.path.join(self.models_dir, filename)

        save_dict = {
            "estimator_state_dict": estimator.state_dict(),
            "estimator_config": estimator.get_config(),
            "exp_name": exp_name,
            "seed": seed,
            "round_num": round_num,
            "window": window,
            "teacher_checkpoint": self.teacher_checkpoint,
        }
        if extra_info:
            save_dict.update(extra_info)

        torch.save(save_dict, filepath)
        return filepath


def build_estimator(est_type, encoder_dim=ENCODER_DIM, priv_dim=PRIV_DIM,
                    hidden_size=256, num_layers=2, num_channels=(64, 128, 128),
                    kernel_size=3, dropout=0.1, device="cuda"):
    """문자열 타입에 따라 적절한 estimator 인스턴스를 생성한다."""
    est_type = est_type.upper()
    if est_type == "LSTM":
        return LSTMStateEstimator(
            encoder_dim=encoder_dim, priv_dim=priv_dim,
            hidden_size=hidden_size, num_layers=num_layers,
        ).to(device)
    elif est_type == "TCN":
        return TCNStateEstimator(
            encoder_dim=encoder_dim, priv_dim=priv_dim,
            num_channels=num_channels, kernel_size=kernel_size, dropout=dropout,
        ).to(device)
    elif est_type == "MLP":
        return MLPStateEstimator(
            encoder_dim=encoder_dim, priv_dim=priv_dim,
            hidden_size=hidden_size,
        ).to(device)
    else:
        raise ValueError(f"Unknown estimator type: {est_type}. Choose from LSTM, TCN, MLP.")


def load_estimator(checkpoint_path, device="cuda"):
    """저장된 체크포인트에서 estimator를 복원한다."""
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except pickle.UnpicklingError:
        # PyTorch 2.6+에서 일부 기존 체크포인트는 weights_only 로딩이 실패할 수 있다.
        # 로컬/신뢰 가능한 체크포인트라는 전제에서만 fallback을 사용한다.
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["estimator_config"]
    estimator = build_estimator(cfg["type"], device=device, **{
        k: v for k, v in cfg.items() if k != "type"
    })
    estimator.load_state_dict(ckpt["estimator_state_dict"])
    estimator.to(device)
    estimator.eval()
    return estimator, ckpt
