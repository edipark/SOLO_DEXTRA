# File: isaaclab_tasks/direct/dextra_amp/agents/amp_distillation.py

import torch
import torch.nn.functional as F
from skrl.agents.torch.amp import AMP

class AMP_Distillation(AMP):
    """AMP with teacher distillation"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Distillation hyperparameters
        self.distillation_loss_scale = self.cfg.get("distillation_loss_scale", 2.0)
        self.kl_loss_scale = self.cfg.get("kl_loss_scale", 1.0)
        
        print(f"\n{'='*60}")
        print(f"🎓 Distillation Config")
        print(f"{'='*60}")
        print(f"Distillation loss scale: {self.distillation_loss_scale}")
        print(f"KL loss scale: {self.kl_loss_scale}")
        print(f"Discriminator loss scale: {self.cfg.get('discriminator_loss_scale', 5.0)}")
        print(f"{'='*60}\n")
    
    def _update(self, timestep, timesteps):
        """Override update to add distillation loss"""
        
        # Sample from memory
        sampled_states = self.memory.sample(
            names=self._state_preprocessor_key,
            batch_size=self._rollouts * self._learning_epochs
        )[0]
        
        sampled_actions = self.memory.sample(
            names="actions",
            batch_size=self._rollouts * self._learning_epochs
        )[0]
        
        # ✅ Get teacher data
        teacher_actions = self.memory.sample(
            names="teacher_actions",
            batch_size=self._rollouts * self._learning_epochs
        )[0]
        
        teacher_log_std = self.memory.sample(
            names="teacher_log_std",
            batch_size=self._rollouts * self._learning_epochs
        )[0]
        
        # Preprocess
        sampled_states = self._state_preprocessor(sampled_states)
        
        # Student forward
        _, student_log_prob, student_dist = self.policy.act(
            {"states": sampled_states, "taken_actions": sampled_actions},
            role="policy"
        )
        
        student_mean = student_dist.mean
        student_log_std = student_dist.stddev.log()
        
        # === Loss 1: Action MSE ===
        action_mse_loss = F.mse_loss(student_mean, teacher_actions)
        
        # === Loss 2: KL Divergence ===
        # KL(student || teacher) for Gaussian
        student_var = student_log_std.exp().pow(2)
        teacher_var = teacher_log_std.exp().pow(2)
        
        kl_loss = (
            teacher_log_std - student_log_std +
            (student_var + (student_mean - teacher_actions).pow(2)) / (2 * teacher_var) -
            0.5
        ).sum(dim=-1).mean()
        
        # === Loss 3: Original AMP losses ===
        # (PPO + Discriminator - 기존 코드 사용)
        # 여기서는 간단히 표시만
        amp_losses = self._compute_amp_losses(sampled_states, sampled_actions, student_log_prob)
        
        # === Total Loss ===
        policy_loss = (
            amp_losses["policy_loss"] +
            self.distillation_loss_scale * action_mse_loss +
            self.kl_loss_scale * kl_loss
        )
        
        # Backward
        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        if self._grad_norm_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
        self.optimizer_policy.step()
        
        # Update value & discriminator (기존 방식)
        self._update_value(sampled_states)
        self._update_discriminator()
        
        # Logging
        self.track_data("Loss/distillation_mse", action_mse_loss.item())
        self.track_data("Loss/distillation_kl", kl_loss.item())
        self.track_data("Loss/policy_total", policy_loss.item())
        
        # 기존 AMP logging도 유지
        for key, value in amp_losses.items():
            self.track_data(f"Loss/{key}", value.item())