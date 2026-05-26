"""Teacher policy wrapper for DAgger distillation.

Loads a pretrained SKRL AMP teacher checkpoint (43D obs -> 12D actions)
including the RunningStandardScaler state_preprocessor, and provides
deterministic action predictions for student training.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TeacherPolicy(nn.Module):
    """Load an SKRL AMP checkpoint and expose deterministic inference.

    The SKRL GaussianMixin policy stores weights under the key ``"policy"``
    with parameter names like ``net_container.{layer_idx}.weight``.  The
    observation preprocessor (``RunningStandardScaler``) is stored under
    ``"state_preprocessor"`` with buffers ``running_mean``, ``running_variance``,
    and ``current_count``.

    Args:
        checkpoint_path: Path to ``best_agent.pt`` or ``agent_<step>.pt``.
        obs_dim: Observation dimensionality expected by the policy network.
        action_dim: Action dimensionality output by the policy network.
        hidden_dims: Hidden layer sizes matching the training YAML.
        device: Torch device string.
    """

    def __init__(
        self,
        checkpoint_path: str,
        obs_dim: int = 43,
        action_dim: int = 12,
        hidden_dims: tuple[int, ...] = (512, 256),
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self._device = device
        self._obs_dim = obs_dim

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        print(f"[TeacherPolicy] Loaded checkpoint: {checkpoint_path}")
        print(f"[TeacherPolicy] Top-level keys: {list(ckpt.keys())}")

        # --- build the same MLP architecture as SKRL's GaussianMixin ---
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.policy_net = nn.Sequential(*layers)

        # --- load policy weights ---
        policy_sd = ckpt["policy"]
        net_sd: dict[str, torch.Tensor] = {}
        for k, v in policy_sd.items():
            if k.startswith("net_container."):
                net_sd[k[len("net_container."):]] = v
        self.policy_net.load_state_dict(net_sd)
        print(f"[TeacherPolicy] Policy MLP loaded ({obs_dim} -> {hidden_dims} -> {action_dim})")

        # --- load state preprocessor (RunningStandardScaler) ---
        if "state_preprocessor" in ckpt:
            sp = ckpt["state_preprocessor"]
            self._running_mean: torch.Tensor = sp["running_mean"].float().to(device)
            self._running_var: torch.Tensor = sp["running_variance"].float().to(device)
            self._has_preprocessor = True
            print(f"[TeacherPolicy] RunningStandardScaler loaded  mean_norm={self._running_mean.norm():.4f}")
        else:
            self._running_mean = torch.zeros(obs_dim, device=device)
            self._running_var = torch.ones(obs_dim, device=device)
            self._has_preprocessor = False
            print("[TeacherPolicy] WARNING: no state_preprocessor in checkpoint")

        self.eval()
        self.to(device)

    def _preprocess(self, obs: torch.Tensor) -> torch.Tensor:
        """Replicate SKRL RunningStandardScaler forward (eval mode)."""
        if not self._has_preprocessor:
            return obs
        eps = 1e-8
        clip = 5.0
        return torch.clamp(
            (obs - self._running_mean) / (torch.sqrt(self._running_var) + eps),
            min=-clip,
            max=clip,
        )

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return deterministic mean actions for the given 43-D observations.

        Args:
            obs: ``[N, 43]`` raw (un-normalised) observations.

        Returns:
            actions: ``[N, 12]`` mean actions.
        """
        return self.policy_net(self._preprocess(obs))

    @torch.no_grad()
    def get_actions(self, obs: torch.Tensor) -> torch.Tensor:
        """Convenience alias for :meth:`forward`."""
        return self.forward(obs)
