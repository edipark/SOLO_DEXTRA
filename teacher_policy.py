# File: isaaclab_tasks/direct/dextra_amp/teacher_policy.py

import torch
import torch.nn as nn


class TeacherPolicyWrapper(nn.Module):
    """
    Teacher policy wrapper for distillation
    
    Loads a pretrained teacher policy that was trained with privileged
    information (43D observations) and provides action predictions to
    guide student training.
    
    Architecture: 43D → 512 → 256 → 12D
    """
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        super().__init__()
        self.device = device
        
        # Load checkpoint
        print(f"\n{'='*60}")
        print(f"Loading teacher checkpoint...")
        print(f"Path: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Get policy state dict
        policy_dict = checkpoint["policy"]
        
        print(f"\n🔍 Policy state dict sample:")
        for i, (key, value) in enumerate(list(policy_dict.items())[:5]):
            if isinstance(value, torch.Tensor):
                print(f"  [{i}] {key}: {value.shape}")
        
        # ✅ Extract network weights (remove "net_container." prefix)
        net_dict = {}
        log_std = None
        
        for key, value in policy_dict.items():
            if key.startswith("net_container."):
                # Remove prefix: "net_container.0.weight" → "0.weight"
                new_key = key[14:]
                net_dict[new_key] = value
            elif key == "log_std_parameter":
                log_std = value
        
        print(f"\n📦 Extracted {len(net_dict)} network parameters")
        
        # Build network architecture
        self.policy_net = self._build_network()
        
        # Load weights
        if len(net_dict) == 0:
            print("❌ ERROR: net_dict is empty!")
            print("Attempting to load with strict=False...")
            self.policy_net.load_state_dict(policy_dict, strict=False)
        else:
            self.policy_net.load_state_dict(net_dict)
            print(f"✅ Network weights loaded successfully")
        
        # Load log_std
        if log_std is not None:
            self.log_std = log_std.to(device)
            print(f"✅ Log std loaded: shape {self.log_std.shape}")
        else:
            print("⚠️  No log_std found in checkpoint, using zeros")
            self.log_std = torch.zeros(12, device=device)
        
        # Set to eval mode
        self.policy_net.eval()
        self.policy_net.to(device)
        
        print(f"✅ Teacher policy loaded and ready!")
        print(f"   Architecture: 43 → 512 → 256 → 12")
        print(f"   Device: {device}")
        print(f"{'='*60}\n")
    
    def _build_network(self):
        """
        Rebuild teacher network architecture
        
        Must match the architecture used during teacher training:
        - Input: 43D (privileged observations)
        - Hidden: 512 → 256
        - Output: 12D (actions)
        """
        return nn.Sequential(
            nn.Linear(43, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 12)
        )
    
    @torch.no_grad()
    def get_actions(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get teacher action predictions (no gradient computation)
        
        Args:
            obs: [N, 43] privileged observations from environment
            
        Returns:
            actions: [N, 12] mean action predictions
            log_std: [N, 12] log standard deviations (expanded from parameter)
        """
        mean = self.policy_net(obs)
        log_std = self.log_std.expand(obs.shape[0], -1)
        return mean, log_std