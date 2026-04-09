# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Dextra AMP environment for learning humanoid locomotion via Adversarial Motion Priors.
"""

import gymnasium as gym

from . import agents
from .dextra_amp_env_cfg import DextraAmpWalkEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Dextra-Amp-Walk-Direct-v0",
    #entry_point="isaaclab_tasks.direct.dextra_amp.dextra_amp_env:DextraAmpEnv",  # ← 우리의 custom env!
    entry_point=f"{__name__}.dextra_amp_env:DextraAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DextraAmpWalkEnvCfg,
        #"skrl_cfg_entry_point": f"{agents.__name__}:skrl_walk_amp_cfg.yaml",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
    },
)

# 기존 env 등록 외에 추가
gym.register(
    id="Isaac-Dextra-AMP-Student-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dextra_amp_env_cfg:DextraAmpWalkEnvCfg",
        "skrl_amp_cfg_entry_point": f"{__name__}.agents:skrl_amp_student_cfg",
    },
    disable_env_checker=True,
)