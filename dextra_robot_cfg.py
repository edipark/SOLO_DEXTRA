"""Dextra lower-body robot configuration for Isaac Lab."""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

DEXTRA_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=os.path.join(ASSETS_DIR, "Dextra_lowerbody.urdf"),
        fix_base=False,
        merge_fixed_joints=False,
        self_collision=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=6.0,
                damping=0.6,
            ),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.2865),
    ),
)
