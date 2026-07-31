"""Thin AX-18A implicit actuator with a position-error dead zone.

The stiffness, damping, and final effort saturation are handled directly by
the PhysX implicit joint drive. This class only reshapes the commanded
position so the spring part of the drive is inactive inside the configured
dead zone and continuous at its boundary.
"""

from __future__ import annotations

import torch

from isaaclab.actuators import ImplicitActuator, ImplicitActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class AX18AImplicitActuator(ImplicitActuator):
    """PhysX implicit PD drive with an AX-18A-style position dead zone."""

    def __init__(self, cfg: "AX18AImplicitActuatorCfg", *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        if cfg.dead_zone < 0.0:
            raise ValueError(f"dead_zone must be non-negative, got {cfg.dead_zone}")
        self._dead_zone = float(cfg.dead_zone)

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        if control_action.joint_positions is None:
            raise ValueError("AX18AImplicitActuator requires joint position commands")

        position_error = control_action.joint_positions - joint_pos
        active_error = torch.sign(position_error) * torch.clamp(
            torch.abs(position_error) - self._dead_zone,
            min=0.0,
        )
        control_action.joint_positions = joint_pos + active_error

        # PhysX applies stiffness, damping, and effort_limit_sim to the
        # transformed target. The parent also records approximate computed
        # and clipped efforts for rewards and diagnostics.
        return super().compute(control_action, joint_pos, joint_vel)


@configclass
class AX18AImplicitActuatorCfg(ImplicitActuatorCfg):
    """Configuration for :class:`AX18AImplicitActuator`."""

    class_type: type = AX18AImplicitActuator

    # These gains are the actual implicit PD gains, not carrier parameters.
    stiffness: float = 5.4
    damping: float = 0.4

    # PhysX clips the combined spring, damping, and feed-forward effort.
    effort_limit_sim: float = 1.8
    velocity_limit_sim: float = 10.16

    armature: float = 0.00054
    friction: float = 0.0

    # AX-18A position resolution: approximately 0.29 degrees per tick.
    dead_zone: float = 0.29 * (3.14159265358979 / 180.0)
