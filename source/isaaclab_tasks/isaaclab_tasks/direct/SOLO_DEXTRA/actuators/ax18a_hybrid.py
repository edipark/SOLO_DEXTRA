"""Hybrid implicit actuator model for the Dynamixel AX-18A.

This model keeps the AX-18A position-error non-linearities (compliance
margin, compliance slope, and punch), but delegates velocity damping and the
final torque limit to the PhysX implicit joint drive.  The non-linear spring
torque is converted into an equivalent position target on every physics step::

    tau_spring = ax18a_compliance(q_command - q)
    q_equivalent = q + (tau_spring - tau_friction) / drive_stiffness

PhysX then solves the following drive implicitly::

    tau = drive_stiffness * (q_equivalent - q)
          + damping * (qdot_command - qdot)

The drive's ``effort_limit_sim`` clips the combined spring and damping torque.
This avoids both the explicit high-frequency derivative feedback and the
effort-limit bypass present in the legacy explicit AX-18A model.
"""

from __future__ import annotations

import torch

from isaaclab.actuators import ImplicitActuator, ImplicitActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class AX18AHybridActuator(ImplicitActuator):
    """AX-18A compliance command shaper backed by a PhysX implicit drive.

    ``stiffness`` is the carrier stiffness of the implicit drive, not the
    effective AX-18A compliance stiffness.  The latter is defined by the
    compliance curve.  At the beginning of each physics step, conversion to an
    equivalent target makes the carrier spring produce exactly the requested
    non-linear compliance torque.

    The class intentionally does not apply another torque-speed/back-EMF
    envelope.  The fitted implicit ``damping`` represents the effective
    closed-loop damping, while ``effort_limit_sim`` is the final motor-output
    limit applied by PhysX.
    """

    _AX18A_RAD_PER_STEP: float = 0.29 * (torch.pi / 180.0)
    _VEL_EPS: float = 0.01
    _MIN_STIFFNESS: float = 1.0e-6

    def __init__(self, cfg: "AX18AHybridActuatorCfg", *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        if torch.any(self.stiffness <= self._MIN_STIFFNESS):
            raise ValueError("AX18AHybridActuator stiffness must be positive")
        if torch.any(self.effort_limit > float(cfg.stall_torque) + 1.0e-6):
            raise ValueError("effort_limit_sim must not exceed AX-18A stall_torque")
        if cfg.compliance_margin < 0.0:
            raise ValueError("compliance_margin must be non-negative")
        if cfg.compliance_slope < 0.0:
            raise ValueError("compliance_slope must be non-negative")
        if not 0.0 <= cfg.punch <= 1023.0:
            raise ValueError("punch must be in [0, 1023]")

        self._stall_torque = float(cfg.stall_torque)
        self._margin_rad = float(cfg.compliance_margin)
        self._slope_rad = float(cfg.compliance_slope) * float(self._AX18A_RAD_PER_STEP)
        self._punch_fraction = float(cfg.punch) / 1023.0
        self._punch_torque = self._punch_fraction * self.effort_limit
        self._coulomb = float(cfg.coulomb_friction)
        self._viscous = float(cfg.viscous_friction_coeff)
        self._friction_torque = torch.zeros_like(self.computed_effort)
        self._compliance_effort = torch.zeros_like(self.computed_effort)
        self._zero_command = torch.zeros_like(self.computed_effort)

    def reset(self, env_ids):
        # The model has no temporal state.  All output is recomputed from the
        # current command and measured joint state on every physics step.
        pass

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        if control_action.joint_positions is None:
            raise ValueError("AX18AHybridActuator requires joint position commands")

        pos_error = control_action.joint_positions - joint_pos
        abs_error = torch.abs(pos_error)
        in_dead_zone = abs_error < self._margin_rad

        if self._slope_rad > 0.0:
            torque_ratio = torch.clamp(
                (abs_error - self._margin_rad) / self._slope_rad,
                min=0.0,
                max=1.0,
            )
        else:
            torque_ratio = torch.ones_like(abs_error)

        # Punch and the compliance plateau scale with the current Torque Limit.
        # The buffer is updated by effort-limit DR as well, but computing from
        # the live limit here also keeps the relation correct for other callers.
        self._punch_torque.copy_(self._punch_fraction * self.effort_limit)
        torque_magnitude = torch.where(
            in_dead_zone,
            torch.zeros_like(abs_error),
            self._punch_torque + (self.effort_limit - self._punch_torque) * torque_ratio,
        )
        self._compliance_effort.copy_(torque_magnitude * torch.sign(pos_error))

        # Passive gearbox friction is represented as a spring-target offset so
        # it is included in the same final PhysX drive-force saturation.
        smooth_sign = torch.tanh(joint_vel / self._VEL_EPS)
        self._friction_torque.copy_(self._coulomb * smooth_sign + self._viscous * joint_vel)

        drive_feedforward = self._compliance_effort - self._friction_torque
        if control_action.joint_efforts is not None:
            drive_feedforward = drive_feedforward + control_action.joint_efforts

        # Convert the desired non-linear spring/feed-forward torque to a target
        # for the implicit linear drive.  Damping remains inside PhysX.
        control_action.joint_positions = joint_pos + drive_feedforward / self.stiffness
        # Explicit effort must be cleared every step; returning None would
        # leave a previously written simulation effort buffer unchanged.
        control_action.joint_efforts = self._zero_command

        if control_action.joint_velocities is None:
            control_action.joint_velocities = self._zero_command
        velocity_error = control_action.joint_velocities - joint_vel

        # These values are diagnostic approximations.  PhysX performs the
        # actual implicit solve and final maximum-force clipping.
        self.computed_effort = drive_feedforward + self.damping * velocity_error
        self.applied_effort = self._clip_effort(self.computed_effort)
        return control_action


@configclass
class AX18AHybridActuatorCfg(ImplicitActuatorCfg):
    """Configuration for :class:`AX18AHybridActuator`."""

    class_type: type = AX18AHybridActuator

    # PhysX implicit carrier drive.  Stiffness only converts the requested
    # compliance torque to an equivalent target; damping is the effective
    # closed-loop damping to be identified from the hardware response.
    stiffness: float = 11.1
    damping: float = 0.177

    # For an implicit actuator these are the actual PhysX drive limits.
    effort_limit_sim: float = 1.8
    velocity_limit_sim: float = 10.16

    armature: float = 0.00054
    friction: float = 0.0

    # AX-18A parameters retained from the legacy actuator configuration.
    stall_torque: float = 1.8
    coulomb_friction: float = 0.0
    viscous_friction_coeff: float = 0.0
    compliance_margin: float = 0.29 * (3.14159265358979 / 180.0)
    compliance_slope: float = 64.0
    punch: float = 32.0
