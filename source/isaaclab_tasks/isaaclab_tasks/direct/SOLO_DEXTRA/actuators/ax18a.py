"""AX-18A Dynamixel servo actuator model for the Dextra robot.

Specifications (12 V supply):
  - Stall torque    : 1.8 N·m  (effort_limit)
  - No-load speed   : 97 RPM = 10.16 rad/s  (velocity_limit)
  - Communication   : Dynamixel TTL, ≤1 ms latency @ 1 Mbps

Modelling choice
----------------
:class:`AX18AActuator` extends :class:`~isaaclab.actuators.ActuatorBase` directly
and implements the Dynamixel compliance torque model:

1. **Compliance torque** – Mimics the AX-18A hardware Compliance Margin /
   Compliance Slope registers.  No torque in the dead-zone; linear ramp up
   to ``effort_limit`` over the slope range.  Replaces the PD stiffness term.

2. **Damping** – Velocity-proportional term for stability.

3. **Coulomb + viscous friction** – Explicit gearbox friction model.

Usage
-----
::

    from .actuators import AX18AActuatorCfg

    robot = DEXTRA_CFG.replace(
        actuators={
            "legs": AX18AActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=0.0,   # unused; required field
                damping=0.45,
            )
        }
    )
"""

from __future__ import annotations

import torch

from isaaclab.actuators import ActuatorBase, ActuatorBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


# ---------------------------------------------------------------------------
# Actuator model (defined first so AX18AActuatorCfg can reference it directly)
# ---------------------------------------------------------------------------

class AX18AActuator(ActuatorBase):
    """Compliance-based actuator model for the Dynamixel AX-18A servo.

    Directly inherits :class:`~isaaclab.actuators.ActuatorBase` and implements
    the AX-18A hardware Compliance Margin / Slope torque profile.

    Torque pipeline (per control step)::

        ratio   = clamp((|e| - margin) / slope, 0, 1)           # 0 in dead-zone, 1 at saturation
        τ_p     = ratio * effort_limit * sign(e)                 # compliance (P) term
        τ_d     = τ_p - kd * q̇                                  # subtract damping
        τ_c     = τ_d  if sign(τ_d)==sign(τ_p) else 0           # prevent direction reversal (outside dead-zone)
              (inside dead-zone: τ_c = τ_d freely for settling)
        τ_max   = effort_limit * max(0, 1 - |q̇|/vel_limit)      # torque-speed saturation
        τ_c     = clamp(τ_c, -τ_max, τ_max)
        τ_fric  = coulomb * tanh(q̇/ε) + viscous * q̇             # smooth friction
        τ_out   = sign(τ_c) * max(0, |τ_c| - |τ_fric|)          # friction reduces magnitude only
    """

    # 1 AX-18A position step = 0.29° in radians
    _AX18A_RAD_PER_STEP: float = 0.29 * (3.14159265358979 / 180.0)

    # Small epsilon for tanh-based smooth sign approximation (rad/s).
    # Avoids gradient discontinuity of torch.sign() near zero velocity.
    _VEL_EPS: float = 0.01

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._coulomb = cfg.coulomb_friction
        self._viscous = cfg.viscous_friction_coeff
        self._friction_torque = torch.zeros_like(self.computed_effort)

        # compliance_margin stored in radians; compliance_slope in AX-18A steps → convert
        self._margin_rad = float(cfg.compliance_margin)
        self._slope_rad = float(cfg.compliance_slope) * self._AX18A_RAD_PER_STEP

        # stall_torque: physical motor limit → used for torque-speed curve (τ_stall/ω_0).
        # effort_limit (= Torque Limit register, Addr 34): software cap on compliance output.
        # The two can differ; e.g. factory Max Torque default = 983/1023 × 1.8 ≈ 1.73 N·m.
        self._stall_torque = float(cfg.stall_torque)

        # Punch scales with effort_limit (Torque Limit), not stall_torque,
        # because punch is a PWM percentage relative to the current Torque Limit.
        self._punch_torque = float(cfg.punch) / 1023.0 * self.effort_limit

    def reset(self, env_ids):
        pass

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        pos_error = control_action.joint_positions - joint_pos
        abs_error = torch.abs(pos_error)

        # ------------------------------------------------------------------
        # 1. Compliance torque — hardware-accurate AX-18A profile.
        #
        #   Dead zone  (|e| < margin) : output = 0
        #   Slope zone (margin ≤ |e| < margin+slope):
        #       output = punch + (effort_limit − punch) × ratio   (linear ramp)
        #       where ratio = (|e| − margin) / slope ∈ [0, 1)
        #   Saturated  (|e| ≥ margin+slope) : output = effort_limit
        #
        #   The Punch register (Addr 48, default 32/1023 ≈ 3.1 % of peak)
        #   models the minimum current the motor applies when error exits the
        #   dead zone, creating a small torque discontinuity at the boundary.
        # ------------------------------------------------------------------
        in_dead_zone = abs_error < self._margin_rad

        if self._slope_rad > 0.0:
            torque_ratio = torch.clamp(
                (abs_error - self._margin_rad) / self._slope_rad, min=0.0, max=1.0
            )
        else:
            torque_ratio = torch.ones_like(abs_error)

        tau_magnitude = torch.where(
            in_dead_zone,
            torch.zeros_like(abs_error),
            self._punch_torque + (self.effort_limit - self._punch_torque) * torque_ratio,
        )
        compliance_tau_p = tau_magnitude * torch.sign(pos_error)

        # ------------------------------------------------------------------
        # 2. Damping — models DC motor back-EMF velocity feedback.
        #
        #   τ = τ_p − kd × v
        #
        #   The combined value is allowed to go negative even when τ_p > 0.
        #   This is physically correct: back-EMF reduces applied current and
        #   can produce a net braking torque during fast motion, which is the
        #   primary mechanism preventing overshoot and oscillation.
        #
        #   Previous implementation clamped to zero when damping > compliance,
        #   which suppressed braking torque during fast motion and caused the
        #   oscillation / overshoot noted in the repo memory notes.
        # ------------------------------------------------------------------
        compliance_tau = compliance_tau_p - self.damping * joint_vel

        # ------------------------------------------------------------------
        # 3. Torque-speed saturation (DC motor 4-quadrant curve).
        #    τ_max(v) = stall_torque × max(0, 1 − |v| / vel_limit)
        #
        #    Uses stall_torque (physical motor limit), NOT effort_limit.
        #    Rationale: the torque-speed curve is a property of the motor
        #    physics (d = τ_stall / ω_0 = 0.177 N·m·s/rad from spec).
        #    effort_limit is a software register cap applied in step 1 already;
        #    the torque-speed curve only further limits at high velocity.
        # ------------------------------------------------------------------
        vel_fraction = torch.clamp(torch.abs(joint_vel) / self.velocity_limit, max=1.0)
        tau_max = self._stall_torque * (1.0 - vel_fraction)
        compliance_tau = torch.clamp(compliance_tau, min=-tau_max, max=tau_max)

        # ------------------------------------------------------------------
        # 4. Coulomb + viscous friction loss.
        #    Use tanh(v/ε) instead of sign(v) for smooth gradients near v≈0.
        #    Friction only *reduces* torque magnitude; never flips direction.
        # ------------------------------------------------------------------
        smooth_sign = torch.tanh(joint_vel / self._VEL_EPS)
        self._friction_torque = self._coulomb * smooth_sign + self._viscous * joint_vel

        # Apply friction without reversing the torque direction:
        #   τ_out = sign(τ_c) × max(0, |τ_c| − |τ_friction|)
        tau_sign = torch.sign(compliance_tau)
        tau_magnitude = torch.clamp(
            torch.abs(compliance_tau) - torch.abs(self._friction_torque), min=0.0
        )

        self.computed_effort = compliance_tau
        self.applied_effort = tau_sign * tau_magnitude

        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@configclass
class AX18AActuatorCfg(ActuatorBaseCfg):
    """Configuration for the AX-18A Dynamixel servo actuator model.

    Default values reflect the AX-18A datasheet at 12 V with hardware-default
    compliance register values (Margin=1, Slope=32).

    All per-joint parameters can be overridden as a ``dict[str, float]``
    to assign different values per joint (e.g. for ankle vs. hip joints).
    """

    class_type: type = AX18AActuator

    # ---- Required by ActuatorBaseCfg (unused in compliance mode) ----
    stiffness: float = 0.0
    """Not used in compliance mode; kept as required field."""

    # ---- AX-18A datasheet values (12 V) ----
    stall_torque: float = 1.8
    """Physical stall torque of the motor [N·m] (spec: 1.8 N·m @ 12 V, 2.2 A).

    Used for the torque-speed saturation curve:
        τ_max(v) = stall_torque × (1 − |v| / velocity_limit)
    This is a fixed motor property. Do NOT lower this to limit output —
    use effort_limit (Torque Limit register) for that."""

    effort_limit: float = 1.8
    """Torque Limit register value [N·m] — software cap on compliance output.

    Corresponds to AX-18A Torque Limit (Addr 34), initialised from Max Torque
    (Addr 14, default 983/1023 × 1.8 ≈ 1.73 N·m).
    Set this ≤ stall_torque to limit joint effort for safety.
    Punch minimum current also scales with this value (not stall_torque)."""

    velocity_limit: float = 10.16
    """No-load speed [rad/s] (97 RPM at 12 V)."""

    damping: float = 0.177
    """Back-EMF velocity damping [N·m·s/rad].

    Derived directly from the AX-18A torque-speed curve:
        d = stall_torque / velocity_limit = 1.8 / 10.16 ≈ 0.177 N·m·s/rad.
    At this value, net compliance torque reaches zero exactly at no-load speed,
    matching the hardware characteristic. viscous_friction_coeff should be 0
    when using this value to avoid double-counting."""

    # ---- Inertia / friction ----
    armature: float = 0.00054
    """Rotor inertia reflected to the joint [kg·m²]."""

    friction: float = 0.05
    """Static friction coefficient applied by PhysX solver."""

    # ---- AX-18A specific ----
    coulomb_friction: float = 0.04
    """Explicit Coulomb friction [N·m] (gearbox static friction)."""

    viscous_friction_coeff: float = 0.0
    """Viscous friction coefficient [N·m·s/rad].

    Set to 0 when damping = stall_torque / velocity_limit (spec-derived),
    as gearbox viscous losses are already captured in that damping value."""

    # ---- Compliance (AX-18A hardware register defaults) ----
    # 1 unit = 0.29° = 5.061e-3 rad
    compliance_margin: float = 0.29 * (3.14159265358979 / 180.0)
    """Dead-zone half-width [rad].  AX-18A default: 1 unit ≈ 5.06e-3 rad."""

    compliance_slope: float = 32.0
    """Torque ramp-up width [AX-18A position steps].

    Converted to radians internally (step × 5.06e-3 rad).
    AX-18A factory default: 32 → slope_rad ≈ 0.162 rad → k ≈ 11.1 N·m/rad.
    Typical values: 1, 2, 4, 8, 16, 32, 64, 128."""

    punch: float = 32.0
    """Punch register value (Addr 48) — minimum motor output current threshold.

    AX-18A range: 32 (0x20) ~ 1023 (0x3FF).  Default factory value: 32.
    Converted internally to torque: (punch / 1023) × effort_limit.
    This creates a small torque step at the compliance margin boundary,
    modelling the minimum drive current needed to overcome internal stiction."""
