#!/usr/bin/env python3
"""Sweep AX-18A damping values in a fixed-base thigh step-response test.

The script creates one complete Dextra robot per damping value. Every robot
is rigidly fixed at the base and suspended above the ground. All joints hold
their baseline target while one selected joint receives the same positive and
negative position steps. No policy, estimator, termination, or domain
randomization is involved.

Outputs are written to a new timestamped directory for every run:

* ``responses.csv``: command, position, velocity, and torque at physics rate.
* ``step_response.png``: command/response comparison for all damping values.
* ``summary.json``: response metrics for each damping value.
* ``config.json``: exact experiment and actuator settings.

Example:

    ./isaaclab.sh -p \
      source/isaaclab_tasks/isaaclab_tasks/direct/SOLO_DEXTRA/sysid/run_sim_step_response.py \
      --headless --joint-name L_Thigh_Joint --step-deg 5 \
      --damping-values 0.03 0.05 0.08 0.12 0.177
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from datetime import datetime

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Fixed-base AX-18A damping sweep")
parser.add_argument("--joint-name", type=str, default="L_Thigh_Joint",
                    help="Exact robot joint name to excite")
parser.add_argument("--damping-values", type=float, nargs="+",
                    default=[0.03, 0.05, 0.08, 0.12, 0.177],
                    help="Actuator damping values [N m s/rad]")
parser.add_argument("--step-deg", type=float, default=5.0,
                    help="Positive and negative target step magnitude [deg]")
parser.add_argument("--baseline-deg", type=float, default=0.0,
                    help="Selected-joint baseline target [deg]")
parser.add_argument("--settle-s", type=float, default=0.75,
                    help="Initial baseline settling duration [s]")
parser.add_argument("--step-hold-s", type=float, default=1.5,
                    help="Duration of each positive/negative step [s]")
parser.add_argument("--center-hold-s", type=float, default=0.8,
                    help="Baseline hold between the two steps [s]")
parser.add_argument("--final-hold-s", type=float, default=0.8,
                    help="Final baseline hold after the negative step [s]")
parser.add_argument("--physics-hz", type=float, default=120.0,
                    help="Simulation and logging rate [Hz]")
parser.add_argument("--torque-limit-ratio", type=float, default=0.18,
                    help="AX-18A Torque Limit register ratio")
parser.add_argument("--stall-torque", type=float, default=1.8,
                    help="AX-18A output stall torque [N m]")
parser.add_argument("--velocity-limit", type=float, default=10.16,
                    help="AX-18A no-load velocity [rad/s]")
parser.add_argument("--compliance-margin", type=int, default=1,
                    help="AX-18A Compliance Margin register value")
parser.add_argument("--compliance-slope", type=float, default=64.0,
                    help="AX-18A Compliance Slope register value")
parser.add_argument("--punch", type=float, default=32.0,
                    help="AX-18A Punch register value")
parser.add_argument("--armature", type=float, default=0.00054,
                    help="Reflected rotor inertia [kg m^2]")
parser.add_argument("--coulomb-friction", type=float, default=0.04,
                    help="Explicit gearbox Coulomb friction [N m]")
parser.add_argument("--viscous-friction", type=float, default=0.0,
                    help="Additional viscous gearbox friction [N m s/rad]")
parser.add_argument("--base-height", type=float, default=0.6,
                    help="Fixed base height, chosen to keep both feet airborne [m]")
parser.add_argument("--robot-spacing", type=float, default=0.55,
                    help="Spacing between damping-sweep robots [m]")
parser.add_argument("--output-root", type=str, default="logs/ax18a_sysid/sim",
                    help="Root directory for timestamped result directories")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# Heavy imports must follow AppLauncher.
import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
from isaaclab.assets import Articulation
from isaaclab.sim import PhysxCfg, SimulationCfg, SimulationContext

from isaaclab_tasks.direct.SOLO_DEXTRA.actuators import AX18AActuatorCfg
from isaaclab_tasks.direct.SOLO_DEXTRA.dextra_robot_cfg import DEXTRA_CFG


AX18A_RAD_PER_TICK = math.radians(0.29)


def validate_args() -> None:
    if not args_cli.damping_values:
        raise ValueError("--damping-values must contain at least one value")
    if any(value < 0.0 for value in args_cli.damping_values):
        raise ValueError("all damping values must be non-negative")
    if len(set(args_cli.damping_values)) != len(args_cli.damping_values):
        raise ValueError("--damping-values must not contain duplicates")
    if args_cli.step_deg <= 0.0:
        raise ValueError("--step-deg must be positive")
    if args_cli.physics_hz <= 0.0:
        raise ValueError("--physics-hz must be positive")
    if not 0.0 < args_cli.torque_limit_ratio <= 1.0:
        raise ValueError("--torque-limit-ratio must be in (0, 1]")
    if not 0 <= args_cli.compliance_margin <= 255:
        raise ValueError("--compliance-margin must be in [0, 255]")
    if not 1 <= args_cli.compliance_slope <= 254:
        raise ValueError("--compliance-slope must be in [1, 254]")
    for name in ("settle_s", "step_hold_s", "center_hold_s", "final_hold_s"):
        if getattr(args_cli, name) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def damping_key(value: float) -> str:
    return f"{value:.6g}"


def create_output_dir() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.abspath(os.path.join(args_cli.output_root, f"{args_cli.joint_name}_{timestamp}"))
    os.makedirs(output_dir, exist_ok=False)
    return output_dir


def build_robot_cfg(damping: float, robot_index: int) -> object:
    """Create one fixed-base full robot with the requested damping value."""
    cfg = copy.deepcopy(DEXTRA_CFG)
    cfg.prim_path = f"/World/Robot_{robot_index:02d}"
    cfg.spawn.fix_base = True
    cfg.init_state.pos = (
        (robot_index - 0.5 * (len(args_cli.damping_values) - 1)) * args_cli.robot_spacing,
        0.0,
        args_cli.base_height,
    )
    cfg.actuators = {
        "legs": AX18AActuatorCfg(
            joint_names_expr=[".*HipYaw.*", ".*HipRoll.*", ".*Thigh.*", ".*Calf.*", ".*Ankle.*"],
            stall_torque=args_cli.stall_torque,
            effort_limit=args_cli.stall_torque * args_cli.torque_limit_ratio,
            velocity_limit=args_cli.velocity_limit,
            damping=damping,
            armature=args_cli.armature,
            friction=0.0,
            coulomb_friction=args_cli.coulomb_friction,
            viscous_friction_coeff=args_cli.viscous_friction,
            compliance_margin=args_cli.compliance_margin * AX18A_RAD_PER_TICK,
            compliance_slope=args_cli.compliance_slope,
            punch=args_cli.punch,
        )
    }
    return cfg


def build_command_schedule() -> tuple[list[dict], float]:
    segments = [
        {"phase": "settle", "duration": args_cli.settle_s, "offset_rad": 0.0},
        {"phase": "positive", "duration": args_cli.step_hold_s, "offset_rad": math.radians(args_cli.step_deg)},
        {"phase": "center", "duration": args_cli.center_hold_s, "offset_rad": 0.0},
        {"phase": "negative", "duration": args_cli.step_hold_s, "offset_rad": -math.radians(args_cli.step_deg)},
        {"phase": "final", "duration": args_cli.final_hold_s, "offset_rad": 0.0},
    ]
    cursor = 0.0
    for segment in segments:
        segment["start_s"] = cursor
        cursor += segment["duration"]
        segment["end_s"] = cursor
    return segments, cursor


def command_at_time(time_s: float, segments: list[dict]) -> tuple[str, float]:
    for segment in segments:
        if time_s < segment["end_s"] - 1.0e-12:
            return segment["phase"], segment["offset_rad"]
    last = segments[-1]
    return last["phase"], last["offset_rad"]


def write_csv(output_dir: str, records: dict[float, dict[str, list]]) -> None:
    path = os.path.join(output_dir, "responses.csv")
    fields = [
        "time_s", "phase", "damping_nms_per_rad", "target_rad", "position_rad",
        "velocity_rad_s", "position_error_rad", "damping_torque_nm",
        "computed_torque_nm", "applied_torque_nm", "effort_limit_nm",
    ]
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for damping in args_cli.damping_values:
            data = records[damping]
            for row_idx in range(len(data["time_s"])):
                writer.writerow({name: data[name][row_idx] for name in fields})


def compute_summary(records: dict[float, dict[str, list]], baseline_rad: float) -> dict:
    step_rad = math.radians(args_cli.step_deg)
    summary = {}
    for damping in args_cli.damping_values:
        data = records[damping]
        target = np.asarray(data["target_rad"])
        position = np.asarray(data["position_rad"])
        velocity = np.asarray(data["velocity_rad_s"])
        applied = np.asarray(data["applied_torque_nm"])
        computed = np.asarray(data["computed_torque_nm"])
        phases = np.asarray(data["phase"])

        positive = position[phases == "positive"]
        negative = position[phases == "negative"]
        tail_count_pos = max(1, int(0.2 * len(positive)))
        tail_count_neg = max(1, int(0.2 * len(negative)))
        positive_target = baseline_rad + step_rad
        negative_target = baseline_rad - step_rad

        summary[damping_key(damping)] = {
            "damping_nms_per_rad": damping,
            "position_rmse_rad": float(np.sqrt(np.mean((target - position) ** 2))),
            "max_abs_velocity_rad_s": float(np.max(np.abs(velocity))),
            "max_abs_computed_torque_nm": float(np.max(np.abs(computed))),
            "max_abs_applied_torque_nm": float(np.max(np.abs(applied))),
            "positive_peak_rad": float(np.max(positive)),
            "positive_overshoot_deg": float(math.degrees(max(0.0, np.max(positive) - positive_target))),
            "positive_tail_error_deg": float(math.degrees(np.mean(positive[-tail_count_pos:]) - positive_target)),
            "negative_peak_rad": float(np.min(negative)),
            "negative_overshoot_deg": float(math.degrees(max(0.0, negative_target - np.min(negative)))),
            "negative_tail_error_deg": float(math.degrees(np.mean(negative[-tail_count_neg:]) - negative_target)),
        }
    return summary


def save_plot(output_dir: str, records: dict[float, dict[str, list]], segments: list[dict], effort_limit: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, len(args_cli.damping_values)))

    reference = records[args_cli.damping_values[0]]
    time_s = np.asarray(reference["time_s"])
    axes[0].step(time_s, reference["target_rad"], where="post", color="black", linewidth=2.0,
                 linestyle="--", label="position command")

    for color, damping in zip(colors, args_cli.damping_values):
        data = records[damping]
        label = f"damping={damping:g}"
        axes[0].plot(data["time_s"], data["position_rad"], color=color, linewidth=1.4, label=label)
        axes[1].plot(data["time_s"], data["velocity_rad_s"], color=color, linewidth=1.2, label=label)
        axes[2].plot(data["time_s"], data["applied_torque_nm"], color=color, linewidth=1.1, label=label)

    axes[2].axhline(+effort_limit, color="black", linestyle=":", linewidth=1.0,
                    label=f"compliance effort limit ±{effort_limit:.3f} N m")
    axes[2].axhline(-effort_limit, color="black", linestyle=":", linewidth=1.0)

    for axis in axes:
        for segment in segments[1:]:
            axis.axvline(segment["start_s"], color="gray", linewidth=0.8, alpha=0.45)
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("position [rad]")
    axes[1].set_ylabel("velocity [rad/s]")
    axes[2].set_ylabel("applied torque [N m]")
    axes[2].set_xlabel("time [s]")
    axes[0].legend(ncol=3, fontsize=9)
    axes[1].legend(ncol=3, fontsize=9)
    axes[2].legend(ncol=3, fontsize=9)
    fig.suptitle(
        f"Fixed-base AX-18A step response — {args_cli.joint_name}, ±{args_cli.step_deg:g}°, "
        f"torque ratio={args_cli.torque_limit_ratio:g}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(os.path.join(output_dir, "step_response.png"), dpi=170)
    plt.close(fig)


def main() -> None:
    validate_args()
    output_dir = create_output_dir()
    sim_dt = 1.0 / args_cli.physics_hz
    effort_limit = args_cli.stall_torque * args_cli.torque_limit_ratio

    sim_cfg = SimulationCfg(
        dt=sim_dt,
        render_interval=max(1, round(args_cli.physics_hz / 30.0)),
        device=args_cli.device,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.2, 2.0, 1.4], target=[0.0, 0.0, args_cli.base_height - 0.15])

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    robots: list[Articulation] = []
    for robot_index, damping in enumerate(args_cli.damping_values):
        robots.append(Articulation(build_robot_cfg(damping, robot_index)))

    sim.reset()
    joint_indices: list[int] = []
    baseline_rad = math.radians(args_cli.baseline_deg)
    baseline_targets: list[torch.Tensor] = []
    for robot in robots:
        if args_cli.joint_name not in robot.data.joint_names:
            raise ValueError(
                f"unknown --joint-name {args_cli.joint_name!r}; available: {robot.data.joint_names}"
            )
        joint_index = robot.data.joint_names.index(args_cli.joint_name)
        joint_indices.append(joint_index)
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = torch.zeros_like(joint_pos)
        joint_pos[:, joint_index] = baseline_rad
        target = joint_pos.clone()
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        baseline_targets.append(target)

    segments, total_duration = build_command_schedule()
    total_steps = math.ceil(total_duration / sim_dt)
    records: dict[float, dict[str, list]] = {}
    fields = [
        "time_s", "phase", "damping_nms_per_rad", "target_rad", "position_rad",
        "velocity_rad_s", "position_error_rad", "damping_torque_nm",
        "computed_torque_nm", "applied_torque_nm", "effort_limit_nm",
    ]
    for damping in args_cli.damping_values:
        records[damping] = {field: [] for field in fields}

    print("\n[sysid-sim] Fixed-base AX-18A damping sweep")
    print(f"  Joint:              {args_cli.joint_name}")
    print(f"  Damping values:     {args_cli.damping_values}")
    print(f"  Command:            ±{args_cli.step_deg:g} deg from {args_cli.baseline_deg:g} deg")
    print(f"  Torque limit:       {effort_limit:.4f} N m (ratio={args_cli.torque_limit_ratio:g})")
    print(f"  Physics/log rate:   {args_cli.physics_hz:g} Hz")
    print(f"  Duration:           {total_duration:.3f} s ({total_steps} steps)")
    print(f"  Output:             {output_dir}\n")

    for step_index in range(total_steps):
        if not simulation_app.is_running():
            print("[sysid-sim] Simulation app stopped before the experiment completed")
            break
        command_time = step_index * sim_dt
        phase, offset_rad = command_at_time(command_time, segments)
        target_rad = baseline_rad + offset_rad

        # Log the state at command_time together with the torque computed from
        # that same state. The torque is then applied over [t, t + sim_dt).
        # This avoids pairing a pre-step torque with a post-step velocity.
        for damping, robot, joint_index, baseline_target in zip(
            args_cli.damping_values, robots, joint_indices, baseline_targets
        ):
            target = baseline_target.clone()
            target[:, joint_index] = target_rad
            robot.set_joint_position_target(target)
            robot.write_data_to_sim()

            position = float(robot.data.joint_pos[0, joint_index].item())
            velocity = float(robot.data.joint_vel[0, joint_index].item())
            computed_torque = float(robot.data.computed_torque[0, joint_index].item())
            applied_torque = float(robot.data.applied_torque[0, joint_index].item())
            row = {
                "time_s": command_time,
                "phase": phase,
                "damping_nms_per_rad": damping,
                "target_rad": target_rad,
                "position_rad": position,
                "velocity_rad_s": velocity,
                "position_error_rad": target_rad - position,
                "damping_torque_nm": -damping * velocity,
                "computed_torque_nm": computed_torque,
                "applied_torque_nm": applied_torque,
                "effort_limit_nm": effort_limit,
            }
            for field, value in row.items():
                records[damping][field].append(value)

        sim.step()
        for robot in robots:
            robot.update(sim_dt)

    write_csv(output_dir, records)
    summary = compute_summary(records, baseline_rad)
    with open(os.path.join(output_dir, "summary.json"), "w") as file:
        json.dump(summary, file, indent=2)

    config = {
        "joint_name": args_cli.joint_name,
        "damping_values": args_cli.damping_values,
        "step_deg": args_cli.step_deg,
        "baseline_deg": args_cli.baseline_deg,
        "physics_hz": args_cli.physics_hz,
        "fixed_base": True,
        "base_height_m": args_cli.base_height,
        "domain_randomization": False,
        "segments": segments,
        "actuator": {
            "stall_torque_nm": args_cli.stall_torque,
            "effort_limit_ratio": args_cli.torque_limit_ratio,
            "effort_limit_nm": effort_limit,
            "velocity_limit_rad_s": args_cli.velocity_limit,
            "compliance_margin_register": args_cli.compliance_margin,
            "compliance_slope_register": args_cli.compliance_slope,
            "punch_register": args_cli.punch,
            "armature_kgm2": args_cli.armature,
            "coulomb_friction_nm": args_cli.coulomb_friction,
            "viscous_friction_nms_per_rad": args_cli.viscous_friction,
        },
    }
    with open(os.path.join(output_dir, "config.json"), "w") as file:
        json.dump(config, file, indent=2)

    save_plot(output_dir, records, segments, effort_limit)
    print("[sysid-sim] Done")
    print(f"  CSV:     {os.path.join(output_dir, 'responses.csv')}")
    print(f"  Plot:    {os.path.join(output_dir, 'step_response.png')}")
    print(f"  Summary: {os.path.join(output_dir, 'summary.json')}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
