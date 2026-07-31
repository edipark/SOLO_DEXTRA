#!/usr/bin/env python3
"""Sweep AX-18A damping values using the active Dextra AMP configuration.

The simulation and robot/actuator configurations are inherited from
``DextraAmpEnvCfg``.  The only actuator parameter changed by this experiment
is ``damping``.  One complete robot is created per damping value, rigidly
fixed at the base and suspended above the ground.  All joints hold their
baseline target while one selected joint receives the same positive and
negative position steps.  No policy, estimator, termination, or domain
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
import sys
import threading
from datetime import datetime

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Fixed-base AX-18A damping sweep")
parser.add_argument("--joint-name", type=str, default="L_Thigh_Joint",
                    help="Exact robot joint name to excite")
parser.add_argument("--damping-values", type=float, nargs="+",
                    default=None,
                    help="Actuator damping values [N m s/rad]. Default: 0.5/0.8/1.0/1.2/1.5x AMP nominal")
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
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from isaaclab_tasks.direct.SOLO_DEXTRA.dextra_amp_env_cfg import DextraAmpEnvCfg


_active_sim: SimulationContext | None = None


@configclass
class DextraAmpStepResponseCfg(DextraAmpEnvCfg):
    """Step-response config whose dynamics come directly from AMP training."""

    pass


def validate_args() -> None:
    if not args_cli.damping_values:
        raise ValueError("--damping-values must contain at least one value")
    if any(value < 0.0 for value in args_cli.damping_values):
        raise ValueError("all damping values must be non-negative")
    if len(set(args_cli.damping_values)) != len(args_cli.damping_values):
        raise ValueError("--damping-values must not contain duplicates")
    if args_cli.step_deg <= 0.0:
        raise ValueError("--step-deg must be positive")
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


def build_robot_cfg(base_robot_cfg: object, damping: float, robot_index: int) -> object:
    """Copy the AMP robot config and override only actuator damping."""
    cfg = copy.deepcopy(base_robot_cfg)
    cfg.prim_path = f"/World/Robot_{robot_index:02d}"
    cfg.spawn.fix_base = True
    cfg.init_state.pos = (
        (robot_index - 0.5 * (len(args_cli.damping_values) - 1)) * args_cli.robot_spacing,
        0.0,
        args_cli.base_height,
    )
    if "legs" not in cfg.actuators:
        raise KeyError("DextraAmpEnvCfg.robot must define a 'legs' actuator group")
    cfg.actuators["legs"].damping = damping
    return cfg


def optional_float(cfg: object, name: str) -> float | None:
    """Return a scalar config field as float while preserving missing values."""
    value = getattr(cfg, name, None)
    return None if value is None else float(value)


def actuator_effort_limit(actuator_cfg: object) -> float:
    """Resolve the active simulation effort limit for implicit or explicit configs."""
    value = getattr(actuator_cfg, "effort_limit_sim", None)
    if value is None:
        value = getattr(actuator_cfg, "effort_limit", None)
    if value is None:
        raise ValueError("AMP actuator config has no simulation effort limit")
    return float(value)


def actuator_config_snapshot(actuator_cfg: object) -> dict:
    """Record the inherited actuator values needed to reproduce this run."""
    class_type = getattr(actuator_cfg, "class_type", None)
    snapshot = {
        "config_class": type(actuator_cfg).__name__,
        "actuator_class": getattr(class_type, "__name__", str(class_type)),
        "joint_names_expr": list(getattr(actuator_cfg, "joint_names_expr", [])),
        "damping_values_nms_per_rad": args_cli.damping_values,
    }
    for field in (
        "stiffness",
        "stall_torque",
        "effort_limit",
        "effort_limit_sim",
        "velocity_limit",
        "velocity_limit_sim",
        "armature",
        "friction",
        "coulomb_friction",
        "viscous_friction_coeff",
        "compliance_margin",
        "compliance_slope",
        "punch",
    ):
        value = optional_float(actuator_cfg, field)
        if value is not None:
            snapshot[field] = value
    return snapshot


def use_amp_centered_damping_sweep(actuator_cfg: object) -> None:
    """Build the default sweep around the active AMP actuator damping."""
    if args_cli.damping_values is not None:
        return
    nominal_damping = optional_float(actuator_cfg, "damping")
    if nominal_damping is None or nominal_damping <= 0.0:
        raise ValueError("AMP actuator damping must be positive to construct the default sweep")
    args_cli.damping_values = [
        round(nominal_damping * multiplier, 6)
        for multiplier in (0.5, 0.8, 1.0, 1.2, 1.5)
    ]


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


def save_plot(
    output_dir: str,
    records: dict[float, dict[str, list]],
    segments: list[dict],
    effort_limit: float,
    torque_limit_ratio: float,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    damping_count = len(args_cli.damping_values)
    fig = plt.figure(figsize=(18, 3.8 * damping_count), constrained_layout=True)
    subfigures = fig.subfigures(nrows=damping_count, ncols=1, squeeze=False)

    for row_index, damping in enumerate(args_cli.damping_values):
        data = records[damping]
        time_s = np.asarray(data["time_s"])
        subfigure = subfigures[row_index, 0]
        subfigure.suptitle(f"damping = {damping:g} N m s/rad", fontsize=12, fontweight="bold")
        axes = subfigure.subplots(1, 3, sharex=True)

        axes[0].step(
            time_s, data["target_rad"], where="post", color="black", linewidth=1.8, linestyle="--"
        )
        axes[0].plot(time_s, data["position_rad"], color="tab:blue", linewidth=1.4)
        axes[1].plot(time_s, data["velocity_rad_s"], color="tab:orange", linewidth=1.2)
        axes[2].plot(time_s, data["applied_torque_nm"], color="tab:red", linewidth=1.1)
        axes[2].axhline(+effort_limit, color="black", linestyle=":", linewidth=1.0)
        axes[2].axhline(-effort_limit, color="black", linestyle=":", linewidth=1.0)

        axes[0].set_title("Position: command (black dashed), response (blue)", fontsize=10)
        axes[1].set_title("Joint velocity", fontsize=10)
        axes[2].set_title(f"Applied torque: effort limit ±{effort_limit:.3f} N m (dotted)", fontsize=10)
        axes[0].set_ylabel("position [rad]")
        axes[1].set_ylabel("velocity [rad/s]")
        axes[2].set_ylabel("torque [N m]")

        for axis in axes:
            for segment in segments[1:]:
                axis.axvline(segment["start_s"], color="gray", linewidth=0.8, alpha=0.45)
            axis.set_xlabel("time [s]")
            axis.grid(True, alpha=0.25)

    fig.suptitle(
        f"Fixed-base AX-18A step response — {args_cli.joint_name}, ±{args_cli.step_deg:g}°, "
        f"torque ratio={torque_limit_ratio:g}",
        fontsize=14,
    )
    fig.savefig(os.path.join(output_dir, "step_response.png"), dpi=170)
    plt.close(fig)


def _close_simulation_resources() -> None:
    """Release the raw simulation context and close Kit."""
    global _active_sim

    if _active_sim is not None:
        try:
            if not _active_sim.has_gui():
                _active_sim.stop()
            _active_sim.clear_all_callbacks()
            _active_sim.clear_instance()
        except Exception as error:
            print(f"[sysid-sim] WARNING: SimulationContext cleanup failed: {error}")
        finally:
            _active_sim = None

    try:
        simulation_app.close()
    except Exception as error:
        print(f"[sysid-sim] WARNING: Kit cleanup failed: {error}")


def shutdown_simulation(exit_code: int) -> None:
    """Bound the entire native cleanup sequence, including sim.stop()."""
    cleanup_thread = threading.Thread(target=_close_simulation_resources, daemon=True)
    cleanup_thread.start()
    cleanup_thread.join(timeout=15.0)
    if cleanup_thread.is_alive():
        print("[sysid-sim] WARNING: simulation shutdown timed out; forcing process exit")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)


def main() -> None:
    global _active_sim

    experiment_cfg = DextraAmpStepResponseCfg()
    base_robot_cfg = experiment_cfg.robot
    if "legs" not in base_robot_cfg.actuators:
        raise KeyError("DextraAmpEnvCfg.robot must define a 'legs' actuator group")
    base_actuator_cfg = base_robot_cfg.actuators["legs"]
    use_amp_centered_damping_sweep(base_actuator_cfg)
    validate_args()
    output_dir = create_output_dir()

    sim_cfg = copy.deepcopy(experiment_cfg.sim)
    sim_cfg.device = args_cli.device
    sim_dt = float(sim_cfg.dt)
    physics_hz = 1.0 / sim_dt
    effort_limit = actuator_effort_limit(base_actuator_cfg)
    stall_torque = optional_float(base_actuator_cfg, "stall_torque")
    torque_limit_ratio = (
        effort_limit / stall_torque
        if stall_torque is not None and stall_torque > 0.0
        else float(experiment_cfg.effort_limit_ratio)
    )

    sim = SimulationContext(sim_cfg)
    _active_sim = sim
    sim.set_camera_view(eye=[2.2, 2.0, 1.4], target=[0.0, 0.0, args_cli.base_height - 0.15])

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    robots: list[Articulation] = []
    for robot_index, damping in enumerate(args_cli.damping_values):
        robots.append(Articulation(build_robot_cfg(base_robot_cfg, damping, robot_index)))

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
    print(f"  Config base:         {type(experiment_cfg).__name__} -> DextraAmpEnvCfg")
    print(f"  Actuator model:      {type(base_actuator_cfg).__name__}")
    print(f"  Damping values:     {args_cli.damping_values}")
    print(f"  Command:            ±{args_cli.step_deg:g} deg from {args_cli.baseline_deg:g} deg")
    print(f"  Torque limit:       {effort_limit:.4f} N m (ratio={torque_limit_ratio:g})")
    print(f"  Physics/log rate:   {physics_hz:g} Hz (inherited)")
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
        "config_base": "DextraAmpEnvCfg",
        "actuator_override_fields": ["damping"],
        "damping_values": args_cli.damping_values,
        "step_deg": args_cli.step_deg,
        "baseline_deg": args_cli.baseline_deg,
        "physics_hz": physics_hz,
        "physics_dt_s": sim_dt,
        "policy_decimation": experiment_cfg.decimation,
        "fixed_base": True,
        "base_height_m": args_cli.base_height,
        "domain_randomization": False,
        "segments": segments,
        "effort_limit_ratio": torque_limit_ratio,
        "actuator": actuator_config_snapshot(base_actuator_cfg),
    }
    with open(os.path.join(output_dir, "config.json"), "w") as file:
        json.dump(config, file, indent=2)

    save_plot(output_dir, records, segments, effort_limit, torque_limit_ratio)
    print("[sysid-sim] Done")
    print(f"  CSV:     {os.path.join(output_dir, 'responses.csv')}")
    print(f"  Plot:    {os.path.join(output_dir, 'step_response.png')}")
    print(f"  Summary: {os.path.join(output_dir, 'summary.json')}")


if __name__ == "__main__":
    try:
        main()
    finally:
        shutdown_simulation(exit_code=1 if sys.exc_info()[0] is not None else 0)
