"""Open-loop reference-motion tracking test for the Dextra AMP environment.

This test removes the policy from the loop. At every policy step it samples the
same ``MotionLoader`` used by AMP, converts the reference joint positions to the
environment's normalized action, and lets the configured actuator/physics track
that target. It logs tracking error and the implicit actuator's approximate
pre/post-clipping torque.

Example (from the IsaacLab root)::

    ./isaaclab.sh -p \
        source/isaaclab_tasks/isaaclab_tasks/direct/SOLO_DEXTRA/reference_tracking_test.py \
        --headless --stiffness 5.4 --damping 0.4 --effort-limit 0.54

The default output directory is ``logs/reference_tracking/<timestamp>``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Track an AMP reference motion without an RL policy")
parser.add_argument("--motion-file", type=str, default=None, help="Motion NPZ (default: environment config)")
parser.add_argument("--speed-scale", type=float, default=None, help="MotionLoader time-stretch factor")
parser.add_argument("--stiffness", type=float, default=None, help="Override implicit-drive stiffness [N m/rad]")
parser.add_argument("--damping", type=float, default=None, help="Override implicit-drive damping [N m s/rad]")
parser.add_argument("--effort-limit", type=float, default=None, help="Override torque limit [N m]")
parser.add_argument("--dead-zone-deg", type=float, default=None, help="Override position dead zone [deg]")
parser.add_argument("--cycles", type=float, default=1.0, help="Number of motion cycles to run")
parser.add_argument("--steps", type=int, default=None, help="Exact policy steps (overrides --cycles)")
parser.add_argument(
    "--start",
    choices=("reference", "default"),
    default="reference",
    help="Initialize from motion frame 0 or the robot default pose",
)
parser.add_argument(
    "--summary-ignore-seconds",
    type=float,
    default=0.2,
    help="Exclude this initial interval from steady tracking summary",
)
parser.add_argument("--output-dir", type=str, default=None, help="Directory for CSV, JSON, NPZ and plot")
parser.add_argument("--print-every", type=int, default=30, help="Console report interval in policy steps")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Do not pass this script's arguments to modules imported after AppLauncher.
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# Heavy imports must follow AppLauncher.
import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401


def _sample_reference(env, time_s: float):
    """Sample and reorder the reference into the articulation joint order."""
    motion = env._motion_loader
    sample = motion.sample(num_samples=1, times=np.asarray([time_s], dtype=np.float64))
    q = sample[0][:, env.motion_dof_indexes]
    qd = sample[1][:, env.motion_dof_indexes]
    return q, qd, sample


def _initialize_from_reference(env) -> None:
    """Put the robot at reference frame zero without kinematically replaying it later."""
    q, qd, sample = _sample_reference(env, 0.0)
    body_pos, body_quat, body_lin_vel, body_ang_vel = sample[2], sample[3], sample[4], sample[5]
    base_idx = env._motion_loader.get_body_index([env.cfg.reference_body])[0]

    root_state = env.robot.data.default_root_state.clone()
    root_state[:, 0:3] = body_pos[:, base_idx] + env.scene.env_origins
    root_state[:, 3:7] = body_quat[:, base_idx]
    root_state[:, 7:10] = body_lin_vel[:, base_idx]
    root_state[:, 10:13] = body_ang_vel[:, base_idx]

    env.robot.write_root_link_pose_to_sim(root_state[:, :7])
    env.robot.write_root_com_velocity_to_sim(root_state[:, 7:])
    env.robot.write_joint_state_to_sim(q, qd)
    env.robot.set_joint_position_target(q)
    env.robot.write_data_to_sim()


def _positive_lag_seconds(reference: np.ndarray, actual: np.ndarray, step_dt: float) -> np.ndarray:
    """Return per-joint lag; positive means actual motion trails the reference."""
    max_lag = min(int(round(0.5 / step_dt)), max(1, reference.shape[0] // 4))
    lags = np.zeros(reference.shape[1], dtype=np.float64)
    for joint_id in range(reference.shape[1]):
        ref = reference[:, joint_id] - reference[:, joint_id].mean()
        act = actual[:, joint_id] - actual[:, joint_id].mean()
        if np.linalg.norm(ref) < 1.0e-8 or np.linalg.norm(act) < 1.0e-8:
            lags[joint_id] = np.nan
            continue
        best_score = -np.inf
        best_lag = 0
        for lag in range(-max_lag, max_lag + 1):
            if lag > 0:
                ref_part, act_part = ref[:-lag], act[lag:]
            elif lag < 0:
                ref_part, act_part = ref[-lag:], act[:lag]
            else:
                ref_part, act_part = ref, act
            denom = np.linalg.norm(ref_part) * np.linalg.norm(act_part)
            score = float(np.dot(ref_part, act_part) / denom) if denom > 1.0e-12 else -np.inf
            if score > best_score:
                best_score, best_lag = score, lag
        lags[joint_id] = best_lag * step_dt
    return lags


def _save_plot(output_dir: Path, arrays: dict[str, np.ndarray], joint_names: list[str]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # plotting is optional; data files are the primary output
        print(f"[tracking] plot skipped: {exc}")
        return

    time_s = arrays["time_s"]
    q_ref, q = arrays["q_ref"], arrays["q"]
    error = q_ref - q
    saturation = arrays["saturated"].mean(axis=0) * 100.0

    fig, axes = plt.subplots(4, 1, figsize=(15, 14), constrained_layout=True)
    for index, name in enumerate(joint_names):
        axes[0].plot(time_s, q_ref[:, index], "--", linewidth=0.9, alpha=0.75, label=f"ref {name}")
        axes[0].plot(time_s, q[:, index], linewidth=0.9, alpha=0.75, label=f"sim {name}")
    axes[0].set_ylabel("joint position [rad]")
    axes[0].grid(alpha=0.25)
    axes[0].legend(ncol=4, fontsize=7)

    axes[1].plot(time_s, error)
    axes[1].set_ylabel("q_ref - q [rad]")
    axes[1].grid(alpha=0.25)

    axes[2].bar(np.arange(len(joint_names)), saturation)
    axes[2].set_xticks(np.arange(len(joint_names)), joint_names, rotation=45, ha="right")
    axes[2].set_ylabel("torque saturation [%]")
    axes[2].set_ylim(0.0, 100.0)
    axes[2].grid(axis="y", alpha=0.25)

    axes[3].plot(time_s, arrays["base_height"], label="base height [m]")
    axes[3].plot(time_s, arrays["base_vx"], label="base vx [m/s]")
    axes[3].plot(time_s, arrays["left_foot_height"], label="left foot z [m]", alpha=0.8)
    axes[3].plot(time_s, arrays["right_foot_height"], label="right foot z [m]", alpha=0.8)
    axes[3].set_xlabel("time [s]")
    axes[3].grid(alpha=0.25)
    axes[3].legend()

    fig.savefig(output_dir / "tracking.png", dpi=150)
    plt.close(fig)


def main() -> None:
    from isaaclab_tasks.direct.SOLO_DEXTRA.dextra_amp_env_cfg import DextraAmpWalkEnvCfg

    cfg = DextraAmpWalkEnvCfg()
    cfg.scene.num_envs = 1
    cfg.early_termination = False  # detect and log a fall without automatic reset
    cfg.vel_window_min_vx = 0.0
    cfg.termination_min_vel_x = 0.0
    cfg.action_rate_penalty_weight = 0.0
    cfg.events = None  # nominal deterministic feasibility test, with no startup DR

    if args_cli.motion_file is not None:
        cfg.motion_file = os.path.abspath(args_cli.motion_file)
    if args_cli.speed_scale is not None:
        if args_cli.speed_scale <= 0.0:
            raise ValueError("--speed-scale must be positive")
        cfg.motion_speed_scale = float(args_cli.speed_scale)

    actuator_cfg = cfg.robot.actuators["legs"]
    if args_cli.stiffness is not None:
        actuator_cfg.stiffness = float(args_cli.stiffness)
    if args_cli.damping is not None:
        actuator_cfg.damping = float(args_cli.damping)
    if args_cli.effort_limit is not None:
        if args_cli.effort_limit <= 0.0:
            raise ValueError("--effort-limit must be positive")
        actuator_cfg.effort_limit_sim = float(args_cli.effort_limit)
        actuator_cfg.effort_limit = None
        cfg.effort_limit_ratio = float(args_cli.effort_limit) / float(cfg.ax18a_stall_torque)
        cfg.effort_limit_ratio_range = (cfg.effort_limit_ratio, cfg.effort_limit_ratio)
    if args_cli.dead_zone_deg is not None:
        if args_cli.dead_zone_deg < 0.0:
            raise ValueError("--dead-zone-deg must be non-negative")
        actuator_cfg.dead_zone = math.radians(float(args_cli.dead_zone_deg))

    env = gym.make("Isaac-Dextra-Amp-Walk-Direct-v0", cfg=cfg)
    raw_env = env.unwrapped
    env.reset()
    if args_cli.start == "reference":
        _initialize_from_reference(raw_env)

    policy_dt = float(raw_env.step_dt)
    duration = float(raw_env._motion_loader.duration)
    num_steps = args_cli.steps
    if num_steps is None:
        if args_cli.cycles <= 0.0:
            raise ValueError("--cycles must be positive")
        num_steps = max(1, int(math.ceil(duration * float(args_cli.cycles) / policy_dt)))
    cfg.episode_length_s = max(cfg.episode_length_s, (num_steps + 2) * policy_dt)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args_cli.output_dir or f"logs/reference_tracking/{timestamp}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    joint_names = list(raw_env.robot.data.joint_names)
    ref_body_idx = raw_env.ref_body_index
    left_foot_idx, right_foot_idx = raw_env.key_body_indexes
    series: dict[str, list[np.ndarray | float | bool]] = {
        "time_s": [],
        "motion_time_s": [],
        "q_ref": [],
        "qd_ref": [],
        "q": [],
        "qd": [],
        "computed_torque": [],
        "applied_torque": [],
        "effort_limit": [],
        "saturated": [],
        "base_height": [],
        "base_vx": [],
        "left_foot_height": [],
        "right_foot_height": [],
        "fallen": [],
    }

    print("\n" + "=" * 72)
    print("Reference tracking test (policy bypassed)")
    print(f"motion:       {cfg.motion_file}")
    print(f"duration:     {duration:.4f} s, policy dt: {policy_dt:.6f} s, steps: {num_steps}")
    print(f"start:        {args_cli.start}")
    print(f"stiffness:    {actuator_cfg.stiffness}")
    print(f"damping:      {actuator_cfg.damping}")
    print(f"effort limit: {actuator_cfg.effort_limit_sim} N m")
    print(f"dead zone:    {math.degrees(float(actuator_cfg.dead_zone)):.4f} deg")
    print(f"output:       {output_dir}")
    print("=" * 72 + "\n")

    first_fall_time: float | None = None
    with torch.inference_mode():
        for step in range(num_steps):
            if not simulation_app.is_running():
                print("[tracking] simulation stopped early")
                break

            sim_time = (step + 1) * policy_dt
            # One-cycle endpoint stays at the last frame; additional cycles wrap.
            if sim_time <= duration:
                motion_time = sim_time
            else:
                motion_time = sim_time % duration
            q_ref, qd_ref, _ = _sample_reference(raw_env, motion_time)

            action = (q_ref - raw_env.action_offset) / raw_env.action_scale
            if torch.any(torch.abs(action) > 1.0 + 1.0e-6):
                raise RuntimeError("Reference joint position exceeds the environment action range")

            env.step(action.clamp(-1.0, 1.0))
            data = raw_env.robot.data

            q = data.joint_pos[0].detach().cpu().numpy().copy()
            qd = data.joint_vel[0].detach().cpu().numpy().copy()
            computed = data.computed_torque[0].detach().cpu().numpy().copy()
            applied = data.applied_torque[0].detach().cpu().numpy().copy()
            limit = data.joint_effort_limits[0].detach().cpu().numpy().copy()
            saturated = np.abs(computed) >= (np.abs(limit) - 1.0e-5)
            base_height = float(data.body_pos_w[0, ref_body_idx, 2].item())
            fallen = base_height < float(raw_env.cfg.termination_height)
            if fallen and first_fall_time is None:
                first_fall_time = sim_time

            series["time_s"].append(sim_time)
            series["motion_time_s"].append(motion_time)
            series["q_ref"].append(q_ref[0].detach().cpu().numpy().copy())
            series["qd_ref"].append(qd_ref[0].detach().cpu().numpy().copy())
            series["q"].append(q)
            series["qd"].append(qd)
            series["computed_torque"].append(computed)
            series["applied_torque"].append(applied)
            series["effort_limit"].append(limit)
            series["saturated"].append(saturated)
            series["base_height"].append(base_height)
            series["base_vx"].append(float(data.body_lin_vel_w[0, ref_body_idx, 0].item()))
            series["left_foot_height"].append(float(data.body_pos_w[0, left_foot_idx, 2].item()))
            series["right_foot_height"].append(float(data.body_pos_w[0, right_foot_idx, 2].item()))
            series["fallen"].append(fallen)

            if step == 0 or (step + 1) % max(1, args_cli.print_every) == 0 or fallen:
                q_rmse = float(np.sqrt(np.mean((q_ref[0].cpu().numpy() - q) ** 2)))
                sat_pct = float(saturated.mean() * 100.0)
                print(
                    f"[step {step + 1:4d}/{num_steps}] t={sim_time:7.3f}s "
                    f"q_rmse={q_rmse:.4f} rad sat={sat_pct:5.1f}% "
                    f"z={base_height:.3f}m vx={series['base_vx'][-1]:+.3f}m/s"
                )

    env.close()
    arrays = {key: np.asarray(value) for key, value in series.items()}
    if arrays["time_s"].size == 0:
        raise RuntimeError("No tracking samples were collected")

    ignore_steps = min(
        int(round(max(0.0, args_cli.summary_ignore_seconds) / policy_dt)),
        max(0, arrays["time_s"].shape[0] - 1),
    )
    steady = slice(ignore_steps, None)
    q_error = arrays["q_ref"] - arrays["q"]
    qd_error = arrays["qd_ref"] - arrays["qd"]
    q_rmse = np.sqrt(np.mean(q_error[steady] ** 2, axis=0))
    qd_rmse = np.sqrt(np.mean(qd_error[steady] ** 2, axis=0))
    saturation_fraction = arrays["saturated"][steady].mean(axis=0)
    lags = _positive_lag_seconds(arrays["q_ref"][steady], arrays["q"][steady], policy_dt)

    actuator = raw_env.robot.actuators["legs"]
    actual_stiffness = actuator.stiffness[0].detach().cpu().numpy().tolist()
    actual_damping = actuator.damping[0].detach().cpu().numpy().tolist()
    summary = {
        "motion_file": os.path.abspath(cfg.motion_file),
        "motion_speed_scale": float(cfg.motion_speed_scale),
        "motion_duration_s": duration,
        "policy_dt_s": policy_dt,
        "steps_collected": int(arrays["time_s"].shape[0]),
        "start": args_cli.start,
        "summary_ignore_seconds": float(args_cli.summary_ignore_seconds),
        "joint_names": joint_names,
        "configured_stiffness": actuator_cfg.stiffness,
        "configured_damping": actuator_cfg.damping,
        "configured_effort_limit_nm": actuator_cfg.effort_limit_sim,
        "configured_dead_zone_deg": math.degrees(float(actuator_cfg.dead_zone)),
        "actual_stiffness_per_joint": actual_stiffness,
        "actual_damping_per_joint": actual_damping,
        "q_rmse_rad_overall": float(np.sqrt(np.mean(q_error[steady] ** 2))),
        "qd_rmse_rad_s_overall": float(np.sqrt(np.mean(qd_error[steady] ** 2))),
        "saturation_fraction_overall": float(arrays["saturated"][steady].mean()),
        "q_rmse_rad_per_joint": dict(zip(joint_names, q_rmse.tolist())),
        "qd_rmse_rad_s_per_joint": dict(zip(joint_names, qd_rmse.tolist())),
        "saturation_fraction_per_joint": dict(zip(joint_names, saturation_fraction.tolist())),
        "phase_lag_s_per_joint": dict(zip(joint_names, lags.tolist())),
        "mean_base_vx_m_s": float(arrays["base_vx"][steady].mean()),
        "min_base_height_m": float(arrays["base_height"].min()),
        "first_fall_time_s": first_fall_time,
        "torque_note": (
            "Implicit-actuator computed/applied torque is Isaac Lab's PD-law approximation; "
            "PhysX does not expose the exact implicit-drive torque."
        ),
    }

    np.savez_compressed(output_dir / "tracking.npz", **arrays)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    scalar_names = [
        "time_s",
        "motion_time_s",
        "base_height",
        "base_vx",
        "left_foot_height",
        "right_foot_height",
        "fallen",
    ]
    vector_names = [
        "q_ref",
        "qd_ref",
        "q",
        "qd",
        "computed_torque",
        "applied_torque",
        "effort_limit",
        "saturated",
    ]
    with open(output_dir / "tracking.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        header = scalar_names.copy()
        for field in vector_names:
            header.extend(f"{field}_{joint}" for joint in joint_names)
        writer.writerow(header)
        for row_id in range(arrays["time_s"].shape[0]):
            row = [arrays[field][row_id] for field in scalar_names]
            for field in vector_names:
                row.extend(arrays[field][row_id].tolist())
            writer.writerow(row)

    _save_plot(output_dir, arrays, joint_names)

    print("\n" + "=" * 72)
    print(f"q RMSE:             {summary['q_rmse_rad_overall']:.5f} rad")
    print(f"qd RMSE:            {summary['qd_rmse_rad_s_overall']:.5f} rad/s")
    print(f"torque saturation:  {100.0 * summary['saturation_fraction_overall']:.2f}%")
    print(f"mean base vx:       {summary['mean_base_vx_m_s']:+.4f} m/s")
    print(f"minimum base height:{summary['min_base_height_m']:.4f} m")
    print(f"first fall:         {summary['first_fall_time_s']}")
    print(f"saved:              {output_dir}")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
