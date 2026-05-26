#!/usr/bin/env python3
"""SOLO Deployment: Run Teacher + LSTM Estimator on real DEXTRA robot.

Reads joint positions from 12 Dynamixel AX-18A servos, computes velocity
via finite difference, runs the LSTM estimator and teacher policy via
ONNX Runtime, and writes position targets back to the servos at 60 Hz.

Usage::

    # Full deployment
    python deploy.py --config config.yaml

    # Dry-run (no servo writes, for testing inference speed)
    python deploy.py --config config.yaml --dry-run

    # With logging
    python deploy.py --config config.yaml --log

    # Custom model paths
    python deploy.py --config config.yaml \
        --teacher_onnx models/teacher_policy.onnx \
        --estimator_onnx models/lstm_estimator.onnx
"""

import argparse
import signal
import sys
import time

import numpy as np
import yaml

from hardware.dynamixel_interface import DynamixelInterface
from inference.onnx_policy import TeacherPolicyONNX
from inference.onnx_estimator import LSTMEstimatorONNX
from utils.timing import RateController
from utils.logger import CSVLogger

# Matches simulation constants
NUM_JOINTS = 12
ENCODER_DIM = 24  # 12 pos + 12 vel
PRIV_DIM = 19
OBS_DIM = 43


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_hardware(cfg: dict) -> DynamixelInterface:
    joints = cfg["joints"]
    return DynamixelInterface(
        port=cfg["dynamixel"]["port"],
        baudrate=cfg["dynamixel"]["baudrate"],
        servo_ids=[j["servo_id"] for j in joints],
        offsets_raw=[j["offset_raw"] for j in joints],
        lower_rads=[j["lower_rad"] for j in joints],
        upper_rads=[j["upper_rad"] for j in joints],
    )


def main():
    parser = argparse.ArgumentParser(description="SOLO Robot Deployment")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--teacher_onnx", type=str, default=None,
                        help="Override teacher ONNX path")
    parser.add_argument("--estimator_onnx", type=str, default=None,
                        help="Override estimator ONNX path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Inference only, no servo writes")
    parser.add_argument("--log", action="store_true",
                        help="Enable CSV data logging")
    parser.add_argument("--duration", type=float, default=None,
                        help="Run duration in seconds (default: infinite)")
    args = parser.parse_args()

    # --- Load config ---
    cfg = load_config(args.config)
    freq = cfg["control"]["frequency_hz"]
    action_scale = cfg["control"]["action_scale"]
    action_offset = cfg["control"]["action_offset"]
    window = cfg["model"]["window"]
    safety = cfg["safety"]
    vel_filter_cfg = cfg.get("velocity_filter", {})
    ema_alpha = vel_filter_cfg.get("alpha", 0.2) if vel_filter_cfg.get("type") == "ema" else 1.0

    teacher_path = args.teacher_onnx or cfg["model"]["teacher_onnx"]
    estimator_path = args.estimator_onnx or cfg["model"]["estimator_onnx"]

    # Joint limits from config
    joint_lower = np.array([j["lower_rad"] for j in cfg["joints"]], dtype=np.float32)
    joint_upper = np.array([j["upper_rad"] for j in cfg["joints"]], dtype=np.float32)

    print(f"[deploy] Config: {freq}Hz, action_scale={action_scale}, window={window}")
    print(f"[deploy] Teacher: {teacher_path}")
    print(f"[deploy] Estimator: {estimator_path}")
    if args.dry_run:
        print("[deploy] DRY-RUN mode — no servo commands will be sent")

    # --- Initialize hardware ---
    dxl = None
    if not args.dry_run:
        dxl = build_hardware(cfg)
        dxl.connect()

    # --- Initialize inference ---
    policy = TeacherPolicyONNX(teacher_path)
    estimator = LSTMEstimatorONNX(estimator_path, window=window, encoder_dim=ENCODER_DIM)

    # --- Initialize logger ---
    logger = None
    if args.log:
        logger = CSVLogger(log_dir="logs", num_joints=NUM_JOINTS, priv_dim=PRIV_DIM)
        print(f"[deploy] Logging to {logger.filepath}")

    # --- Signal handler for clean shutdown ---
    running = True

    def sigint_handler(sig, frame):
        nonlocal running
        print("\n[deploy] Interrupt received, shutting down...")
        running = False

    signal.signal(signal.SIGINT, sigint_handler)

    # --- Read initial position ---
    if dxl:
        prev_pos = dxl.read_positions()
        # Hold current position briefly for safety
        print(f"[deploy] Holding start position for {safety['startup_hold_sec']}s...")
        dxl.write_position_targets(prev_pos)
        time.sleep(safety["startup_hold_sec"])
    else:
        prev_pos = np.zeros(NUM_JOINTS, dtype=np.float32)

    prev_time = time.monotonic()
    filtered_vel = np.zeros(NUM_JOINTS, dtype=np.float32)

    # --- Control loop ---
    rate = RateController(freq)
    rate.reset()
    step = 0
    watchdog_warn_ms = safety["watchdog_warn_ms"]
    watchdog_stop_ms = safety["watchdog_stop_ms"]

    print(f"[deploy] Starting control loop @ {freq}Hz...")

    while running:
        t0 = time.monotonic()

        # 1. Read joint positions
        if dxl:
            pos = dxl.read_positions()
        else:
            # Dry-run: simulate zero position
            pos = np.zeros(NUM_JOINTS, dtype=np.float32)

        # 2. Compute joint velocity via finite difference + EMA filter
        now = time.monotonic()
        dt = now - prev_time
        if dt > 0:
            raw_vel = (pos - prev_pos) / dt
        else:
            raw_vel = np.zeros(NUM_JOINTS, dtype=np.float32)

        filtered_vel = ema_alpha * raw_vel + (1.0 - ema_alpha) * filtered_vel

        # 3. Build encoder observation: [pos(12), vel(12)] = 24D
        encoder_obs = np.concatenate([pos, filtered_vel]).astype(np.float32)

        # 4. LSTM estimator: history(50×24) → priv_est(19)
        priv_est = estimator.update_and_predict(encoder_obs)

        # 5. Build full observation: [encoder(24), priv_est(19)] = 43D
        obs = np.concatenate([encoder_obs, priv_est]).astype(np.float32)

        # 6. Teacher policy: obs(43) → action(12) in [-1, 1]
        action = policy.predict(obs)

        # 7. Scale action to position targets
        action_clipped = np.clip(action, -1.0, 1.0)
        targets = action_offset + action_scale * action_clipped

        # 8. Safety clip to joint limits
        targets = np.clip(targets, joint_lower, joint_upper)

        # 9. Write to servos
        if dxl:
            dxl.write_position_targets(targets)

        # 10. Log
        loop_dt_ms = (time.monotonic() - t0) * 1000
        if logger:
            logger.log(loop_dt_ms, pos, filtered_vel, priv_est, action, targets)

        # 11. Update state
        prev_pos = pos.copy()
        prev_time = now

        # 12. Watchdog
        if loop_dt_ms > watchdog_stop_ms:
            print(f"[deploy] EMERGENCY: Loop took {loop_dt_ms:.1f}ms > {watchdog_stop_ms}ms!")
            if dxl:
                # Hold current position instead of wild movement
                dxl.write_position_targets(pos)
            break
        elif loop_dt_ms > watchdog_warn_ms:
            print(f"[deploy] WARNING: Loop took {loop_dt_ms:.1f}ms (target: {1000/freq:.1f}ms)")

        step += 1

        # Duration limit
        if args.duration and step >= int(args.duration * freq):
            print(f"[deploy] Duration limit reached ({args.duration}s)")
            break

        # Rate control
        rate.sleep()

    # --- Shutdown ---
    print(f"\n[deploy] Shutting down after {step} steps...")

    if dxl:
        # Gently return to home position
        print("[deploy] Moving to home position...")
        dxl.go_to_home()
        time.sleep(1.0)
        dxl.disconnect()

    if logger:
        logger.close()

    print("[deploy] Done.")


if __name__ == "__main__":
    main()
