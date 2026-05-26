#!/usr/bin/env python3
"""Test Dynamixel AX-18A servo connectivity and basic motion.

Usage::

    # Ping all servos and read positions
    python scripts/test_servos.py --config config.yaml

    # Sine wave test (small ±5° oscillation on all joints)
    python scripts/test_servos.py --config config.yaml --sine

    # Test a single servo
    python scripts/test_servos.py --config config.yaml --servo-id 1
"""

import argparse
import math
import signal
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, "..")
from hardware.dynamixel_interface import DynamixelInterface


def load_config(path: str) -> dict:
    with open(path, "r") as f:
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


def test_ping_and_read(dxl: DynamixelInterface, cfg: dict):
    """Ping all servos and read current positions."""
    print("\n=== Ping & Read Test ===")
    pos = dxl.read_positions()
    joints = cfg["joints"]

    print(f"{'Joint':<24s}  {'ID':>3s}  {'Position (rad)':>14s}  {'Position (deg)':>14s}")
    print("-" * 62)
    for i, j in enumerate(joints):
        deg = math.degrees(pos[i])
        print(f"{j['name']:<24s}  {j['servo_id']:3d}  {pos[i]:14.4f}  {deg:14.2f}°")


def test_sine_wave(dxl: DynamixelInterface, cfg: dict, amplitude_deg: float = 5.0,
                   frequency: float = 0.5, duration: float = 5.0):
    """Run a small sine wave oscillation on all joints for safety check."""
    print(f"\n=== Sine Wave Test (±{amplitude_deg}°, {frequency}Hz, {duration}s) ===")
    print("Press Ctrl+C to stop.\n")

    amplitude = math.radians(amplitude_deg)
    num_joints = len(cfg["joints"])
    dt = 1.0 / 60.0  # 60 Hz update

    running = True

    def handler(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handler)

    t_start = time.monotonic()

    while running:
        t = time.monotonic() - t_start
        if t > duration:
            break

        offset = amplitude * math.sin(2 * math.pi * frequency * t)
        targets = np.full(num_joints, offset, dtype=np.float32)
        dxl.write_position_targets(targets)

        time.sleep(dt)

    # Return to home
    print("Returning to home position...")
    dxl.go_to_home()
    time.sleep(1.0)
    print("Done.")


def test_single_servo(dxl: DynamixelInterface, cfg: dict, servo_id: int):
    """Test a single servo: read position, then small oscillation."""
    joints = cfg["joints"]
    idx = None
    for i, j in enumerate(joints):
        if j["servo_id"] == servo_id:
            idx = i
            break

    if idx is None:
        print(f"Servo ID {servo_id} not found in config.")
        return

    name = joints[idx]["name"]
    pos = dxl.read_positions()
    deg = math.degrees(pos[idx])
    print(f"\n=== Single Servo Test: {name} (ID={servo_id}) ===")
    print(f"Current position: {pos[idx]:.4f} rad ({deg:.2f}°)")

    print(f"Oscillating ±3° for 3 seconds...")
    amplitude = math.radians(3.0)
    num_joints = len(joints)
    t_start = time.monotonic()

    while time.monotonic() - t_start < 3.0:
        t = time.monotonic() - t_start
        targets = np.zeros(num_joints, dtype=np.float32)
        targets[idx] = amplitude * math.sin(2 * math.pi * 0.5 * t)
        dxl.write_position_targets(targets)
        time.sleep(1.0 / 60.0)

    dxl.go_to_home()
    time.sleep(0.5)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Test Dynamixel servos")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--sine", action="store_true", help="Run sine wave test")
    parser.add_argument("--servo-id", type=int, default=None,
                        help="Test a specific servo by ID")
    parser.add_argument("--amplitude", type=float, default=5.0,
                        help="Sine amplitude in degrees (default: 5)")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Sine test duration in seconds (default: 5)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dxl = build_hardware(cfg)

    try:
        dxl.connect()
        test_ping_and_read(dxl, cfg)

        if args.servo_id is not None:
            test_single_servo(dxl, cfg, args.servo_id)
        elif args.sine:
            test_sine_wave(dxl, cfg, amplitude_deg=args.amplitude,
                           duration=args.duration)
    finally:
        dxl.disconnect()


if __name__ == "__main__":
    main()
