#!/usr/bin/env python3
"""Serial latency diagnosis: measures read_positions() timing over 200 steps.

Run on the robot (Raspberry Pi):
    python scripts/diagnose_serial_latency.py --config ../config.yaml

Does NOT write any commands. Safe to run with robot powered on.
"""

import argparse
import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hardware.dynamixel_interface import DynamixelInterface
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config.yaml")
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    cfg = load_config(args.config)

    dxl = DynamixelInterface(
        port=cfg["dynamixel"]["port"],
        baudrate=cfg["dynamixel"]["baudrate"],
        servo_ids=[j["servo_id"] for j in cfg["joints"]],
        offsets_raw=[j["offset_raw"] for j in cfg["joints"]],
        lower_rads=[j["lower_rad"] for j in cfg["joints"]],
        upper_rads=[j["upper_rad"] for j in cfg["joints"]],
    )
    dxl.connect()

    # Warm up
    for _ in range(10):
        dxl.read_positions()

    # Check USB latency_timer
    try:
        with open("/sys/bus/usb-serial/devices/ttyUSB0/latency_timer") as f:
            lat = f.read().strip()
        print(f"[diag] USB latency_timer = {lat} ms  (should be 1 for low-latency)")
        if int(lat) > 1:
            print(f"[diag] WARNING: High latency_timer! Run:")
            print(f"       echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer")
    except FileNotFoundError:
        print("[diag] Could not read latency_timer (non-USB or different path)")

    print(f"\n[diag] Measuring read_positions() over {args.steps} steps...\n")

    read_times = []
    for i in range(args.steps):
        t0 = time.monotonic()
        _ = dxl.read_positions()
        dt_ms = (time.monotonic() - t0) * 1000
        read_times.append(dt_ms)
        if i % 50 == 0:
            print(f"  step {i:4d}: read_dt = {dt_ms:.2f} ms")

    read_times = np.array(read_times)

    print(f"\n--- Read latency stats ({args.steps} steps) ---")
    print(f"  Mean:   {read_times.mean():.2f} ms")
    print(f"  Std:    {read_times.std():.2f} ms")
    print(f"  Min:    {read_times.min():.2f} ms")
    print(f"  Max:    {read_times.max():.2f} ms")
    print(f"  P50:    {np.percentile(read_times, 50):.2f} ms")
    print(f"  P95:    {np.percentile(read_times, 95):.2f} ms")
    print(f"  P99:    {np.percentile(read_times, 99):.2f} ms")
    print()

    # Classify
    over_30 = (read_times > 30).sum()
    over_60 = (read_times > 60).sum()
    over_100 = (read_times > 100).sum()
    print(f"  > 30ms (over 30Hz budget): {over_30} / {args.steps}  ({100*over_30/args.steps:.1f}%)")
    print(f"  > 60ms:                    {over_60} / {args.steps}  ({100*over_60/args.steps:.1f}%)")
    print(f"  > 100ms:                   {over_100} / {args.steps}  ({100*over_100/args.steps:.1f}%)")

    if read_times.mean() > 15:
        print("\n[diag] RESULT: read_positions() is the bottleneck.")
        print("  → Serial latency is too high. Set USB latency_timer=1.")
    elif read_times.max() > 60:
        print("\n[diag] RESULT: Occasional spikes in serial read (intermittent).")
        print("  → Check servo connection quality and USB cable.")
    else:
        print("\n[diag] RESULT: Serial read is fast. Check ONNX inference time instead.")
        print("  → Run: python deploy.py --dry-run --log")

    dxl.disconnect()


if __name__ == "__main__":
    main()
