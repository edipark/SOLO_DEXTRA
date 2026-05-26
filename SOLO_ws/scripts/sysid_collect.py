#!/usr/bin/env python3
"""SysID data collection: sequentially step each joint and log response.

Run on Raspberry Pi with robot SUSPENDED (legs free).

For each of 12 joints:
  1. Hold all joints at baseline.
  2. Step target by +step_deg.
  3. Record at SAMPLE_HZ for DURATION_S.
  4. Step target by -step_deg (back through 0 to negative).
  5. Record again.
  6. Return to baseline.

Output:
  logs/sysid/joint{N}_{name}.csv  (per joint)
  logs/sysid/manifest.json        (test parameters)

Usage:
    cd ~/SOLO_ws  # or wherever deploy.py lives
    python scripts/sysid_collect.py --config config.yaml --step_deg 10

Safety:
    - Robot must be SUSPENDED (legs not bearing weight).
    - step_deg should stay small (<= 15°).
    - Keep hand on power switch.
    - Press Ctrl+C to abort cleanly (returns to baseline).
"""

import argparse
import json
import os
import sys
import signal
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hardware.dynamixel_interface import DynamixelInterface

# ---------- Test parameters ---------------------------------------------------
SAMPLE_HZ = 100      # sampling rate during step recording
DURATION_S = 1.5     # record duration after each step
HOLD_BEFORE_S = 0.5  # hold baseline before stepping
HOLD_BETWEEN_S = 0.8 # hold between +step and -step
# ------------------------------------------------------------------------------

# Symmetric L/R joint pairs for --symmetric mode.
# Partner joint moves simultaneously with the primary to keep the robot body
# balanced (reduces asymmetric perturbation during step response measurement).
#
# MIRROR_SIGN is chosen so both legs perform the same PHYSICAL motion:
#   X/Z-axis joints (HipYaw, HipRoll, AnkleRoll): same URDF sign already
#     gives same physical direction → mirror_sign = +1
#   Y-axis joints (Thigh, Calf, AnklePitch): motors are installed mirror-
#     symmetric, so same URDF sign gives OPPOSITE physical direction →
#     use mirror_sign = -1 to achieve physically symmetric motion.
#
# Format: { primary_idx: (partner_idx, mirror_sign) }
SYMMETRIC_PAIRS = {
     0: ( 1, +1),   # L/R HipYaw     (Z-axis, same URDF = same physical)
     1: ( 0, +1),
     2: ( 3, +1),   # L/R HipRoll    (X-axis, same URDF = same physical)
     3: ( 2, +1),
     4: ( 5, -1),   # L/R Thigh      (Y-axis, same URDF = opposite physical)
     5: ( 4, -1),
     6: ( 7, -1),   # L/R Calf       (Y-axis, same URDF = opposite physical)
     7: ( 6, -1),
     8: ( 9, -1),   # L/R AnklePitch (Y-axis, same URDF = opposite physical)
     9: ( 8, -1),
    10: (11, +1),   # L/R AnkleRoll  (X-axis, same URDF = same physical)
    11: (10, +1),
}


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def smooth_move(dxl, current_targets, joint_idx, new_value, duration_s=0.5,
                rate_hz=50):
    """Linearly interpolate joint_idx target from current to new_value over
    duration_s. Used to safely return to baseline."""
    n = max(1, int(duration_s * rate_hz))
    start = current_targets[joint_idx]
    targets = current_targets.copy()
    for i in range(n + 1):
        alpha = i / n
        targets[joint_idx] = start + alpha * (new_value - start)
        dxl.write_position_targets(targets)
        time.sleep(1.0 / rate_hz)
    return targets


def smooth_move_pair(dxl, current_targets, moves, duration_s=0.5, rate_hz=50):
    """Simultaneously interpolate multiple joints from current to target values.

    moves: list of (joint_idx, new_value)
    Both/all joints move together over the same duration, preventing
    the abrupt snap-back that occurs when only the primary is smoothed.
    """
    n = max(1, int(duration_s * rate_hz))
    starts = {idx: current_targets[idx] for idx, _ in moves}
    targets = current_targets.copy()
    for i in range(n + 1):
        alpha = i / n
        for idx, new_val in moves:
            targets[idx] = starts[idx] + alpha * (new_val - starts[idx])
        dxl.write_position_targets(targets)
        time.sleep(1.0 / rate_hz)
    return targets


# ---------- Chirp collection --------------------------------------------------

def collect_chirp_joint(dxl, joint_idx, baseline_targets, writer,
                        amp_rad, f_start_hz, f_end_hz, duration_s, sample_hz,
                        symmetric_pair=None):
    """Log-swept chirp excitation on one joint.

    Instantaneous frequency increases from f_start_hz to f_end_hz on a log
    scale over duration_s seconds.  Amplitude is \u00b1amp_rad around baseline.
    The partner joint (if given) mirrors the chirp for body balance.

    Writes rows: [t_s, cmd_rad, pos_rad]
    Returns: targets array after excitation ends.
    """
    n_samples = int(duration_s * sample_hz)
    dt        = 1.0 / sample_hz
    b_j       = baseline_targets[joint_idx]
    targets   = baseline_targets.copy()

    # Log-chirp: \u03c6(t) = 2\u03c0\u00b7f_start\u00b7T/ln(k) \u00b7 [k^(t/T) \u2212 1]
    #   => instantaneous freq f(t) = f_start \u00b7 k^(t/T)
    k             = f_end_hz / f_start_hz
    phase_coeff   = 2.0 * np.pi * f_start_hz * duration_s / np.log(k)

    t_start = time.monotonic()
    for i in range(n_samples):
        t_loop = time.monotonic()
        t_i    = i * dt

        phase      = phase_coeff * (k ** (t_i / duration_s) - 1.0)
        cmd_offset = amp_rad * np.sin(phase)

        targets[joint_idx] = b_j + cmd_offset
        if symmetric_pair is not None:
            pidx, mirror_sign = symmetric_pair
            targets[pidx] = baseline_targets[pidx] + mirror_sign * cmd_offset

        dxl.write_position_targets(targets)
        pos    = dxl.read_positions()
        t_real = time.monotonic() - t_start

        writer.writerow([f"{t_real:.5f}",
                         f"{targets[joint_idx]:.5f}",
                         f"{pos[joint_idx]:.5f}"])

        elapsed = time.monotonic() - t_loop
        if elapsed < dt:
            time.sleep(dt - elapsed)

    return targets

# ------------------------------------------------------------------------------


def record_step(dxl, joint_idx, baseline_pos, current_targets, step_rad,
                writer, trial_label, partner_override=None):
    """Apply a step on joint_idx and record at SAMPLE_HZ for DURATION_S.

    writer: csv-like, expects rows [direction, t_s, target_rad, pos_rad].
    partner_override: (partner_idx, partner_target_rad) — if set, that joint
        is commanded simultaneously (symmetric anti-collision mode).
    """
    targets = current_targets.copy()
    new_target = baseline_pos + step_rad
    targets[joint_idx] = new_target
    if partner_override is not None:
        pidx, pval = partner_override
        targets[pidx] = pval

    n_samples = int(SAMPLE_HZ * DURATION_S)
    dt = 1.0 / SAMPLE_HZ

    # Send step command
    t_step = time.monotonic()
    dxl.write_position_targets(targets)

    for _ in range(n_samples):
        t_loop = time.monotonic()
        pos = dxl.read_positions()
        t_rel = time.monotonic() - t_step
        writer.writerow([trial_label, f"{t_rel:.5f}",
                         f"{new_target:.5f}",
                         f"{pos[joint_idx]:.5f}"])
        elapsed = time.monotonic() - t_loop
        if elapsed < dt:
            time.sleep(dt - elapsed)

    return targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--step_deg", type=float, default=10.0,
                        help="Step magnitude in degrees (default: 10)")
    parser.add_argument("--joints", type=str, default=None,
                        help="Comma-separated joint indices (e.g. '0,1,2'). "
                             "Default: all 12.")
    parser.add_argument("--out_dir", type=str, default="logs/sysid")
    parser.add_argument("--symmetric", action="store_true",
                        help="Move the L/R partner joint simultaneously "
                             "to prevent leg collision (step and chirp modes). "
                             "Uses SYMMETRIC_PAIRS mapping.")
    parser.add_argument("--mode", choices=["step", "chirp"], default="step",
                        help="Excitation type: step (default) or chirp (log-swept sine)")
    parser.add_argument("--amp_deg", type=float, default=4.0,
                        help="Chirp amplitude \u00b1deg around baseline (chirp only, default: 4)")
    parser.add_argument("--f_start", type=float, default=0.5,
                        help="Chirp start frequency Hz (chirp only, default: 0.5)")
    parser.add_argument("--f_end", type=float, default=25.0,
                        help="Chirp end frequency Hz (chirp only, default: 25.0)")
    parser.add_argument("--chirp_dur", type=float, default=15.0,
                        help="Chirp sweep duration per joint in seconds (default: 15)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    joint_cfgs = cfg["joints"]
    num_joints = len(joint_cfgs)
    step_rad = np.deg2rad(args.step_deg)

    if args.joints:
        joint_indices = [int(x) for x in args.joints.split(",")]
    else:
        joint_indices = list(range(num_joints))

    os.makedirs(args.out_dir, exist_ok=True)

    # ----- Hardware connect -----
    dxl = DynamixelInterface(
        port=cfg["dynamixel"]["port"],
        baudrate=cfg["dynamixel"]["baudrate"],
        servo_ids=[j["servo_id"] for j in joint_cfgs],
        offsets_raw=[j["offset_raw"] for j in joint_cfgs],
        lower_rads=[j["lower_rad"] for j in joint_cfgs],
        upper_rads=[j["upper_rad"] for j in joint_cfgs],
    )
    dxl.connect()

    # Use current pose as baseline target (assumes robot is suspended in default pose)
    baseline = dxl.read_positions().copy()
    targets = baseline.copy()
    dxl.write_position_targets(targets)
    print(f"[sysid] Baseline pose (joint_idx | servo_id | name):")
    for i, j in enumerate(joint_cfgs):
        print(f"   [{i:2d}] servo_id={j['servo_id']:2d}  {j['name']:24s}"
              f"  {baseline[i]:+.4f} rad  ({np.rad2deg(baseline[i]):+.1f}°)")
    print()

    # ----- Signal handler for clean shutdown -----
    aborted = {"flag": False}

    def sigint_handler(sig, frame):
        print("\n[sysid] Interrupt — returning to baseline...")
        aborted["flag"] = True

    signal.signal(signal.SIGINT, sigint_handler)

    # ----- Manifest -----
    manifest = {
        "mode": args.mode,
        "step_deg": args.step_deg,
        "step_rad": step_rad,
        "sample_hz": SAMPLE_HZ,
        "duration_s": DURATION_S if args.mode == "step" else args.chirp_dur,
        "hold_before_s": HOLD_BEFORE_S,
        "hold_between_s": HOLD_BETWEEN_S,
        "joint_names": [j["name"] for j in joint_cfgs],
        "baseline_rad": baseline.tolist(),
        "joints_tested": joint_indices,
        "frequency_hz_config": cfg["control"]["frequency_hz"],
    }
    if args.mode == "chirp":
        manifest.update({
            "chirp_amp_deg":    args.amp_deg,
            "chirp_f_start_hz": args.f_start,
            "chirp_f_end_hz":   args.f_end,
        })
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    if args.mode == "step":
        print(f"[sysid] Step: \u00b1{args.step_deg}\u00b0 ({step_rad:+.4f} rad)")
        print(f"[sysid] Sampling: {SAMPLE_HZ} Hz, duration: {DURATION_S}s per step")
    else:
        print(f"[sysid] Chirp: \u00b1{args.amp_deg}\u00b0 amplitude, "
              f"{args.f_start}\u2013{args.f_end} Hz log-sweep, "
              f"{args.chirp_dur}s per joint")
        print(f"[sysid] Sampling: {SAMPLE_HZ} Hz")
    print(f"[sysid] Will test {len(joint_indices)} joint(s): {joint_indices}\n")
    input("[sysid] Robot SUSPENDED? Press ENTER to start (Ctrl+C to abort)...")

    # ----- Main test loop -----
    import csv
    for jidx in joint_indices:
        if aborted["flag"]:
            break

        jname = joint_cfgs[jidx]["name"]
        lower = joint_cfgs[jidx]["lower_rad"]
        upper = joint_cfgs[jidx]["upper_rad"]
        b = baseline[jidx]

        # Skip if excursion out of range
        excursion = step_rad if args.mode == "step" else np.deg2rad(args.amp_deg)
        if (b + excursion) > upper or (b - excursion) < lower:
            print(f"[sysid] Joint {jidx} ({jname}): excursion out of range "
                  f"[{lower:.3f}, {upper:.3f}] — SKIP")
            continue

        servo_id = joint_cfgs[jidx]["servo_id"]
        out_csv = os.path.join(args.out_dir, f"joint{jidx:02d}_{jname}.csv")
        print(f"\n{'='*60}")
        print(f"  Joint idx : {jidx}")
        print(f"  Name      : {jname}")
        print(f"  Servo ID  : {servo_id}  ← SENDING COMMANDS TO THIS MOTOR")
        print(f"  Range     : [{lower:.3f}, {upper:.3f}] rad  "
              f"([{np.rad2deg(lower):.1f}°, {np.rad2deg(upper):.1f}°])")
        print(f"  Baseline  : {b:+.4f} rad  ({np.rad2deg(b):+.1f}°)")
        print(f"  Target +  : {b+step_rad:+.4f} rad  ({np.rad2deg(b+step_rad):+.1f}°)")
        print(f"  Target -  : {b-step_rad:+.4f} rad  ({np.rad2deg(b-step_rad):+.1f}°)")
        print(f"  CSV       : {out_csv}")
        print(f"{'='*60}")

        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)

            # Hold baseline briefly
            targets = baseline.copy()
            dxl.write_position_targets(targets)
            time.sleep(HOLD_BEFORE_S)

            # Symmetric partner setup (same for both modes)
            sym_pair = None
            if args.symmetric and jidx in SYMMETRIC_PAIRS:
                pidx, mirror_sign = SYMMETRIC_PAIRS[jidx]
                sym_pair = (pidx, mirror_sign)
                dir_str = "same" if mirror_sign == +1 else "opposite"
                print(f"  [symmetric] servo {joint_cfgs[pidx]['servo_id']} "
                      f"({joint_cfgs[pidx]['name']}) → {dir_str} URDF sign "
                      f"(physically symmetric)")

            if args.mode == "chirp":
                # ---- Chirp mode -------------------------------------------
                writer.writerow(["t_s", "cmd_rad", "pos_rad"])
                amp_rad = np.deg2rad(args.amp_deg)
                print(f"  [chirp] servo {servo_id} ±{args.amp_deg}° "
                      f"{args.f_start}–{args.f_end} Hz  {args.chirp_dur}s  recording...")
                targets = collect_chirp_joint(
                    dxl, jidx, targets, writer,
                    amp_rad=amp_rad,
                    f_start_hz=args.f_start,
                    f_end_hz=args.f_end,
                    duration_s=args.chirp_dur,
                    sample_hz=SAMPLE_HZ,
                    symmetric_pair=sym_pair)
                print(f"  [chirp] done.")
                # Return to baseline smoothly
                return_moves = [(jidx, baseline[jidx])]
                if sym_pair is not None:
                    return_moves.append((sym_pair[0], baseline[sym_pair[0]]))
                targets = smooth_move_pair(dxl, targets, return_moves, duration_s=0.8)

            else:
                # ---- Step mode --------------------------------------------
                writer.writerow(["direction", "t_s", "target_rad", "pos_rad"])
                p_pos_override = None
                p_neg_override = None
                if sym_pair is not None:
                    pidx, mirror_sign = sym_pair
                    p_base = baseline[pidx]
                    p_pos_override = (pidx, p_base + mirror_sign * step_rad)
                    p_neg_override = (pidx, p_base - mirror_sign * step_rad)

                # +step
                print(f"  [+step] servo {servo_id} → target {b+step_rad:+.4f} rad "
                      f"({np.rad2deg(b+step_rad):+.1f}°)  recording...")
                targets = record_step(dxl, jidx, b, targets, +step_rad, writer, "pos",
                                      partner_override=p_pos_override)
                print(f"  [+step] done.")

                # Hold then return — primary AND partner return together
                time.sleep(HOLD_BETWEEN_S)
                return_moves = [(jidx, b)]
                if p_pos_override is not None:
                    return_moves.append((p_pos_override[0], baseline[p_pos_override[0]]))
                targets = smooth_move_pair(dxl, targets, return_moves, duration_s=0.4)
                time.sleep(HOLD_BETWEEN_S)

                if aborted["flag"]:
                    break

                # -step
                print(f"  [-step] servo {servo_id} → target {b-step_rad:+.4f} rad "
                      f"({np.rad2deg(b-step_rad):+.1f}°)  recording...")
                targets = record_step(dxl, jidx, b, targets, -step_rad, writer, "neg",
                                      partner_override=p_neg_override)
                print(f"  [-step] done.")

                # Return to baseline
                time.sleep(HOLD_BETWEEN_S)
                return_moves = [(jidx, b)]
                if p_neg_override is not None:
                    return_moves.append((p_neg_override[0], baseline[p_neg_override[0]]))
                targets = smooth_move_pair(dxl, targets, return_moves, duration_s=0.4)

        print(f"  saved → {out_csv}")

    # ----- Final: return all to baseline, disable -----
    print("\n[sysid] Returning to baseline pose...")
    dxl.write_position_targets(baseline)
    time.sleep(0.5)
    dxl.disconnect()
    print(f"[sysid] Done. Logs in {args.out_dir}/")
    print(f"[sysid] Next: copy {args.out_dir}/ to dev machine and run "
          f"sysid_analyze.py")


if __name__ == "__main__":
    main()
