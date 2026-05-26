#!/usr/bin/env python3
"""SysID analysis: compare real joint response with ImplicitActuator simulation.

Reads CSVs from sysid_collect.py, fits each joint's response to a 2nd-order
PD model (J*ddq = k*(q_target - q) - d*dq), and recommends per-joint
stiffness/damping for IsaacLab ImplicitActuatorCfg.

Usage:
    python scripts/sysid_analyze.py --in_dir logs/sysid --out_dir logs/sysid/plots

Outputs:
  - logs/sysid/plots/joint{N}_{name}.png  (target vs real vs sim, per joint)
  - logs/sysid/plots/summary.png          (12-subplot grid)
  - logs/sysid/recommendations.txt        (k, d, ωn, ζ, bandwidth per joint)

Notes on model:
  ImplicitActuator effectively solves at the joint level:
      J*ddq + d*dq + k*q = k*q_target
  Equivalent 2nd-order:
      ωn² = k/J,  2ζωn = d/J  →  k = J*ωn²,  d = 2ζωn*J

  J (joint inertia) is fixed by the URDF in sim, so fitting (k, d) is
  equivalent to fitting (ωn, ζ). Below we report ωn, ζ, then convert to
  recommended k, d for an inertia range typical of AX-18A links.
"""

import argparse
import csv
import json
import os
import sys
from glob import glob
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import csd, welch, coherence as _scipy_coherence


# ------- AX-18A link inertia estimates (URDF-dependent, rough) ----------------
# Better: extract from your URDF/usd. These are placeholders.
DEFAULT_J_KGM2 = 0.002

# AX-18A hardware compliance stiffness (compliance_slope=32):
# k = stall_torque * (180/pi) / (255 * slope) * 1024 ≈ 11.1 N·m/rad
# Used as sanity-check anchor; individual k_rec should be near this value.
K_AX18A_HW = 11.1   # N·m/rad

# Bilateral symmetry pairs: primary_idx → (partner_idx, mirror_sign)
# mirror_sign is for motion direction; stiffness/damping are always the same.
# Must stay in sync with SYMMETRIC_PAIRS in sysid_collect.py.
SYMMETRIC_PAIRS = {
     0: ( 1, +1),   # L/R HipYaw
     1: ( 0, +1),
     2: ( 3, +1),   # L/R HipRoll
     3: ( 2, +1),
     4: ( 5, -1),   # L/R Thigh
     5: ( 4, -1),
     6: ( 7, -1),   # L/R Calf
     7: ( 6, -1),
     8: ( 9, -1),   # L/R AnklePitch
     9: ( 8, -1),
    10: (11, +1),   # L/R AnkleRoll
    11: (10, +1),
}
J_RANGE_KGM2 = [0.0005, 0.001, 0.002, 0.003, 0.005]
# ------------------------------------------------------------------------------

# Default URDF path (relative to SOLO_ws/)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_URDF = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "..",
                 "source", "isaaclab_tasks", "isaaclab_tasks",
                 "direct", "SOLO_DEXTRA", "assets",
                 "Dextra_lowerbody.urdf"))


def extract_urdf_inertias(urdf_path: str, joint_names: list[str]) -> dict[str, float]:
    """CRB (Composite Rigid Body) inertia at zero configuration.

    For each joint, sums inertia contributions from ALL downstream links
    (full subtree), not just the immediate child link.  Uses the parallel-axis
    theorem and the full 3×3 inertia tensor projection onto the joint axis.

    Assumes no RPY rotation in joint origins (all URDF frames are axis-aligned
    at zero config), which is true for Dextra_lowerbody.urdf.

    Returns dict {joint_name: J_kgm2 (effective)}.
    """
    if not os.path.exists(urdf_path):
        print(f"[urdf] File not found: {urdf_path}  — using default J={DEFAULT_J_KGM2}")
        return {}

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # --- Parse link inertia tensors (including off-diagonal) ---
    link_inertia: dict[str, dict] = {}
    for link in root.findall("link"):
        lname = link.get("name")
        inertial = link.find("inertial")
        zero = {"ixx": 0.0, "iyy": 0.0, "izz": 0.0,
                "ixy": 0.0, "ixz": 0.0, "iyz": 0.0,
                "mass": 0.0, "com": [0.0, 0.0, 0.0]}
        if inertial is None:
            link_inertia[lname] = zero
            continue
        I_el = inertial.find("inertia")
        m_el = inertial.find("mass")
        com_el = inertial.find("origin")
        com_xyz = ([float(v) for v in com_el.get("xyz", "0 0 0").split()]
                   if com_el is not None else [0.0, 0.0, 0.0])
        link_inertia[lname] = {
            "ixx": float(I_el.get("ixx", 0)) if I_el is not None else 0.0,
            "iyy": float(I_el.get("iyy", 0)) if I_el is not None else 0.0,
            "izz": float(I_el.get("izz", 0)) if I_el is not None else 0.0,
            "ixy": float(I_el.get("ixy", 0)) if I_el is not None else 0.0,
            "ixz": float(I_el.get("ixz", 0)) if I_el is not None else 0.0,
            "iyz": float(I_el.get("iyz", 0)) if I_el is not None else 0.0,
            "mass": float(m_el.get("value", 0)) if m_el is not None else 0.0,
            "com": com_xyz,
        }

    # --- Build kinematic tree ---
    # joint_data[jname] = {child, parent, axis, origin}
    # children_of_link[link] = [list of joint names whose parent is that link]
    joint_data: dict[str, dict] = {}
    children_of_link: dict[str, list] = {}
    for joint in root.findall("joint"):
        jname = joint.get("name")
        jtype = joint.get("type", "fixed")
        parent_el = joint.find("parent")
        child_el = joint.find("child")
        axis_el = joint.find("axis")
        origin_el = joint.find("origin")
        if parent_el is None or child_el is None:
            continue
        parent_link = parent_el.get("link")
        child_link = child_el.get("link")
        axis = ([float(x) for x in axis_el.get("xyz", "0 0 1").split()]
                if axis_el is not None else [0.0, 0.0, 1.0])
        origin_xyz = ([float(v) for v in origin_el.get("xyz", "0 0 0").split()]
                      if origin_el is not None else [0.0, 0.0, 0.0])
        joint_data[jname] = {
            "child": child_link, "parent": parent_link,
            "axis": axis, "origin": origin_xyz, "type": jtype,
        }
        children_of_link.setdefault(parent_link, []).append(jname)

    # --- Compute link frame origin positions in base frame at zero config ---
    # At zero config with no RPY offsets, this is just a chain of translations.
    link_frame_pos: dict[str, list] = {}

    def _build_frame_pos(link_name: str, pos: list) -> None:
        link_frame_pos[link_name] = pos[:]
        for jn in children_of_link.get(link_name, []):
            jo = joint_data[jn]["origin"]
            child_pos = [pos[i] + jo[i] for i in range(3)]
            _build_frame_pos(joint_data[jn]["child"], child_pos)

    _build_frame_pos("base_link", [0.0, 0.0, 0.0])

    # --- BFS helper: all descendant links of a joint ---
    def _descendants(jname: str) -> list:
        result = []
        queue = [joint_data[jname]["child"]]
        while queue:
            ln = queue.pop(0)
            result.append(ln)
            for nj in children_of_link.get(ln, []):
                queue.append(joint_data[nj]["child"])
        return result

    # --- CRB: sum over all downstream links ---
    joint_J: dict[str, float] = {}
    for jname in joint_names:
        if jname not in joint_data:
            continue
        ax, ay, az = joint_data[jname]["axis"]
        # Joint frame origin in base (= child link frame origin at zero config)
        j_pos = link_frame_pos.get(joint_data[jname]["child"], [0.0, 0.0, 0.0])

        J_total = 0.0
        for ln in _descendants(jname):
            I = link_inertia.get(ln)
            if I is None or I["mass"] == 0.0:
                continue
            m = I["mass"]
            # CoM in base frame = link frame origin + CoM offset in link frame
            lp = link_frame_pos.get(ln, [0.0, 0.0, 0.0])
            com_b = [lp[i] + I["com"][i] for i in range(3)]
            # Vector from joint origin to CoM
            rx = com_b[0] - j_pos[0]
            ry = com_b[1] - j_pos[1]
            rz = com_b[2] - j_pos[2]
            # Inertia of link about its CoM along joint axis (full tensor)
            I_axis = (ax**2 * I["ixx"] + ay**2 * I["iyy"] + az**2 * I["izz"]
                      + 2*ax*ay * I["ixy"] + 2*ax*az * I["ixz"] + 2*ay*az * I["iyz"])
            # Parallel-axis (Steiner): m * |r_perp|^2 = m * (|r|^2 - (r·a)^2)
            r_dot_a = ax*rx + ay*ry + az*rz
            d_perp_sq = max(0.0, rx**2 + ry**2 + rz**2 - r_dot_a**2)
            J_total += I_axis + m * d_perp_sq

        joint_J[jname] = J_total

    return joint_J


def simulate_2nd_order(t, q_target, wn, zeta, q0=0.0, dq0=0.0):
    """Analytical 2nd-order step response — numerically stable for any (wn, zeta).

    Computes the exact closed-form response of:
        J*ddq + d*dq + k*(q - q_target) = 0
    with ωn²=k/J, 2ζωn=d/J, for constant q_target (dq0=0 assumed).

    Replaces the previous RK4 integrator which became unstable for
    large (ζ*ωn*dt) values (e.g. AnkleRoll with DEFAULT_J caused 1e241 overflow).
    """
    t = np.asarray(t, dtype=float)
    target = (float(np.asarray(q_target).flat[-1])
              if np.ndim(q_target) > 0 else float(q_target))
    step = target - q0
    if abs(step) < 1e-14 or wn <= 0 or zeta <= 0:
        return np.full_like(t, q0, dtype=float)

    # Unit step response of H(s) = ωn² / (s² + 2ζωn·s + ωn²)
    if zeta < 1.0:                  # underdamped
        wd = wn * np.sqrt(1.0 - zeta**2)
        y = 1.0 - np.exp(-zeta * wn * t) * (
            np.cos(wd * t) + (zeta / np.sqrt(1.0 - zeta**2)) * np.sin(wd * t))
    elif abs(zeta - 1.0) < 1e-9:   # critically damped
        y = 1.0 - (1.0 + wn * t) * np.exp(-wn * t)
    else:                           # overdamped (zeta > 1)
        sq = np.sqrt(zeta**2 - 1.0)
        s1 = wn * (-zeta + sq)
        s2 = wn * (-zeta - sq)
        c1 = -s2 / (s1 - s2)
        c2 =  s1 / (s1 - s2)
        y = 1.0 - c1 * np.exp(s1 * t) - c2 * np.exp(s2 * t)

    return q0 + step * y


def fit_wn_zeta(t, q_real, q_target_scalar, q0):
    """Find (ωn, ζ) that minimize MSE between simulation and real response."""
    target_arr = np.full_like(t, q_target_scalar)

    def cost(params):
        wn, zeta = params
        if wn <= 0 or zeta <= 0:
            return 1e6
        q_sim = simulate_2nd_order(t, target_arr, wn, zeta, q0=q0, dq0=0.0)
        return np.mean((q_sim - q_real) ** 2)

    # Grid initial seed
    best = (None, 1e9)
    for wn0 in [5, 10, 20, 30, 50, 80]:
        for z0 in [0.3, 0.5, 0.7, 0.9, 1.2]:
            c = cost([wn0, z0])
            if c < best[1]:
                best = ([wn0, z0], c)

    res = minimize(cost, x0=best[0], method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-7, "maxiter": 500})
    wn_fit, zeta_fit = res.x
    return wn_fit, zeta_fit, res.fun


def measure_step_metrics(t, q, target, q0):
    """Rise time (10→90%), overshoot, settling time."""
    step = target - q0
    if abs(step) < 1e-6:
        return {"rise_time_s": np.nan, "overshoot_pct": 0.0,
                "settle_time_s": np.nan}

    progress = (q - q0) / step  # normalized 0→1
    try:
        i10 = np.argmax(progress >= 0.1)
        i90 = np.argmax(progress >= 0.9)
        rise = t[i90] - t[i10] if i90 > i10 else np.nan
    except Exception:
        rise = np.nan

    # overshoot
    if step > 0:
        overshoot = max(0.0, (q.max() - target) / step) * 100
    else:
        overshoot = max(0.0, (target - q.min()) / abs(step)) * 100

    # settling: last index where |q - target| > 2% of |step|
    band = 0.02 * abs(step)
    out_of_band = np.where(np.abs(q - target) > band)[0]
    settle = t[out_of_band[-1]] if len(out_of_band) > 0 else 0.0

    return {"rise_time_s": rise, "overshoot_pct": overshoot,
            "settle_time_s": settle}


def bandwidth_from_wn_zeta(wn, zeta):
    """-3dB bandwidth (Hz) for 2nd-order system."""
    if zeta >= 1.0:
        # critically/over-damped: wbw = wn * sqrt(sqrt(1+...)-1) ... complicated
        # use approx: wbw ≈ wn for ζ=1
        return wn / (2 * np.pi) * 0.64
    term = np.sqrt(4*zeta**4 - 4*zeta**2 + 2)
    wbw = wn * np.sqrt(1 - 2*zeta**2 + term)
    return wbw / (2 * np.pi)


def load_csv(path):
    """Load CSV with rows [direction, t_s, target_rad, pos_rad]."""
    data = {"pos": {"t": [], "target": [], "q": []},
            "neg": {"t": [], "target": [], "q": []}}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = row["direction"]
            data[d]["t"].append(float(row["t_s"]))
            data[d]["target"].append(float(row["target_rad"]))
            data[d]["q"].append(float(row["pos_rad"]))
    for d in data:
        for k in data[d]:
            data[d][k] = np.array(data[d][k])
    return data


def analyze_joint(jidx, jname, csv_path, out_dir, k_current=4.5, d_current=0.45,
                  J_kgm2=DEFAULT_J_KGM2):
    """Fit one joint's data and produce per-direction plot + metrics."""
    data = load_csv(csv_path)
    results = {}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(f"Joint {jidx}: {jname}", fontsize=13, fontweight="bold")

    for k_dir, (direction, ax) in enumerate(zip(["pos", "neg"], axes)):
        d = data[direction]
        if len(d["t"]) < 5:
            ax.text(0.5, 0.5, "(no data)", ha="center", transform=ax.transAxes)
            continue

        t = d["t"]
        q = d["q"]
        target = d["target"][-1]  # constant after step
        q0 = q[0]

        # Fit
        wn, zeta, mse = fit_wn_zeta(t, q, target, q0)
        bw = bandwidth_from_wn_zeta(wn, zeta)
        m = measure_step_metrics(t, q, target, q0)

        # Simulated trajectory at fitted (ωn, ζ)
        q_sim_fit = simulate_2nd_order(t, target, wn, zeta, q0=q0)

        # Simulated trajectory at CURRENT sim params for this joint's actual J
        wn_curr = np.sqrt(k_current / J_kgm2)
        zeta_curr = d_current / (2 * np.sqrt(k_current * J_kgm2))
        q_sim_curr = simulate_2nd_order(t, target, wn_curr, zeta_curr, q0=q0)

        # Plot
        ax.plot(t, np.full_like(t, target), "k--", lw=1, label="target")
        ax.plot(t, q, "b-", lw=1.5, label="real")
        ax.plot(t, q_sim_fit, "r-", lw=1.2, alpha=0.85,
                label=f"fit (ωn={wn:.1f}, ζ={zeta:.2f})")
        ax.plot(t, q_sim_curr, "g:", lw=1.2,
                label=f"current sim (k={k_current}, d={d_current})")
        ax.axhline(q0, color="gray", lw=0.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position (rad)")
        ax.set_title(f"{direction} step  (rise={m['rise_time_s']*1000:.0f}ms, "
                     f"OS={m['overshoot_pct']:.1f}%, BW={bw:.1f}Hz)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)

        results[direction] = {
            "wn_rad_s": wn, "zeta": zeta, "bandwidth_hz": bw, "mse": mse,
            **m
        }

    plt.tight_layout()
    out_png = os.path.join(out_dir, f"joint{jidx:02d}_{jname}.png")
    plt.savefig(out_png, dpi=110)
    plt.close(fig)

    # Average ± over directions
    avg = {}
    for k in ["wn_rad_s", "zeta", "bandwidth_hz", "rise_time_s",
              "overshoot_pct", "settle_time_s"]:
        vals = [results[d][k] for d in results if not np.isnan(results[d][k])]
        avg[k] = np.mean(vals) if vals else np.nan

    return avg, results, out_png, data


def make_all_joints_figure(all_joint_results, out_path, k_current, d_current):
    """Single figure: 4 rows × 3 cols = 12 subplots, one per joint.
    Each subplot overlays +step (blue) and −step (red), both normalized
    to [0→1] displacement.
    real=solid, 2nd-order fit=dashed, current sim params=green dotted.
    """
    cols, rows = 3, 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 14))
    axes_flat = axes.flatten()

    for idx, (jidx, jname, csv_data, avg, J_joint) in enumerate(all_joint_results):
        # Per-joint current-sim response (uses actual URDF J, not a global default)
        wn_c = np.sqrt(k_current / J_joint) if J_joint > 0 else 74.5
        z_c = (d_current / (2 * np.sqrt(k_current * J_joint))
               if J_joint > 0 else 3.68)
        ax = axes_flat[idx]
        wn = avg["wn_rad_s"]
        zeta = avg["zeta"]
        bw = avg["bandwidth_hz"]
        rise_ms = avg["rise_time_s"] * 1000 if not np.isnan(avg["rise_time_s"]) else float("nan")
        os_pct = avg["overshoot_pct"]

        sim_cur_drawn = False
        any_data = False
        for direction, color in [("pos", "#2166ac"), ("neg", "#d6604d")]:
            d = csv_data[direction]
            if len(d["t"]) < 5:
                continue
            t = np.array(d["t"])
            q = np.array(d["q"])
            tgt = float(d["target"][-1])
            q0 = float(q[0])
            step = tgt - q0
            if abs(step) < 1e-6:
                continue
            any_data = True

            # normalize: q0→0, target→1
            q_norm = (q - q0) / abs(step)
            q_fit = simulate_2nd_order(t, tgt, wn, zeta, q0=q0)
            q_fit_norm = (q_fit - q0) / abs(step)
            q_curr = simulate_2nd_order(t, tgt, wn_c, z_c, q0=q0)
            q_curr_norm = (q_curr - q0) / abs(step)

            sign_str = "+" if step > 0 else "−"
            ax.plot(t, q_norm, color=color, lw=1.5, label=f"real {sign_str}")
            ax.plot(t, q_fit_norm, color=color, lw=1.0, ls="--", alpha=0.75,
                    label=f"fit {sign_str}")
            if not sim_cur_drawn:
                ax.plot(t, q_curr_norm, color="green", lw=1.0, ls=":",
                        alpha=0.6, label="sim_cur")
                sim_cur_drawn = True

        ax.axhline(1.0, color="black", lw=0.8, ls="-.", alpha=0.3)
        ax.axhline(0.0, color="black", lw=0.4)
        ax.set_title(
            f"J{jidx}  {jname}\n"
            f"ωn={wn:.1f} r/s  ζ={zeta:.2f}  BW={bw:.1f} Hz\n"
            f"rise={rise_ms:.0f} ms   OS={os_pct:.1f}%",
            fontsize=7.5,
        )
        ax.set_ylim(-0.3, 1.6)
        ax.set_xlabel("t (s)", fontsize=7)
        ax.set_ylabel("(q−q₀)/|Δq|", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
        if idx == 0 and any_data:
            ax.legend(fontsize=6.5, loc="lower right")

    for j in range(len(all_joint_results), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(
        "SysID — All 12 Joints  (normalized step response)\n"
        "blue = real +step    red = real −step    dashed = 2nd-order fit    "
        "green·dot = current sim",
        fontsize=9, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[analyze] All-joints figure → {out_path}")


def analyze_joint_chirp(jidx, jname, csv_path, out_dir,
                        k_current=11.1, d_current=1.1, J_kgm2=DEFAULT_J_KGM2):
    """Frequency-domain sysID from chirp (log-swept sine) excitation.

    Loads [t_s, cmd_rad, pos_rad] CSV, estimates H(jω) via Welch CSD, then
    fits a 2nd-order model using only the coherent low-frequency band where
    the servo actually follows the command.

    Body/frame resonances contaminate high-frequency bins; coherence gating
    and a physical damping prior (AX-18A is overdamped) prevent the optimizer
    from locking onto structural modes.

    Returns (avg_metrics_dict, csv_data_dict).
    """
    # ---- 1. Load data -------------------------------------------------------
    t_list, cmd_list, pos_list = [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            t_list.append(float(row["t_s"]))
            cmd_list.append(float(row["cmd_rad"]))
            pos_list.append(float(row["pos_rad"]))

    t   = np.array(t_list)
    cmd = np.array(cmd_list)
    pos = np.array(pos_list)
    fs  = 1.0 / np.mean(np.diff(t))

    # ---- 2. Spectral estimates ----------------------------------------------
    cmd_ac = cmd - np.mean(cmd)
    pos_ac = pos - np.mean(pos)

    nperseg = min(512, len(t) // 4)
    f_hz, Pxx = welch(cmd_ac,        fs=fs, nperseg=nperseg)
    _,    Pxy = csd(cmd_ac, pos_ac,  fs=fs, nperseg=nperseg)
    _,    Coh = _scipy_coherence(cmd_ac, pos_ac, fs=fs, nperseg=nperseg)
    H = Pxy / (Pxx + 1e-30)

    # ---- 3. Coherence-gated fitting mask ------------------------------------
    # AX-18A is overdamped; useful servo signal is below ~8 Hz.
    # Body/frame modes contaminate above that; coherence drops where body
    # motion is uncorrelated with the command, but may stay HIGH at the body
    # resonance (command → leg → body → spurious encoder delta).
    # Double-gate: coherence threshold + hard frequency ceiling.
    COH_THRESHOLD = 0.50
    FIT_F_MAX_HZ  = 8.0    # Hz — conservative ceiling below body resonances

    mask_full = (f_hz >= 0.3) & (f_hz <= FIT_F_MAX_HZ)
    mask      = mask_full & (Coh >= COH_THRESHOLD)
    n_coh     = int(np.sum(mask))

    if n_coh < 5:
        # Not enough coherent bins: use full low-freq band with a warning
        print(f"[chirp] J{jidx} {jname}: only {n_coh} coherent bins below "
              f"{FIT_F_MAX_HZ} Hz — body motion likely severe. "
              f"Fitting full 0.3–{FIT_F_MAX_HZ} Hz band (result may be unreliable).")
        mask = mask_full
        n_coh = int(np.sum(mask))

    H_fit = H[mask]
    f_fit = f_hz[mask]
    coh_max_hz = float(f_fit[-1]) if len(f_fit) else 0.0

    # ---- 4. 2nd-order model -------------------------------------------------
    def H_model(f, wn, zeta):
        w = 2.0 * np.pi * f
        return wn**2 / (wn**2 - w**2 + 2j * zeta * wn * w)

    def cost(params):
        wn_, zeta_ = abs(params[0]), abs(params[1])
        if wn_ < 2.0 or zeta_ < 0.1:
            return 1e9
        diff = H_fit - H_model(f_fit, wn_, zeta_)
        err  = float(np.mean(np.abs(diff)**2))
        # Physical prior: AX-18A compliance ⇒ ζ ≫ 1 (overdamped)
        # Penalise underdamped solutions to avoid latching onto body modes
        underdamped_penalty = max(0.0, 1.0 - zeta_)**2 * 3.0
        return err + underdamped_penalty

    # Multi-start: seeds span the physically plausible (ωn, ζ) space
    seeds = [(100.0, 3.0), (50.0, 5.0), (150.0, 2.0),
             (200.0, 8.0), (30.0, 10.0), (80.0, 4.0)]
    best_res, best_cost = None, np.inf
    for wn0, z0 in seeds:
        r = minimize(cost, [wn0, z0], method="Nelder-Mead",
                     options={"xatol": 0.3, "fatol": 1e-12, "maxiter": 8000})
        if r.fun < best_cost:
            best_cost = r.fun
            best_res  = r

    wn   = abs(best_res.x[0])
    zeta = abs(best_res.x[1])
    bw   = bandwidth_from_wn_zeta(wn, zeta)

    fn_fit = wn / (2.0 * np.pi)
    fit_extrapolated = fn_fit > coh_max_hz  # ωn is outside the coherent band
    if fit_extrapolated:
        print(f"[chirp] J{jidx} {jname}: ωn={wn:.1f} r/s ({fn_fit:.1f} Hz) is ABOVE "
              f"coherent band ({coh_max_hz:.1f} Hz) — extrapolated, less reliable.")

    # ---- 5. Plot: 2×2 grid -------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Joint {jidx}: {jname}  [Chirp SysID]", fontsize=12,
                 fontweight="bold")

    # --- time domain ---------------------------------------------------------
    ax = axes[0, 0]
    ax.plot(t, cmd, "k--", lw=0.7, alpha=0.7, label="cmd")
    ax.plot(t, pos, "b-",  lw=0.8, label="pos")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Position (rad)")
    ax.set_title("Time domain")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # --- coherence -----------------------------------------------------------
    ax = axes[0, 1]
    ax.semilogx(f_hz[1:], Coh[1:], "b-", lw=1.0)
    ax.axhline(COH_THRESHOLD, color="r", ls="--", lw=1.0,
               label=f"threshold = {COH_THRESHOLD}")
    ax.axvline(FIT_F_MAX_HZ, color="gray", ls=":", lw=1.0,
               label=f"fit ceiling = {FIT_F_MAX_HZ} Hz")
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("γ²")
    ax.set_title("Coherence  (blue ≥ threshold = trusted bins)")
    ax.set_ylim([0, 1.1]); ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    ax.set_xlim([0.2, min(fs / 2.0, 60.0)])

    f_lo    = 0.3
    f_hi    = min(fs / 2.0, 60.0)
    f_dense = np.logspace(np.log10(f_lo), np.log10(f_hi), 400)

    # --- Bode magnitude ------------------------------------------------------
    ax = axes[1, 0]
    H_db = 20.0 * np.log10(np.abs(H) + 1e-30)
    ax.semilogx(f_hz[1:], H_db[1:], "b-", lw=0.8, alpha=0.6, label="measured")
    # Shade trusted fit band
    f_band = f_hz[mask]
    H_band = H_db[mask]
    if len(f_band) > 1:
        ax.fill_between(f_band, H_band - 50, H_band + 50,
                        alpha=0.08, color="green", label="fit band")
    H_m = H_model(f_dense, wn, zeta)
    extrap_label = " (extrapolated)" if fit_extrapolated else ""
    ax.semilogx(f_dense, 20*np.log10(np.abs(H_m)), "r-", lw=1.5, alpha=0.9,
                label=f"fit ωn={wn:.0f}r/s ζ={zeta:.2f}{extrap_label}")
    if J_kgm2 > 0:
        wn_c = np.sqrt(k_current / J_kgm2)
        z_c  = d_current / (2.0 * np.sqrt(k_current * J_kgm2))
        H_c  = H_model(f_dense, wn_c, z_c)
        ax.semilogx(f_dense, 20*np.log10(np.abs(H_c)), "g:", lw=1.2, alpha=0.8,
                    label=f"cur sim (k={k_current}, d={d_current})")
    ax.axvline(FIT_F_MAX_HZ, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("|H| (dB)")
    ax.set_title(f"Bode magnitude — fit in green band\n"
                 f"ωn={wn:.1f} r/s  ζ={zeta:.2f}  BW={bw:.1f} Hz"
                 f"{'  ⚠ EXTRAPOLATED' if fit_extrapolated else ''}")
    ax.legend(fontsize=7.5); ax.grid(alpha=0.3, which="both")
    ax.set_xlim([f_lo, f_hi]); ax.set_ylim([-60, 80])

    # --- Bode phase ----------------------------------------------------------
    ax = axes[1, 1]
    H_ph = np.angle(H, deg=True)
    ax.semilogx(f_hz[1:], H_ph[1:], "b-", lw=0.8, alpha=0.6, label="measured")
    ax.semilogx(f_dense, np.angle(H_m, deg=True), "r-", lw=1.5, alpha=0.9,
                label="fit")
    ax.axvline(FIT_F_MAX_HZ, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Phase (°)")
    ax.set_title("Bode phase")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    ax.set_xlim([f_lo, f_hi]); ax.set_ylim([-200, 20])

    plt.tight_layout()
    out_png = os.path.join(out_dir, f"joint{jidx:02d}_{jname}.png")
    plt.savefig(out_png, dpi=110)
    plt.close(fig)

    avg = {
        "wn_rad_s": wn, "zeta": zeta, "bandwidth_hz": bw,
        "rise_time_s": float("nan"), "overshoot_pct": float("nan"),
        "settle_time_s": float("nan"),
        "_freq_hz":    f_hz.tolist(),
        "_H_mag":      np.abs(H).tolist(),
        "_coherence":  Coh.tolist(),
        "_coh_bw_hz":  coh_max_hz,
        "_extrapolated": bool(fit_extrapolated),
    }
    csv_data = {
        "pos": {"t": t.tolist(), "q": pos.tolist(), "target": cmd.tolist()},
        "neg": {"t": [],         "q": [],            "target": []},
    }
    return avg, csv_data


def make_chirp_summary_figure(all_joint_results, out_path, k_current, d_current):
    """4×3 grid of Bode magnitude plots for all chirp-analyzed joints."""
    cols, rows = 3, 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 14))
    axes_flat = axes.flatten()

    for idx, (jidx, jname, _csv, avg, J_joint) in enumerate(all_joint_results):
        ax  = axes_flat[idx]
        wn  = avg["wn_rad_s"]
        zeta = avg["zeta"]
        bw   = avg["bandwidth_hz"]
        f_hz  = np.array(avg.get("_freq_hz",   []))
        H_mag = np.array(avg.get("_H_mag",     []))
        Coh   = np.array(avg.get("_coherence", []))
        coh_bw   = avg.get("_coh_bw_hz",   None)
        extrap   = avg.get("_extrapolated", False)

        if len(f_hz) > 2:
            H_db = 20.0 * np.log10(H_mag + 1e-30)
            ax.semilogx(f_hz[1:], H_db[1:], "b-", lw=0.7, alpha=0.5, label="measured")
            # Shade coherent band
            if len(Coh) == len(f_hz):
                coh_mask = Coh >= 0.50
                if coh_mask.any():
                    ax.fill_between(f_hz[coh_mask], H_db[coh_mask] - 40,
                                    H_db[coh_mask] + 40,
                                    alpha=0.12, color="green")
        f_model = np.logspace(np.log10(0.4), np.log10(40.0), 300)
        w_model = 2.0 * np.pi * f_model
        if wn > 0:
            H_m = wn**2 / (wn**2 - w_model**2 + 2j * zeta * wn * w_model)
            lbl = "fit" + (" ⚠" if extrap else "")
            ax.semilogx(f_model, 20*np.log10(np.abs(H_m)), "r-", lw=1.2, alpha=0.85,
                        label=lbl)
        if J_joint > 0:
            wn_c = np.sqrt(k_current / J_joint)
            z_c  = d_current / (2.0 * np.sqrt(k_current * J_joint))
            H_c  = wn_c**2 / (wn_c**2 - w_model**2 + 2j * z_c * wn_c * w_model)
            ax.semilogx(f_model, 20*np.log10(np.abs(H_c)), "g:", lw=1.0, alpha=0.7)

        coh_str = f" coh≤{coh_bw:.1f}Hz" if coh_bw else ""
        ax.set_title(
            f"J{jidx}  {jname}\n"
            f"ωn={wn:.1f} r/s  ζ={zeta:.2f}  BW={bw:.1f} Hz{coh_str}",
            fontsize=7.5)
        ax.set_xlabel("Hz", fontsize=7)
        ax.set_ylabel("|H| dB", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, which="both")
        ax.set_xlim([0.4, 40])
        ax.set_ylim([-40, 70])
        if idx == 0:
            ax.legend(fontsize=6.5, loc="lower left")

    for j in range(len(all_joint_results), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(
        "SysID Chirp — All 12 Joints  (Bode magnitude)\n"
        "blue = measured    red = 2nd-order fit    green dot = current sim",
        fontsize=9, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[analyze] Chirp summary figure → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", default="logs/sysid")
    parser.add_argument("--out_dir", default=None,
                        help="Output dir for plots (default: <in_dir>/plots)")
    parser.add_argument("--k_current", type=float, default=11.1,
                        help="Current sim stiffness for comparison (default 11.1)")
    parser.add_argument("--d_current", type=float, default=1.1,
                        help="Current sim damping for comparison (default 1.1)")
    parser.add_argument("--J_kgm2", type=float, default=DEFAULT_J_KGM2,
                        help=f"Fallback joint inertia when URDF not found "
                             f"(default {DEFAULT_J_KGM2})")
    parser.add_argument("--urdf", type=str, default=_DEFAULT_URDF,
                        help="Path to URDF for automatic joint inertia extraction "
                             "(default: Dextra_lowerbody.urdf)")
    parser.add_argument("--mirror", action="store_true",
                        help="Mirror fitted parameters to bilateral symmetric partners. "
                             "Use when only one side (e.g. --joints 0,2,4,6,8,10) "
                             "was collected. Partners share the same k_eff/d_eff.")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.in_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    # Load manifest
    manifest_path = os.path.join(args.in_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"[analyze] Step: ±{manifest['step_deg']}°, "
              f"sample_hz={manifest['sample_hz']}")

    # Find all CSV files
    csv_files = sorted(glob(os.path.join(args.in_dir, "joint*.csv")))
    if not csv_files:
        print(f"[analyze] No CSVs found in {args.in_dir}")
        sys.exit(1)

    # --- Load URDF inertias (per joint) ---
    # Joint names used in sysid are the 'name' field from config.yaml, which
    # must match the URDF joint names (e.g. "L_HipYaw_Joint").
    urdf_J: dict[str, float] = {}
    if os.path.exists(args.urdf):
        # Collect all joint names from CSV filenames
        candidate_jnames = []
        for csv_path in sorted(glob(os.path.join(args.in_dir, "joint*.csv"))):
            fname = os.path.basename(csv_path).replace(".csv", "")
            parts = fname.split("_", 1)
            if len(parts) == 2:
                candidate_jnames.append(parts[1])
        urdf_J = extract_urdf_inertias(args.urdf, candidate_jnames)
        if urdf_J:
            print(f"[analyze] URDF inertias extracted from: {args.urdf}")
            for jn, Jv in urdf_J.items():
                print(f"          {jn}: J = {Jv:.2e} kg·m²")
        else:
            print(f"[analyze] URDF found but no matching joints — using fallback J={args.J_kgm2}")
    else:
        print(f"[analyze] URDF not found ({args.urdf}) — using fallback J={args.J_kgm2}")
    print()

    # Analyze each joint
    summary = []
    all_joint_results = []
    for csv_path in csv_files:
        fname = os.path.basename(csv_path).replace(".csv", "")
        # joint00_L_HipYaw_Joint
        parts = fname.split("_", 1)
        jidx = int(parts[0].replace("joint", ""))
        jname = parts[1]

        J_for_joint = urdf_J.get(jname, args.J_kgm2)

        # Auto-detect step vs chirp from CSV header (first column name)
        with open(csv_path) as _hdr:
            _is_chirp = _hdr.readline().split(",")[0].strip() == "t_s"

        if _is_chirp:
            avg, csv_data = analyze_joint_chirp(
                jidx, jname, csv_path, out_dir,
                k_current=args.k_current, d_current=args.d_current,
                J_kgm2=J_for_joint)
        else:
            avg, per_dir, png, csv_data = analyze_joint(
                jidx, jname, csv_path, out_dir,
                k_current=args.k_current, d_current=args.d_current,
                J_kgm2=J_for_joint)
        summary.append((jidx, jname, avg, J_for_joint))
        all_joint_results.append((jidx, jname, csv_data, avg, J_for_joint))
        k_rec = J_for_joint * avg["wn_rad_s"]**2
        d_rec = 2 * avg["zeta"] * avg["wn_rad_s"] * J_for_joint
        print(f"  J{jidx:2d} {jname:24s}  "
              f"ωn={avg['wn_rad_s']:6.2f} rad/s  ζ={avg['zeta']:.3f}  "
              f"BW={avg['bandwidth_hz']:5.2f}Hz  "
              f"rise={avg['rise_time_s']*1000:5.0f}ms  "
              f"OS={avg['overshoot_pct']:4.1f}%  "
              f"[J={J_for_joint:.2e}  k→{k_rec:.3f}  d→{d_rec:.4f}]")

    # All-joints summary figure (step → normalized step response; chirp → Bode grid)
    summary_png = os.path.join(out_dir, "summary.png")
    if all_joint_results:
        with open(os.path.join(args.in_dir,
                               f"joint{all_joint_results[0][0]:02d}_"
                               f"{all_joint_results[0][1]}.csv")) as _hdr:
            _summary_is_chirp = _hdr.readline().split(",")[0].strip() == "t_s"
        if _summary_is_chirp:
            make_chirp_summary_figure(all_joint_results, summary_png,
                                      args.k_current, args.d_current)
        else:
            make_all_joints_figure(all_joint_results, summary_png,
                                   args.k_current, args.d_current)

    # Recommendations file
    rec_path = os.path.join(args.in_dir, "recommendations.txt")
    with open(rec_path, "w") as f:
        f.write("=" * 78 + "\n")
        f.write("SysID Recommendations for ImplicitActuatorCfg\n")
        f.write("=" * 78 + "\n\n")
        if urdf_J:
            f.write(f"Joint inertias extracted from: {args.urdf}\n")
        else:
            f.write(f"Fallback joint inertia J = {args.J_kgm2} kg·m² (URDF not found)\n")
        f.write(f"Current sim:  stiffness={args.k_current},  damping={args.d_current}\n\n")
        f.write(f"{'Joint':<26} {'J(kg·m²)':>10} {'ωn':>8} {'ζ':>6} {'BW(Hz)':>8} "
                f"{'k_rec':>8} {'d_rec':>8} {'sp(r/s)':>9} {'d_alt':>7}\n")
        f.write("-" * 96 + "\n")

        wn_all, zeta_all, k_all, d_all = [], [], [], []
        k_warn_joints = []
        for jidx, jname, avg, J_j in summary:
            wn   = avg["wn_rad_s"]
            zeta = avg["zeta"]
            k_rec = J_j * wn**2
            d_rec = 2 * zeta * wn * J_j
            # slow_pole = ωn/(2ζ) = k/d — J-free, directly observable from rolloff slope
            slow_pole = wn / (2.0 * zeta) if zeta > 0 else float("nan")
            # d_alt: infer d assuming hardware k = K_AX18A_HW; avoids J uncertainty
            d_alt = K_AX18A_HW / slow_pole if slow_pole > 0 else float("nan")
            wn_all.append(wn); zeta_all.append(zeta)
            k_all.append(k_rec); d_all.append(d_rec)
            # Flag joints where k_rec deviates >3× from hardware value
            extrap = avg.get("_extrapolated", False)
            k_bad  = (k_rec > K_AX18A_HW * 3) or (k_rec < K_AX18A_HW / 3)
            flag   = " ⚠" if (extrap or k_bad) else "  "
            k_warn_joints.append((jidx, jname, extrap, k_bad, slow_pole, d_alt))
            f.write(f"J{jidx:2d} {jname:<22} {J_j:10.2e} {wn:8.2f} {zeta:6.3f} "
                    f"{avg['bandwidth_hz']:8.2f} {k_rec:8.3f} {d_rec:8.4f} "
                    f"{slow_pole:9.2f} {d_alt:7.4f}{flag}\n")

        f.write("-" * 96 + "\n")
        f.write(f"{'MEAN':<26} {'':>10} {np.mean(wn_all):8.2f} {np.mean(zeta_all):6.3f} "
                f"{np.mean([s[2]['bandwidth_hz'] for s in summary]):8.2f} "
                f"{np.mean(k_all):8.3f} {np.mean(d_all):8.4f}\n")
        f.write(f"{'MEDIAN':<26} {'':>10} {np.median(wn_all):8.2f} {np.median(zeta_all):6.3f} "
                f"{'':>8} {np.median(k_all):8.3f} {np.median(d_all):8.4f}\n\n")

        f.write("RECOMMENDED (use median for robustness):\n")
        f.write(f"  ImplicitActuatorCfg(\n")
        f.write(f"      stiffness = {np.median(k_all):.3f},   # was {args.k_current}\n")
        f.write(f"      damping   = {np.median(d_all):.4f},   # was {args.d_current}\n")
        f.write(f"      effort_limit = 1.8,             # AX-18A stall torque\n")
        f.write(f"      ...\n  )\n\n")

        # Warn about flagged joints
        warn_list = [(ji, jn, ep, kb, sp, da) for ji, jn, ep, kb, sp, da
                     in k_warn_joints if ep or kb]
        if warn_list:
            f.write("⚠ FLAGGED JOINTS (k_rec unreliable — use d_alt with k=K_AX18A_HW):\n")
            for ji, jn, ep, kb, sp, da in warn_list:
                reason = []
                if ep: reason.append("fit extrapolated beyond coherent band")
                if kb: reason.append(f"k_rec deviates >3× from K_AX18A_HW={K_AX18A_HW}")
                f.write(f"  J{ji:2d} {jn:<22}: {'; '.join(reason)}\n")
                f.write(f"        → use  stiffness={K_AX18A_HW:.1f}  damping={da:.4f}"
                        f"  (anchored to hardware k, d inferred from slow_pole={sp:.1f} r/s)\n")
            f.write("\n")
            f.write("  slow_pole = ωn/(2ζ) = k/d is J-free and directly observable from\n")
            f.write("  the Bode rolloff slope.  Individual k and d require accurate J (CRB).\n")
            f.write("  For ⚠ joints: trust slow_pole; anchor k=11.1 (hardware compliance);\n")
            f.write("  infer d = k/slow_pole.  Alternatively, re-run with step-response mode\n")
            f.write("  (less body-mode sensitive for heavily overdamped joints).\n\n")

        f.write("NOTE: J values are CRB (Composite Rigid Body) — full downstream subtree.\n")
        f.write("      effort_limit=1.8 N·m models the AX-18A compliance saturation.\n")
        f.write("      The stiffness above corresponds to the slope zone (slope=32 default).\n")
        f.write("      Deadband (~0.3°) is negligible and ignored.\n")

        # ---- Mirror to bilateral symmetric partners -------------------------
        if args.mirror:
            analyzed_indices = {s[0] for s in summary}
            mirrored = []
            for jidx, jname, avg, J_j in summary:
                if jidx not in SYMMETRIC_PAIRS:
                    continue
                pidx, _ = SYMMETRIC_PAIRS[jidx]
                if pidx in analyzed_indices:
                    continue  # partner was explicitly measured, skip
                # Infer partner URDF J by name (L_→R_ or R_→L_)
                if "L_" in jname:
                    pname = jname.replace("L_", "R_", 1)
                elif "R_" in jname:
                    pname = jname.replace("R_", "L_", 1)
                else:
                    pname = f"mirror_of_{jname}"
                J_p = urdf_J.get(pname, J_j)  # same J if URDF symmetric
                mirrored.append((pidx, pname, avg, J_p, jidx))

            if mirrored:
                f.write("\n[Mirrored to symmetric partners (--mirror)]\n")
                f.write(f"{'Joint':<26} {'J(kg·m²)':>10} {'ωn':>8} {'ζ':>6} "
                        f"{'BW(Hz)':>8} {'k_rec':>8} {'d_rec':>8} {'source':>12}\n")
                f.write("-" * 90 + "\n")
                for pidx, pname, avg, J_p, src_idx in sorted(mirrored):
                    wn   = avg["wn_rad_s"]
                    zeta = avg["zeta"]
                    k_p  = J_p * wn**2
                    d_p  = 2 * zeta * wn * J_p
                    f.write(f"J{pidx:2d} {pname:<22} {J_p:10.2e} {wn:8.2f} {zeta:6.3f} "
                            f"{avg['bandwidth_hz']:8.2f} {k_p:8.3f} {d_p:8.4f} "
                            f"  mirror J{src_idx:2d}\n")
                f.write("\nNOTE: mirrored joints share measured k_eff/d_eff (bilateral symmetry).\n")
                f.write("      Verify URDF J_partner ≈ J_primary before using these values.\n")

    print(f"[analyze] Recommendations: {rec_path}")
    with open(rec_path) as f:
        print("\n" + f.read())


if __name__ == "__main__":
    main()
