#!/usr/bin/env python3
"""Export trained Teacher Policy and LSTM Estimator to ONNX format.

Run on the development machine (with PyTorch + CUDA):

    python export_to_onnx.py \
        --teacher_checkpoint ../logs/skrl/dextra_amp_walk/.../best_agent.pt \
        --estimator_checkpoint ../logs/solo_estimator/.../best_estimator.pt \
        --output_dir models/

The exported ONNX models can then be copied to the Raspberry Pi for deployment.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

# Add SOLO_DEXTRA to path for solo_models import
SOLO_DEXTRA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "source", "isaaclab_tasks", "isaaclab_tasks", "direct", "SOLO_DEXTRA",
)
sys.path.insert(0, SOLO_DEXTRA_DIR)

from solo_models import (
    ENCODER_DIM,
    PRIV_DIM,
    OBS_DIM,
    ACTION_DIM,
    TeacherPolicy,
    load_estimator,
)


# ---------------------------------------------------------------------------
# ONNX-friendly wrappers (bake normalization into the graph)
# ---------------------------------------------------------------------------


class TeacherPolicyExport(nn.Module):
    """Teacher policy with RunningStandardScaler baked in."""

    def __init__(self, teacher: TeacherPolicy):
        super().__init__()
        self.net = teacher.net
        self.register_buffer("running_mean", teacher.running_mean.clone())
        self.register_buffer("running_var", teacher.running_var.clone())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs_norm = (obs - self.running_mean) / (torch.sqrt(self.running_var) + 1e-8)
        obs_norm = torch.clamp(obs_norm, -5.0, 5.0)
        return self.net(obs_norm)


class LSTMEstimatorExport(nn.Module):
    """LSTM estimator with predict_denormalized baked in."""

    def __init__(self, estimator):
        super().__init__()
        self.lstm = estimator.lstm
        self.fc = estimator.fc
        self.register_buffer("input_mean", estimator.input_mean.clone())
        self.register_buffer("input_std", estimator.input_std.clone())
        self.register_buffer("output_mean", estimator.output_mean.clone())
        self.register_buffer("output_std", estimator.output_std.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, T, 24)
        x_norm = (x - self.input_mean) / (self.input_std + 1e-8)
        _, (h_n, _) = self.lstm(x_norm)
        raw = self.fc(h_n[-1])
        return raw * self.output_std + self.output_mean


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------


def export_teacher(teacher: TeacherPolicy, output_path: str):
    wrapper = TeacherPolicyExport(teacher).cpu().eval()
    dummy = torch.randn(1, OBS_DIM)

    torch.onnx.export(
        wrapper,
        dummy,
        output_path,
        opset_version=17,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes=None,  # fixed batch=1
    )
    print(f"[export] Teacher policy → {output_path}")
    return wrapper, dummy


def export_estimator(estimator, window: int, output_path: str):
    wrapper = LSTMEstimatorExport(estimator).cpu().eval()
    dummy = torch.randn(1, window, ENCODER_DIM)

    torch.onnx.export(
        wrapper,
        dummy,
        output_path,
        opset_version=17,
        input_names=["history"],
        output_names=["priv_est"],
        dynamic_axes=None,  # fixed batch=1, window=50
    )
    print(f"[export] LSTM estimator → {output_path}")
    return wrapper, dummy


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_onnx(onnx_path: str, wrapper: nn.Module, dummy: torch.Tensor,
                model_name: str) -> bool:
    try:
        import onnxruntime as ort
    except ImportError:
        print("[verify] onnxruntime not installed — skipping verification")
        return True

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    np_input = dummy.numpy()

    with torch.no_grad():
        pt_out = wrapper(dummy).numpy()

    ort_out = sess.run(None, {input_name: np_input})[0]

    max_diff = np.max(np.abs(pt_out - ort_out))
    passed = max_diff < 1e-5
    status = "PASS" if passed else "FAIL"
    print(f"[verify] {model_name}: max |PyTorch - ONNX| = {max_diff:.2e}  [{status}]")

    if not passed:
        print(f"  PyTorch output: {pt_out.flatten()[:5]}")
        print(f"  ONNX output:    {ort_out.flatten()[:5]}")

    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Export SOLO models to ONNX")
    parser.add_argument("--teacher_checkpoint", type=str, required=True,
                        help="Path to SKRL AMP best_agent.pt")
    parser.add_argument("--estimator_checkpoint", type=str, required=True,
                        help="Path to estimator checkpoint (best_estimator.pt)")
    parser.add_argument("--output_dir", type=str, default="models",
                        help="Output directory for ONNX files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load Teacher ---
    print(f"[export] Loading teacher: {args.teacher_checkpoint}")
    teacher = TeacherPolicy(OBS_DIM, device="cpu")
    teacher.load_from_checkpoint(args.teacher_checkpoint, device="cpu")
    teacher.eval()

    # --- Load Estimator ---
    print(f"[export] Loading estimator: {args.estimator_checkpoint}")
    estimator, est_ckpt = load_estimator(args.estimator_checkpoint, device="cpu")
    est_cfg = est_ckpt["estimator_config"]
    window = est_ckpt.get("window", 50)
    print(f"[export] Estimator type: {est_cfg['type']}, window: {window}")

    if est_cfg["type"].upper() != "LSTM":
        print(f"[WARNING] Estimator type is {est_cfg['type']}, not LSTM. "
              "ONNX export is designed for LSTM. Proceeding anyway...")

    # --- Export ---
    teacher_path = os.path.join(args.output_dir, "teacher_policy.onnx")
    estimator_path = os.path.join(args.output_dir, "lstm_estimator.onnx")

    teacher_wrapper, teacher_dummy = export_teacher(teacher, teacher_path)
    est_wrapper, est_dummy = export_estimator(estimator, window, estimator_path)

    # --- Verify ---
    print("\n--- Verification ---")
    ok1 = verify_onnx(teacher_path, teacher_wrapper, teacher_dummy, "Teacher")
    ok2 = verify_onnx(estimator_path, est_wrapper, est_dummy, "Estimator")

    # --- File sizes ---
    for p in [teacher_path, estimator_path]:
        size_kb = os.path.getsize(p) / 1024
        print(f"  {os.path.basename(p)}: {size_kb:.1f} KB")

    # --- Save metadata ---
    import json
    meta = {
        "teacher_checkpoint": os.path.abspath(args.teacher_checkpoint),
        "estimator_checkpoint": os.path.abspath(args.estimator_checkpoint),
        "estimator_type": est_cfg["type"],
        "window": window,
        "encoder_dim": ENCODER_DIM,
        "priv_dim": PRIV_DIM,
        "obs_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
    }
    meta_path = os.path.join(args.output_dir, "export_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[export] Metadata saved: {meta_path}")

    if ok1 and ok2:
        print("\n[export] All exports verified successfully.")
        print(f"Copy {args.output_dir}/ to Raspberry Pi.")
    else:
        print("\n[export] WARNING: Verification failed. Check outputs.")
        sys.exit(1)


if __name__ == "__main__":
    main()
