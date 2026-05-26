"""ONNX Runtime wrapper for the Teacher Policy.

The ONNX model already has normalization baked in:
  input(43D) → RunningStandardScale → clip(-5,5) → MLP → output(12D)
"""

import numpy as np
import onnxruntime as ort


class TeacherPolicyONNX:
    """Teacher policy inference via ONNX Runtime."""

    def __init__(self, onnx_path: str):
        self.session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Run teacher policy.

        Args:
            obs: Observation array of shape (43,) — [encoder_24D, priv_est_19D].

        Returns:
            Action array of shape (12,) in [-1, 1] (normalized).
        """
        obs_input = obs.reshape(1, -1).astype(np.float32)
        result = self.session.run(None, {self.input_name: obs_input})
        return result[0].flatten()
