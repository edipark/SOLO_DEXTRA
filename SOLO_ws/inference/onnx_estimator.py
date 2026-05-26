"""ONNX Runtime wrapper for the LSTM State Estimator with history buffer.

The ONNX model already has normalization baked in:
  input(1, window, 24) → InputNorm → LSTM → FC → OutputDenorm → output(1, 19)

This wrapper manages the sliding-window history buffer in numpy,
matching the play_teacher_with_estimator.py pattern (torch.roll → FIFO).
"""

import numpy as np
import onnxruntime as ort


class LSTMEstimatorONNX:
    """LSTM state estimator with sliding-window history buffer."""

    def __init__(self, onnx_path: str, window: int = 50, encoder_dim: int = 24):
        self.session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

        self.window = window
        self.encoder_dim = encoder_dim

        # History buffer: (1, window, encoder_dim) — zero-initialized
        self.history = np.zeros((1, window, encoder_dim), dtype=np.float32)
        self.valid_count = 0

    def update_and_predict(self, encoder_obs: np.ndarray) -> np.ndarray:
        """Update history buffer and run LSTM estimator.

        Args:
            encoder_obs: Encoder observation of shape (24,) — [joint_pos, joint_vel].

        Returns:
            Privileged state estimate of shape (19,).
        """
        # FIFO shift left, append new observation at the end
        self.history = np.roll(self.history, -1, axis=1)
        self.history[0, -1, :] = encoder_obs.astype(np.float32)
        self.valid_count += 1

        # Run ONNX inference
        result = self.session.run(None, {self.input_name: self.history})
        return result[0].flatten()

    def reset(self):
        """Reset history buffer (call on episode start/robot restart)."""
        self.history[:] = 0.0
        self.valid_count = 0
