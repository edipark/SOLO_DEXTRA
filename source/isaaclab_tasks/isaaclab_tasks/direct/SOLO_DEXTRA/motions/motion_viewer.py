# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
import torch

import mpl_toolkits.mplot3d  # noqa: F401

try:
    from .motion_loader import MotionLoader
except ImportError:
    from motion_loader import MotionLoader

# Dextra lower-body kinematic chain (parent -> child bone connections).
# Each tuple is (parent_body_name, child_body_name).
DEXTRA_BONES = [
    ("base_link", "L_HipYaw_Link_1"),
    ("L_HipYaw_Link_1", "L_HipRoll_Link_1"),
    ("L_HipRoll_Link_1", "L_Thigh_Link_1"),
    ("L_Thigh_Link_1", "L_Calf_Link_1"),
    ("L_Calf_Link_1", "L_AnklePitch_Link_1"),
    ("L_AnklePitch_Link_1", "L_AnkleRoll_Link_1"),
    ("base_link", "R_HipYaw_Link_1"),
    ("R_HipYaw_Link_1", "R_HipRoll_Link_1"),
    ("R_HipRoll_Link_1", "R_Thigh_Link_1"),
    ("R_Thigh_Link_1", "R_Calf_Link_1"),
    ("R_Calf_Link_1", "R_AnklePitch_Link_1"),
    ("R_AnklePitch_Link_1", "R_AnkleRoll_Link_1"),
]


class MotionViewer:
    """
    Helper class to visualize motion data from NumPy-file format.
    Draws skeleton bones for Dextra and annotates body names.
    """

    def __init__(
        self,
        motion_file: str,
        device: torch.device | str = "cpu",
        render_scene: bool = False,
        *,
        print_base_velocity: bool = False,
        print_base_velocity_interval: int = 30,
    ) -> None:
        """Load a motion file and initialize the internal variables.

        Args:
            motion_file: Motion file path to load.
            device: The device to which to load the data.
            render_scene: Whether the scene (space occupied by the skeleton during movement)
                is rendered instead of a reduced view of the skeleton.
            print_base_velocity: If True, print base linear/angular velocity to stdout periodically.
            print_base_velocity_interval: Print every N animation frames when ``print_base_velocity`` is True.

        Raises:
            AssertionError: If the specified motion file doesn't exist.
        """
        self._figure = None
        self._figure_axes = None
        self._render_scene = render_scene
        self._print_base_velocity = print_base_velocity
        self._print_base_velocity_interval = max(1, int(print_base_velocity_interval))

        self._motion_loader = MotionLoader(motion_file=motion_file, device=device)

        self._num_frames = self._motion_loader.num_frames
        self._current_frame = 0
        self._body_positions = self._motion_loader.body_positions.cpu().numpy()
        self._body_names = self._motion_loader.body_names

        try:
            self._base_body_index = self._body_names.index("base_link")
        except ValueError:
            self._base_body_index = 0
        self._base_lin_vel = self._motion_loader.body_linear_velocities[:, self._base_body_index, :].cpu().numpy()
        self._base_ang_vel = self._motion_loader.body_angular_velocities[:, self._base_body_index, :].cpu().numpy()

        # Build bone index pairs for fast drawing
        name_to_idx = {name: i for i, name in enumerate(self._body_names)}
        self._bone_pairs = []
        for parent, child in DEXTRA_BONES:
            if parent in name_to_idx and child in name_to_idx:
                self._bone_pairs.append((name_to_idx[parent], name_to_idx[child]))

        print(f"\nMotion: {motion_file}")
        print(f"  Duration: {self._motion_loader.duration:.2f}s  |  Frames: {self._num_frames}")
        print(f"  dt: {self._motion_loader.dt:.4f}s  |  fps: {1.0 / self._motion_loader.dt:.1f}")
        print(f"\nBodies ({len(self._body_names)}):")
        for i, name in enumerate(self._body_names):
            minimum = np.min(self._body_positions[:, i], axis=0).round(decimals=3)
            maximum = np.max(self._body_positions[:, i], axis=0).round(decimals=3)
            print(f"  [{i:2d}] {name:30s}  min={minimum}  max={maximum}")

        print(f"\nDOFs ({len(self._motion_loader.dof_names)}):")
        dof_pos = self._motion_loader.dof_positions.cpu().numpy()
        for i, name in enumerate(self._motion_loader.dof_names):
            lo = dof_pos[:, i].min()
            hi = dof_pos[:, i].max()
            print(f"  [{i:2d}] {name:30s}  range=[{lo:+.3f}, {hi:+.3f}] rad")

        bi = self._base_body_index
        v = self._base_lin_vel
        w = self._base_ang_vel
        print(f"\nBase ({self._body_names[bi]}) velocity (world frame, from motion file):")
        print(
            f"  linear v:  min=[{v[:, 0].min():+.4f}, {v[:, 1].min():+.4f}, {v[:, 2].min():+.4f}]  "
            f"max=[{v[:, 0].max():+.4f}, {v[:, 1].max():+.4f}, {v[:, 2].max():+.4f}] m/s"
        )
        print(
            f"  angular ω: min=[{w[:, 0].min():+.4f}, {w[:, 1].min():+.4f}, {w[:, 2].min():+.4f}]  "
            f"max=[{w[:, 0].max():+.4f}, {w[:, 1].max():+.4f}, {w[:, 2].max():+.4f}] rad/s"
        )

    def _drawing_callback(self, frame: int) -> None:
        """Drawing callback called each frame."""
        vertices = self._body_positions[self._current_frame]
        self._figure_axes.clear()

        # Draw bones
        for pi, ci in self._bone_pairs:
            xs = [vertices[pi, 0], vertices[ci, 0]]
            ys = [vertices[pi, 1], vertices[ci, 1]]
            zs = [vertices[pi, 2], vertices[ci, 2]]
            color = "#2196F3" if "L_" in self._body_names[ci] else "#F44336"
            self._figure_axes.plot(xs, ys, zs, color=color, linewidth=2.5, solid_capstyle="round")

        # Draw joints
        left_mask = np.array(["L_" in n for n in self._body_names])
        right_mask = np.array(["R_" in n for n in self._body_names])
        base_mask = ~left_mask & ~right_mask

        if base_mask.any():
            self._figure_axes.scatter(*vertices[base_mask].T, c="#4CAF50", s=60, depthshade=False, zorder=5)
        if left_mask.any():
            self._figure_axes.scatter(*vertices[left_mask].T, c="#2196F3", s=40, depthshade=False, zorder=5)
        if right_mask.any():
            self._figure_axes.scatter(*vertices[right_mask].T, c="#F44336", s=40, depthshade=False, zorder=5)

        # Axis limits
        if self._render_scene:
            minimum = np.min(self._body_positions.reshape(-1, 3), axis=0)
            maximum = np.max(self._body_positions.reshape(-1, 3), axis=0)
            center = 0.5 * (maximum + minimum)
            diff = 0.75 * (maximum - minimum)
        else:
            minimum = np.min(vertices, axis=0)
            maximum = np.max(vertices, axis=0)
            center = 0.5 * (maximum + minimum)
            diff = np.array([0.75 * np.max(maximum - minimum).item()] * 3)

        self._figure_axes.set_xlim((center[0] - diff[0], center[0] + diff[0]))
        self._figure_axes.set_ylim((center[1] - diff[1], center[1] + diff[1]))
        self._figure_axes.set_zlim((center[2] - diff[2], center[2] + diff[2]))
        self._figure_axes.set_box_aspect(aspect=diff / max(diff[0], 1e-6))

        # Ground plane
        gx, gy = np.meshgrid(
            [center[0] - diff[0], center[0] + diff[0]],
            [center[1] - diff[1], center[1] + diff[1]],
        )
        self._figure_axes.plot_surface(gx, gy, np.zeros_like(gx), color="green", alpha=0.15)

        t = self._current_frame * self._motion_loader.dt
        self._figure_axes.set_xlabel("X")
        self._figure_axes.set_ylabel("Y")
        self._figure_axes.set_zlabel("Z")
        self._figure_axes.set_title(
            f"Frame {self._current_frame}/{self._num_frames}  |  t={t:.2f}s  |  "
            f"base_z={vertices[0, 2]:.3f}m"
        )

        self._current_frame += 1
        if self._current_frame >= self._num_frames:
            self._current_frame = 0

    def show(self) -> None:
        """Show motion animation."""
        self._figure = plt.figure(figsize=(10, 8))
        self._figure_axes = self._figure.add_subplot(projection="3d")
        self._animation = matplotlib.animation.FuncAnimation(
            fig=self._figure,
            func=self._drawing_callback,
            frames=self._num_frames,
            interval=1000 * self._motion_loader.dt,
        )
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dextra motion skeleton viewer")
    parser.add_argument("--file", type=str, required=True, help="Motion .npz file")
    parser.add_argument(
        "--render-scene",
        action="store_true",
        default=False,
        help="Render full scene view (track entire trajectory) instead of following the skeleton.",
    )
    parser.add_argument("--matplotlib-backend", type=str, default="TkAgg", help="Matplotlib interactive backend")
    args, _ = parser.parse_known_args()

    matplotlib.use(args.matplotlib_backend)

    viewer = MotionViewer(args.file, render_scene=args.render_scene)
    viewer.show()
