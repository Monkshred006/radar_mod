"""Camera rotation trajectories for non-destructive 360° view generation."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from module_09_3d.config import RotationConfig


class RotationTrajectory:
    """Computes a smooth circular camera orbit around a scene target."""

    def __init__(self, config: RotationConfig) -> None:
        self.config = config

    def get_angles_degrees(self) -> List[float]:
        """Return list of unique azimuth angles in degrees."""
        start = self.config.rotation_start_deg
        end = self.config.rotation_end_deg
        step = self.config.rotation_step_deg

        span = abs(end - start)
        if span >= 360.0:
            # Avoid duplicate 0° and 360°
            n_steps = int(round(360.0 / step))
            angles = [start + i * step for i in range(n_steps)]
        else:
            angles = list(np.arange(start, end, step))

        if self.config.clockwise:
            angles = [-a for a in angles]

        return angles

    def get_camera_positions(
        self,
        target: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float32),
    ) -> List[Tuple[float, np.ndarray]]:
        """Compute (azimuth_deg, camera_pos_xyz) for each rotation frame."""
        angles = self.get_angles_degrees()
        dist = self.config.distance
        elev_rad = np.radians(self.config.elevation_deg)

        z = target[2] + dist * np.sin(elev_rad)
        r_xy = dist * np.cos(elev_rad)

        positions = []
        for deg in angles:
            az_rad = np.radians(deg)
            # In right-handed z-up (x=lateral, y=depth):
            # azimuth 0: camera is on -y axis looking towards +y
            x = target[0] + r_xy * np.sin(az_rad)
            y = target[1] - r_xy * np.cos(az_rad)
            pos = np.array([x, y, z], dtype=np.float32)
            positions.append((deg, pos))

        return positions
