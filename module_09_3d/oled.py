"""OLED display interface and software simulation backends for Module 9."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from module_09_3d.config import OLEDConfig
from module_09_3d.interfaces import DisplayBackend


class SimulatedDisplayBackend(DisplayBackend):
    """Software simulation display backend.

    Renders into an in-memory frame buffer and maintains frame history for testing.
    Does NOT require physical OLED hardware.
    """

    def __init__(self, config: Optional[OLEDConfig] = None) -> None:
        self.config = config or OLEDConfig()
        self.current_frame: Optional[np.ndarray] = None
        self.frame_history: List[np.ndarray] = []
        self._active: bool = True
        self.max_history: int = 100

    def display_frame(self, frame: np.ndarray) -> bool:
        """Receive and process a frame for display."""
        if not self._active:
            return False

        h, w = self.config.display_height, self.config.display_width

        # Resize / crop frame if resolution does not match
        if frame.shape[:2] != (h, w):
            processed = np.zeros((h, w, 3), dtype=np.uint8)
            copy_h = min(h, frame.shape[0])
            copy_w = min(w, frame.shape[1])
            processed[:copy_h, :copy_w] = frame[:copy_h, :copy_w]
        else:
            processed = frame.copy()

        # Apply brightness scaling
        if self.config.brightness < 1.0:
            processed = (processed.astype(np.float32) * self.config.brightness).astype(np.uint8)

        self.current_frame = processed
        self.frame_history.append(processed)
        if len(self.frame_history) > self.max_history:
            self.frame_history.pop(0)

        return True

    def is_active(self) -> bool:
        return self._active

    def close(self) -> None:
        self._active = False

    def clear(self) -> None:
        self.current_frame = None
        self.frame_history.clear()


class HardwareDisplayBackend(DisplayBackend):
    """Abstract interface contract for real OLED hardware controllers (e.g. SSD1306, SSD1351).

    NOTE: Physical hardware driver is configured during real deployment.
    """

    def __init__(self, config: Optional[OLEDConfig] = None) -> None:
        self.config = config or OLEDConfig()
        self._connected: bool = False

    def display_frame(self, frame: np.ndarray) -> bool:
        if not self._connected:
            raise RuntimeError("HardwareDisplayBackend: Physical OLED hardware not connected.")
        return True

    def is_active(self) -> bool:
        return self._connected

    def close(self) -> None:
        self._connected = False


def build_display_backend(config: OLEDConfig) -> DisplayBackend:
    """Factory: build the configured OLED display backend."""
    if config.backend_type == "simulated":
        return SimulatedDisplayBackend(config)
    elif config.backend_type == "hardware_interface":
        return HardwareDisplayBackend(config)
    else:
        raise ValueError(f"Unknown display backend_type: '{config.backend_type}'")
