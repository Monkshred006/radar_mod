"""Tests for OLED display backends."""

import numpy as np
import pytest

from module_09_3d.config import OLEDConfig
from module_09_3d.oled import (
    HardwareDisplayBackend,
    SimulatedDisplayBackend,
    build_display_backend,
)


class TestOLED:
    def test_simulated_backend_display_and_history(self):
        cfg = OLEDConfig(display_width=128, display_height=128, brightness=0.5)
        backend = SimulatedDisplayBackend(cfg)

        frame = np.full((128, 128, 3), 200, dtype=np.uint8)
        success = backend.display_frame(frame)

        assert success
        assert backend.current_frame is not None
        assert backend.current_frame.shape == (128, 128, 3)
        # Brightness scaled: 200 * 0.5 = 100
        assert backend.current_frame[0, 0, 0] == 100
        assert len(backend.frame_history) == 1

    def test_simulated_backend_resizes_mismatched_frames(self):
        cfg = OLEDConfig(display_width=64, display_height=64)
        backend = SimulatedDisplayBackend(cfg)

        large_frame = np.ones((128, 128, 3), dtype=np.uint8)
        backend.display_frame(large_frame)
        assert backend.current_frame.shape == (64, 64, 3)

    def test_hardware_backend_disconnected_raises(self):
        backend = HardwareDisplayBackend()
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="not connected"):
            backend.display_frame(frame)

    def test_factory_builds_correct_backend(self):
        sim = build_display_backend(OLEDConfig(backend_type="simulated"))
        assert isinstance(sim, SimulatedDisplayBackend)

        hw = build_display_backend(OLEDConfig(backend_type="hardware_interface"))
        assert isinstance(hw, HardwareDisplayBackend)
