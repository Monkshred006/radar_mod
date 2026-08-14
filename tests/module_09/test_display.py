"""Tests for FrameStream."""

import numpy as np
import pytest

from module_09_3d.config import OLEDConfig
from module_09_3d.display import FrameStream


class TestDisplayStream:
    def test_push_and_next_frame(self):
        stream = FrameStream()
        f1 = np.ones((10, 10, 3), dtype=np.uint8)
        f2 = np.zeros((10, 10, 3), dtype=np.uint8)

        stream.push_frame(f1)
        stream.push_frame(f2)

        assert stream.buffered_frame_count == 2
        assert np.array_equal(stream.next_frame(), f1)
        assert np.array_equal(stream.next_frame(), f2)
        assert stream.next_frame() is None

    def test_loop_playback(self):
        cfg = OLEDConfig(loop_playback=True)
        stream = FrameStream(cfg)

        frames = [np.full((5, 5, 3), i, dtype=np.uint8) for i in range(3)]
        stream.load_sequence(frames)

        # First pass from buffer
        for i in range(3):
            assert stream.next_frame()[0, 0, 0] == i

        # Second pass from loop
        for i in range(3):
            assert stream.next_frame()[0, 0, 0] == i

    def test_frame_dropping_on_overflow(self):
        cfg = OLEDConfig(buffer_max_size=3, drop_frames_on_overflow=True)
        stream = FrameStream(cfg)

        for i in range(5):
            stream.push_frame(np.full((2, 2, 3), i, dtype=np.uint8))

        assert stream.buffered_frame_count == 3
        assert stream.dropped_frame_count == 2
        # Oldest frames 0 and 1 dropped, buffer has 2, 3, 4
        assert stream.next_frame()[0, 0, 0] == 2
