"""Frame streaming and synchronization engine for Module 9."""

from __future__ import annotations

import collections
import time
from typing import Iterator, List, Optional

import numpy as np

from module_09_3d.config import OLEDConfig


class FrameStream:
    """Provides buffered, FPS-controlled frame streaming with overflow dropping policy."""

    def __init__(self, config: Optional[OLEDConfig] = None) -> None:
        self.config = config or OLEDConfig()
        self._buffer: collections.deque[np.ndarray] = collections.deque(
            maxlen=self.config.buffer_max_size
        )
        self._loop_frames: List[np.ndarray] = []
        self._loop_idx: int = 0
        self._dropped_frames: int = 0

    @property
    def buffered_frame_count(self) -> int:
        return len(self._buffer)

    @property
    def dropped_frame_count(self) -> int:
        return self._dropped_frames

    def push_frame(self, frame: np.ndarray) -> None:
        """Push a frame to the stream buffer."""
        if len(self._buffer) >= self.config.buffer_max_size and self.config.drop_frames_on_overflow:
            self._buffer.popleft()  # Drop oldest frame
            self._dropped_frames += 1
        self._buffer.append(frame)

    def load_sequence(self, frames: List[np.ndarray]) -> None:
        """Load a complete sequence for looping or sequential playback."""
        self._loop_frames = list(frames)
        self._loop_idx = 0
        for f in frames:
            self.push_frame(f)

    def next_frame(self) -> Optional[np.ndarray]:
        """Fetch the next available frame, handling looping if configured."""
        if self._buffer:
            return self._buffer.popleft()

        if self.config.loop_playback and self._loop_frames:
            f = self._loop_frames[self._loop_idx]
            self._loop_idx = (self._loop_idx + 1) % len(self._loop_frames)
            return f

        return None

    def stream_with_pacing(self, max_frames: Optional[int] = None) -> Iterator[np.ndarray]:
        """Generator yielding frames with precise target FPS pacing."""
        target_interval = 1.0 / max(self.config.target_fps, 1)
        count = 0

        while True:
            if max_frames is not None and count >= max_frames:
                break

            t0 = time.perf_counter()
            frame = self.next_frame()
            if frame is None:
                break

            yield frame
            count += 1

            elapsed = time.perf_counter() - t0
            sleep_time = target_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def clear(self) -> None:
        self._buffer.clear()
        self._loop_frames.clear()
        self._loop_idx = 0
        self._dropped_frames = 0
