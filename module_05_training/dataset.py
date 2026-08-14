"""Training dataset for Module 5 — correct temporal causality implementation.

CRITICAL DESIGN INVARIANT:
  Module 2 and Module 3 are applied to COMPLETE SCENE TIMELINES.
  Window samples are indexed from the resulting scene-level feature arrays.
  This guarantees:
    - Features at timestep t depend only on t, t-1, t-2, ... within the scene (causal)
    - No cold-start artifact at training window boundaries
    - Training-deployment consistency

  NEVER apply Module 2 / Module 3 independently to each 20-frame training window.
"""

from __future__ import annotations
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from module_01_radar_input.adapters.base import DefaultDirectoryAdapter
from module_01_radar_input.radar_loader import RadarLoader
from module_01_radar_input.config import RadarDatasetConfig


# ──────────────────────────────────────────────────────────────────────────────
# Scene-level feature cache
# ──────────────────────────────────────────────────────────────────────────────

class SceneFeatureCache:
    """Processes complete scene timelines through Module 2 → Module 3.

    Stores scene-level feature arrays keyed by scene_id. Once built,
    window samples can be indexed without reprocessing.

    Args:
        pipe2: A fitted SensorDSPPipeline instance (fitted on training data only).
        pipe3: A SensorFusionPipeline instance.
        channel_names: Ordered list of sensor channel names matching the
            radar data columns.
        cache_dir: If provided, precomputed scene features are saved/loaded
            from this directory.
    """

    def __init__(
        self,
        pipe2: Any,
        pipe3: Any,
        channel_names: List[str],
        cache_dir: Optional[str] = None,
    ):
        self.pipe2 = pipe2
        self.pipe3 = pipe3
        self.channel_names = channel_names
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._cache: Dict[str, Dict[str, Any]] = {}

    def build(self, scene_items: Dict[str, List[Dict[str, Any]]]) -> None:
        """Process all scenes through Module 2 → Module 3 in temporal order.

        Args:
            scene_items: Dict mapping scene_id → sorted list of frame item dicts
                (each item has 'frame_path' and 'timestamp').
        """
        loader = RadarLoader()
        for scene_id, items in scene_items.items():
            if scene_id in self._cache:
                continue

            # Try disk cache first
            if self.cache_dir is not None:
                cached = self._load_from_disk(scene_id)
                if cached is not None:
                    self._cache[scene_id] = cached
                    continue

            # Load all frames of this scene in temporal order
            frames = []
            timestamps = []
            for item in items:
                arr = loader.load_frame(item["frame_path"])
                frames.append(arr)
                timestamps.append(float(item.get("timestamp", 0.0)))

            if len(frames) == 0:
                continue

            # Stack into [T_scene, F]
            scene_radar = np.stack(frames, axis=0).astype(np.float32)
            scene_ts = np.array(timestamps, dtype=np.float64)

            # Build Module 1-style sample dict for the whole scene
            scene_sample = {
                "radar": torch.from_numpy(scene_radar),
                "timestamp": torch.from_numpy(scene_ts),
                "metadata": {"scene_id": scene_id},
            }

            # ── Module 2: apply to complete scene timeline ────────────────────
            from module_02_sensor_dsp.pipeline import SensorDSPPipeline
            raw_signals = SensorDSPPipeline.from_module1_sample(
                scene_sample, self.channel_names
            )
            m2_out = self.pipe2.process_offline(raw_signals)

            # ── Module 3: apply to complete scene timeline ────────────────────
            m3_out = self.pipe3.process_offline(m2_out)

            # Convert Module 3 tensors to numpy for indexing
            scene_features = self._m3_to_numpy(m3_out, scene_ts)

            self._cache[scene_id] = scene_features

            if self.cache_dir is not None:
                self._save_to_disk(scene_id, scene_features)

    def _m3_to_numpy(
        self, m3_out: Dict[str, Any], scene_ts: np.ndarray
    ) -> Dict[str, Any]:
        """Convert Module 3 output tensors to numpy arrays for efficient windowing."""
        result: Dict[str, Any] = {}
        for k, v in m3_out.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.detach().cpu().numpy()
            else:
                result[k] = v
        # Store original timestamps as fallback if not in m3_out
        if "timestamps" not in result:
            result["timestamps"] = scene_ts
        return result

    def get_window(
        self,
        scene_id: str,
        start_t: int,
        window_len: int,
    ) -> Dict[str, Any]:
        """Return a temporal window from a cached scene's feature arrays.

        Args:
            scene_id: Scene identifier.
            start_t: Start timestep index (0-indexed).
            window_len: Number of timesteps in the window (T).

        Returns:
            Dict with the same keys as Module 3 output but sliced to [window_len, ...].
        """
        scene = self._cache[scene_id]
        end_t = start_t + window_len
        window: Dict[str, Any] = {}
        for k, v in scene.items():
            if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] >= end_t:
                window[k] = torch.from_numpy(v[start_t:end_t].copy())
            else:
                window[k] = v  # pass non-temporal items as-is (e.g. group_map)
        return window

    def scene_length(self, scene_id: str) -> int:
        """Return the number of timesteps available for a scene."""
        scene = self._cache.get(scene_id, {})
        # Use 'tokens' or 'features' to determine length
        for key in ("tokens", "features", "timestamps"):
            v = scene.get(key)
            if isinstance(v, np.ndarray) and v.ndim >= 1:
                return v.shape[0]
        return 0

    def _save_to_disk(self, scene_id: str, data: Dict[str, Any]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        safe_name = scene_id.replace("/", "_").replace("\\", "_")
        path = self.cache_dir / f"scene_{safe_name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def _load_from_disk(self, scene_id: str) -> Optional[Dict[str, Any]]:
        safe_name = scene_id.replace("/", "_").replace("\\", "_")
        path = self.cache_dir / f"scene_{safe_name}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# PhotonShield training dataset
# ──────────────────────────────────────────────────────────────────────────────

class PhotonShieldDataset(Dataset):
    """PyTorch Dataset for Module 5 FP32 training.

    Each item is a (module3_window_dict, target) pair where:
    - module3_window_dict is a temporally-correct slice of a scene's feature
      timeline (produced by SceneFeatureCache, not per-window processing).
    - target is produced by the target_adapter callable.

    Args:
        scene_cache: A built SceneFeatureCache.
        target_adapter: Callable(window_dict) → target tensor.
        window_len: Sequence window length T.
        window_stride: Step between consecutive windows within a scene.
    """

    def __init__(
        self,
        scene_cache: SceneFeatureCache,
        target_adapter: Callable[[Dict[str, Any]], Any],
        window_len: int = 20,
        window_stride: int = 1,
    ):
        self.scene_cache = scene_cache
        self.target_adapter = target_adapter
        self.window_len = window_len
        self.window_stride = window_stride

        # Build window index: list of (scene_id, start_t)
        self._index: List[Tuple[str, int]] = []
        for scene_id in scene_cache._cache:
            T_scene = scene_cache.scene_length(scene_id)
            start = 0
            while start + window_len <= T_scene:
                self._index.append((scene_id, start))
                start += window_stride

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Any], Any]:
        scene_id, start_t = self._index[idx]
        window = self.scene_cache.get_window(scene_id, start_t, self.window_len)
        target = self.target_adapter(window)
        return window, target


# ──────────────────────────────────────────────────────────────────────────────
# Collation
# ──────────────────────────────────────────────────────────────────────────────

def collate_module3(
    batch: List[Tuple[Dict[str, Any], Any]]
) -> Tuple[Dict[str, Any], Any]:
    """Custom collate function for PhotonShieldDataset.

    Stacks tensor fields along a new batch dimension (dim=0).
    Non-tensor fields (group_map, metadata, etc.) are taken from the
    first sample — they are identical across all samples in a batch
    for a given configuration.

    Returns:
        (batched_module3_dict, batched_targets)
    """
    windows, targets = zip(*batch)

    # Collect all keys from module3 dicts
    keys = windows[0].keys()
    batched: Dict[str, Any] = {}
    for k in keys:
        vals = [w[k] for w in windows]
        if isinstance(vals[0], torch.Tensor):
            try:
                batched[k] = torch.stack(vals, dim=0)
            except RuntimeError:
                # Mismatched shapes (shouldn't happen with fixed window_len)
                batched[k] = vals[0]
        else:
            batched[k] = vals[0]

    # Stack targets
    if isinstance(targets[0], torch.Tensor):
        batched_targets = torch.stack(list(targets), dim=0)
    elif isinstance(targets[0], dict):
        # Multi-task: stack each task tensor
        batched_targets = {
            k: torch.stack([t[k] for t in targets], dim=0)
            for k in targets[0].keys()
        }
    else:
        batched_targets = torch.tensor(targets)

    return batched, batched_targets


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic dataset builder (for testing, no real data required)
# ──────────────────────────────────────────────────────────────────────────────

def make_synthetic_scene_cache(
    num_scenes: int = 4,
    frames_per_scene: int = 50,
    num_features: int = 101,
    num_sensor_groups: int = 5,
    token_dim: int = 48,
    window_len: int = 10,
) -> SceneFeatureCache:
    """Create a SceneFeatureCache populated with synthetic numpy arrays.

    This function does NOT call Module 1/2/3. It directly populates the
    internal cache with deterministic synthetic data for unit testing.

    Args:
        num_scenes: Number of synthetic scenes.
        frames_per_scene: Length of each scene timeline.
        num_features: Module 3 feature vector dimension.
        num_sensor_groups: S — number of sensor groups.
        token_dim: D — token feature dimension.
        window_len: Used to validate that windows can be extracted.

    Returns:
        A SceneFeatureCache with populated _cache (pipe2/pipe3 are None).
    """
    # Create a stub cache without real pipelines
    cache = SceneFeatureCache.__new__(SceneFeatureCache)
    cache.pipe2 = None
    cache.pipe3 = None
    cache.channel_names = []
    cache.cache_dir = None
    cache._cache = {}

    rng = np.random.RandomState(42)
    for i in range(num_scenes):
        scene_id = f"synthetic_scene_{i:03d}"
        T = frames_per_scene
        cache._cache[scene_id] = {
            "tokens": rng.randn(T, num_sensor_groups, token_dim).astype(np.float32),
            "token_mask": np.ones((T, num_sensor_groups, token_dim), dtype=bool),
            "features": rng.randn(T, num_features).astype(np.float32),
            "timestamps": np.linspace(0, T * 0.1, T, dtype=np.float64),
            "sensor_groups": [f"group_{g}" for g in range(num_sensor_groups)],
        }

    return cache
