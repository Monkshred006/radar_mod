"""Tests for adapters and custom dataset mapping."""

from pathlib import Path
import pytest
from module_01_radar_input.adapters.base import DefaultDirectoryAdapter


def test_default_directory_adapter_discovery(tmp_path):
    # Setup mock scene directories
    scene1 = tmp_path / "scene_01"
    scene1.mkdir()
    (scene1 / "f1.npy").touch()
    (scene1 / "f2.npy").touch()

    scene2 = tmp_path / "scene_02"
    scene2.mkdir()
    (scene2 / "f1.npy").touch()

    adapter = DefaultDirectoryAdapter()
    items = adapter.discover_items(tmp_path)

    assert len(items) == 3
    scenes = [it["scene_id"] for it in items]
    assert scenes == ["scene_01", "scene_01", "scene_02"]
