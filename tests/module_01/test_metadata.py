"""Tests for metadata module."""

from module_01_radar_input.metadata import RadarMetadata


def test_metadata_instantiation():
    meta = RadarMetadata(
        radar_type="FMCW",
        sampling_rate=1e6,
        num_antennas=4,
        frame_dimensions=(128, 64),
        scene_id="scene_01"
    )
    assert meta.radar_type == "FMCW"
    assert meta.frame_rate is None  # Unavailable fields default to None

    d = meta.to_dict()
    assert d["sampling_rate"] == 1e6
    assert d["frame_rate"] is None

    restored = RadarMetadata.from_dict(d)
    assert restored.scene_id == "scene_01"
    assert restored.num_antennas == 4
