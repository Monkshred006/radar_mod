"""Tests for radar loader module."""

import pytest
import numpy as np
import pandas as pd
import json
from module_01_radar_input.radar_loader import RadarLoader
from module_01_radar_input.validation import CorruptedFileError


def test_load_npy(tmp_path):
    fpath = tmp_path / "frame.npy"
    expected = np.random.rand(32, 16).astype(np.float32)
    np.save(fpath, expected)

    loader = RadarLoader()
    loaded = loader.load_frame(fpath)

    assert loaded.dtype == np.float32
    assert loaded.shape == (32, 16)
    np.testing.assert_array_equal(loaded, expected)


def test_load_complex_npy(tmp_path):
    fpath = tmp_path / "complex_frame.npy"
    expected = (np.random.rand(16, 8) + 1j * np.random.rand(16, 8)).astype(np.complex64)
    np.save(fpath, expected)

    loader = RadarLoader()
    loaded = loader.load_frame(fpath)

    assert loaded.dtype == np.complex64
    np.testing.assert_array_equal(loaded, expected)


def test_load_npz(tmp_path):
    fpath = tmp_path / "frame.npz"
    expected = np.random.randint(0, 100, size=(10, 10), dtype=np.int32)
    np.savez(fpath, radar_raw=expected)

    loader = RadarLoader(npz_key="radar_raw")
    loaded = loader.load_frame(fpath)
    assert loaded.dtype == np.int32
    np.testing.assert_array_equal(loaded, expected)


def test_missing_file(tmp_path):
    loader = RadarLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_frame(tmp_path / "non_existent.npy")
