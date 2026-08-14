"""Tests for pipeline.py — including offline, streaming causality, and full integration."""
import numpy as np
import torch
import pytest

from module_02_sensor_dsp.pipeline import SensorDSPPipeline
from module_03_sensor_fusion.pipeline import SensorFusionPipeline
from module_03_sensor_fusion.config import Module3Config


def make_module2_processed_output(T=30):
    ts = np.arange(T) * 0.1
    return {
        "signals": {
            "photodiode_1": np.sin(ts),
            "photodiode_2": np.cos(ts),
            "temperature": np.full(T, 24.5),
            "humidity": np.full(T, 55.0),
            "pressure": np.full(T, 1012.0),
            "accel_x": np.ones(T) * 0.1,
            "accel_y": np.ones(T) * 0.2,
            "accel_z": np.ones(T) * 9.8,
            "gyro_x": np.zeros(T),
            "gyro_y": np.zeros(T),
            "gyro_z": np.zeros(T),
            "distance": np.full(T, 150.0),
        },
        "timestamps": ts,
        "validity": {
            "outlier_masks": {k: np.zeros(T, dtype=bool) for k in ["photodiode_1", "distance"]},
            "missing_masks": {k: np.zeros(T, dtype=bool) for k in ["photodiode_1", "distance"]},
            "interpolated_masks": {k: np.zeros(T, dtype=bool) for k in ["photodiode_1", "distance"]},
        },
        "quality": {},
        "preprocessing_metadata": {},
    }


class TestPipelineOffline:
    def test_offline_output_structure(self):
        m2_out = make_module2_processed_output()
        pipe = SensorFusionPipeline()
        res = pipe.process_offline(m2_out)

        assert "features" in res
        assert "tokens" in res
        assert "token_mask" in res
        assert "timestamps" in res
        assert "sensor_groups" in res
        assert "feature_names" in res

        # T = 30 preserved
        assert res["features"].shape[0] == 30
        assert res["tokens"].shape[0] == 30

        # Tokens: [T, S, D_max]
        assert res["tokens"].ndim == 3
        assert res["tokens"].shape[1] == len(res["sensor_groups"])

    def test_tensor_dtype_float32(self):
        m2_out = make_module2_processed_output()
        pipe = SensorFusionPipeline(Module3Config(dtype="float32"))
        res = pipe.process_offline(m2_out)
        assert res["features"].dtype == torch.float32
        assert res["tokens"].dtype == torch.float32


class TestStreamingCausalityEquivalence:
    """Verify offline vs streaming produce equivalent causal features."""

    def test_streaming_matches_offline_causal(self):
        config = Module3Config(streaming=True)
        pipe = SensorFusionPipeline(config)

        T = 15
        m2_out = make_module2_processed_output(T=T)

        # 1. Offline processing
        res_offline = pipe.process_offline(m2_out)
        offline_tokens = res_offline["tokens"]

        # 2. Streaming sample-by-sample processing
        state = pipe.make_stream_state()
        stream_tokens_list = []

        for i in range(T):
            m2_sample = {
                "signals": {k: float(v[i]) for k, v in m2_out["signals"].items()},
                "timestamps": float(m2_out["timestamps"][i]),
                "validity": {
                    "outlier_flags": {k: False for k in m2_out["signals"]},
                    "missing_flags": {k: False for k in m2_out["signals"]},
                    "interpolated_flags": {k: False for k in m2_out["signals"]},
                },
                "quality": None,
            }
            s_res, state = pipe.process_stream(m2_sample, state)
            stream_tokens_list.append(s_res["tokens"])  # [1, S, D_max]

        stream_tokens_cat = torch.cat(stream_tokens_list, dim=0)  # [T, S, D_max]

        # Check shape equivalence
        assert stream_tokens_cat.shape == offline_tokens.shape

        # Values for latest samples must be close
        np.testing.assert_allclose(
            stream_tokens_cat.numpy(),
            offline_tokens.numpy(),
            rtol=1e-4,
            atol=1e-4
        )
