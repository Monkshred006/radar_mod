"""End-to-End integration and causality tests for Module 9."""

import numpy as np
import pytest
import torch

from module_04_mamba_hybrid.config import MambaHybridConfig, TaskHeadConfig
from module_04_mamba_hybrid.engine import PhotonMambaHybrid
from module_07_decision.config import DecisionConfig, DecisionModelConfig
from module_07_decision.inference import PhotonShieldDecisionPipeline
from module_07_decision.multitask import PhotonShieldMultiTask
from module_08_pinn_rl.config import RLStateConfig
from module_08_pinn_rl.state import RLStateBuilder
from module_09_3d.config import Module9Config, OLEDConfig, RotationConfig
from module_09_3d.inference import PhotonShield3DPipeline
from module_09_3d.interfaces import SceneInput
from module_09_3d.oled import SimulatedDisplayBackend
from module_09_3d.point_cloud import PointCloud
from module_09_3d.scene import Scene3D


class TestEndToEndModule09:
    def test_full_pipeline_from_module4_through_module9(self):
        """Verify: Module 4 Engine -> Module 7 Decisions -> Module 8 RLState -> Module 9 3D Pipeline."""
        # 1. Module 4
        mamba_cfg = MambaHybridConfig(
            d_model=128,
            num_layers=1,
            num_attention_heads=2,
            head_config=TaskHeadConfig(head_type="classification"),
        )
        engine = PhotonMambaHybrid(mamba_cfg)
        engine.eval()

        m3_out = {
            "tokens": torch.randn(1, 4, 5, 48),
            "token_mask": torch.ones(1, 4, 5, 48, dtype=torch.bool),
            "features": torch.randn(1, 4, 101),
            "timestamps": torch.arange(4).float().unsqueeze(0),
        }
        with torch.no_grad():
            eng_out = engine(m3_out)
        pooled = eng_out["pooled_output"]  # [1, 128]

        # 2. Module 7
        dec_m_cfg = DecisionModelConfig(d_model=128)
        multi_task = PhotonShieldMultiTask(dec_m_cfg)
        pipeline_dec = PhotonShieldDecisionPipeline(multi_task, DecisionConfig())
        dec_outputs = pipeline_dec.predict_pooled(pooled)
        dec_0 = dec_outputs[0]

        # 3. Module 8
        rl_builder = RLStateBuilder(RLStateConfig(mamba_latent_dim=128))
        rl_state = rl_builder.build_from_decision_output(pooled[0], dec_0)

        # 4. Module 9 3D Pipeline
        cfg = Module9Config()
        cfg.rotation.rotation_step_deg = 45.0  # 8 frames
        cfg.camera.image_width = 128
        cfg.camera.image_height = 128
        pipeline_3d = PhotonShield3DPipeline(cfg)

        scene_input = SceneInput(
            latent=rl_state.components["mamba_latent"],
            target_probability=dec_0.target_probability,
            anomaly_probability=dec_0.anomaly_probability,
            environmental_assessment=dec_0.environmental_assessment,
            timestamp=1.0,
        )

        scene, frames = pipeline_3d.reconstruct_and_render(scene_input)
        assert scene.point_cloud.num_points > 0
        assert len(frames) == 8
        assert frames[0].shape == (128, 128, 3)

        # 5. Stream to simulated OLED
        disp = SimulatedDisplayBackend(OLEDConfig(display_width=128, display_height=128))
        count = pipeline_3d.stream_frames_to_display(frames, display=disp)
        assert count == 8
        assert disp.current_frame is not None

        # 6. Prism canvas rendering
        prism_canvas = pipeline_3d.render_prism_view(scene)
        assert prism_canvas.shape == (512, 512, 3)

    def test_causal_streaming_invariance(self):
        """Requirement #26: Frame at timestep t is invariant to future scenes t+1, t+2."""
        pipeline = PhotonShield3DPipeline()

        # Timestep 1
        pc1 = PointCloud(points=np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
        scene1 = Scene3D(point_cloud=pc1, timestamp=1.0)
        frames_t1 = pipeline.render_rotating_views(scene1)

        # Timestep 2 (Future scene)
        pc2 = PointCloud(points=np.array([[5.0, 5.0, 5.0]], dtype=np.float32))
        scene2 = Scene3D(point_cloud=pc2, timestamp=2.0)
        _ = pipeline.render_rotating_views(scene2)

        # Re-render timestep 1; must be bit-for-bit identical
        frames_t1_repeat = pipeline.render_rotating_views(scene1)
        for f1, f2 in zip(frames_t1, frames_t1_repeat):
            assert np.array_equal(f1, f2)
