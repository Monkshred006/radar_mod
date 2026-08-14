#!/usr/bin/env bash
set -e

echo "================================================================"
echo " PhotonShield AI (PhotonV0) — Training & Evaluation on RaDICaL"
echo "================================================================"

# 1. Train PhotonV0 on RaDICaL dataset (20 Epochs)
python train_photon_v0.py --config configs/photon_v0.yaml

# 2. Evaluate Best Checkpoint
python evaluate_photon_v0.py --checkpoint checkpoints/photon_v0/best_model.pt --config configs/photon_v0.yaml

# 3. Export ONNX and INT8 Deployment Artifacts
mkdir -p artifacts
python -m module_06_bitnet.export_onnx --model-path checkpoints/photon_v0/best_model.pt --output artifacts/photon_v0.onnx
python -m module_06_bitnet.quantize_int8 --model-path checkpoints/photon_v0/best_model.pt --output artifacts/photon_v0_int8.pt

echo "================================================================"
echo " Training, Evaluation, and Export Workflow Complete!"
echo "================================================================"
