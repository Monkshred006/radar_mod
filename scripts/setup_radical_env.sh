#!/usr/bin/env bash
set -e

echo "================================================================"
echo " PhotonShield AI — Setting up RaDICaL Environment"
echo "================================================================"

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip

pip install torch torchvision torchaudio
pip install mamba-ssm || echo "mamba-ssm native CUDA package not available on this platform, using pure PyTorch fallback backend."
pip install numpy scipy pandas scikit-learn matplotlib tqdm pyyaml
pip install h5py opencv-python onnx onnxruntime

python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
echo "Environment setup complete."
