#!/bin/bash
# Linux GPU environment for SpatialLM 1.1 and Sonata (Pointcept) on Volumet's WSL Ubuntu, RTX 5060 Ti (sm_120).
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq build-essential git wget curl libgl1 libglib2.0-0 libgomp1 > /dev/null
mkdir -p /opt/work && cd /opt/work
if [ ! -d /opt/conda ]; then
  wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O mf.sh
  bash mf.sh -b -p /opt/conda
fi
source /opt/conda/etc/profile.d/conda.sh
conda create -y -q -n slm python=3.11 > /dev/null || true
conda activate slm
pip install -q torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -q "transformers==4.46.1" "tokenizers<0.20.4" safetensors pandas einops "numpy<2" scipy scikit-learn toml \
    "huggingface_hub<0.30" shapely bbox terminaltables open3d addict timm laspy lazrs plyfile
pip install -q torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
pip install -q spconv-cu126
[ -d SpatialLM ] || git clone -q https://github.com/manycore-research/SpatialLM.git
[ -d sonata ] || git clone -q https://github.com/facebookresearch/sonata.git
cd SpatialLM && pip install -q --no-deps -e . && cd ..
export HF_HOME=/opt/work/hf
huggingface-cli download manycore-research/SpatialLM1.1-Qwen-0.5B --local-dir models/SpatialLM1.1-Qwen-0.5B > /dev/null
huggingface-cli download ysmao/SpatialLM1.1-Qwen-0.5B-ARKitScenes-SFT --local-dir models/SpatialLM1.1-Qwen-0.5B-ARKitScenes-SFT > /dev/null
huggingface-cli download manycore-research/SpatialLM-Testset pcd/scene0000_00.ply --repo-type dataset --local-dir testset > /dev/null
python -c "import torch,spconv,torch_scatter;print('torch',torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0));import spconv.pytorch as sp;print('spconv ok')"
echo SETUP_DONE
