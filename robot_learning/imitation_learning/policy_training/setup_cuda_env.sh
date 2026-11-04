#!/bin/bash
# setup_cuda_env.sh — Python env for train.py on a CUDA host (π0.5 needs LeRobot's transformers fork).
# CUDA 12.8 wheels (needed for Blackwell sm_120 GPUs; fine on Ampere/Hopper). Run once:  bash setup_cuda_env.sh
# ENV_NAME (default policy-train) names the conda env; CONDA_BASE defaults to `conda info --base`.
set -e
ENV_NAME=${ENV_NAME:-policy-train}
CONDA_BASE=${CONDA_BASE:-$(conda info --base)}
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda create -y -q -n "$ENV_NAME" python=3.12
conda activate "$ENV_NAME"
pip install -q torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install -q "lerobot==0.4.4" sentencepiece bitsandbytes matplotlib opencv-python-headless av pyarrow
pip install -q "transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi"
conda install -y -q -c conda-forge ffmpeg
python -c "import torch, lerobot; from transformers.models.siglip import check; \
print(torch.__version__, torch.cuda.get_device_name(0), lerobot.__version__, check.check_whether_transformers_replace_is_installed_correctly())"
echo "export MABEL_LEROBOT_PYTHON=$CONDA_BASE/envs/$ENV_NAME/bin/python"
