#!/bin/bash
#SBATCH --job-name=world_model
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=gpu-long          # check: sinfo | grep gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

mkdir -p logs

module load anaconda/2023a-tensorflow     # adjust to whatever has torch
# or: source activate your_env

pip install -q diffusers accelerate h5py --upgrade

python train.py \
    --data_root /path/to/opentouch/data \
    --out checkpoints
