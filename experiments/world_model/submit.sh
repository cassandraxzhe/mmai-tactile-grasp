#!/bin/bash
#SBATCH --job-name=world_model
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

set -euo pipefail

mkdir -p logs

# ── Environment ───────────────────────────────────────────────────────────────
# Adjust the conda env name to whatever you have torch + diffusers installed in
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mech_interp_gpu   # <-- change this to your env name

# Fallback: install deps if not already present
pip install -q diffusers accelerate h5py

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT="/home/akshatat/mmai-tactile-grasp/opentouch/data"
OUT_DIR="checkpoints/world_model_$(date +%Y%m%d_%H%M%S)"

# ── Run ───────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")"

echo "Starting world model training"
echo "  data_root : $DATA_ROOT"
echo "  output    : $OUT_DIR"
echo "  node      : $(hostname)"
echo "  GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

python train.py \
    --data_root "$DATA_ROOT" \
    --out       "$OUT_DIR"

echo "Training complete. Checkpoints saved to $OUT_DIR"
