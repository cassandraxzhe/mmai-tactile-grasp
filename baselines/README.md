# Baselines

This directory contains baseline models developed as part of the 6.S985 course project. Each baseline addresses a distinct component of the broader cross-modal tactile prediction problem.

---

## Dense Contact Field Estimation (`pressure_inpainting/`)

**Author:** Edward Chen

A multimodal inpainting model (`PressureInpaintNet`) that reconstructs the full 16×16 hand pressure map from sparse fingertip-only sensor readings, conditioned on egocentric RGB video and 3D hand landmarks.

**Architecture:**
- Lightweight CNN visual encoder (conv stack → adaptive pooling → 128-dim embedding)
- Pose MLP encoder (63-dim flattened landmarks → 64-dim embedding)
- U-Net style encoder-decoder with skip connections for the pressure map
- Visual and pose features injected at the bottleneck via spatial conditioning

**Loss function:**
A composite loss that combines MSE on the full pressure map with a weighted MAE term specifically penalizing errors in unobserved (non-fingertip) cells:

```
L = MSE(pred, gt) + 0.5 * MAE(pred[unobserved], gt[unobserved])
```

**Training:**
- Dataset: OpenTouch (all HDF5 files, frames with contact only)
- 80/20 train/val split by file
- 50 epochs, Adam optimizer, lr=1e-4, cosine annealing
- Logged to WandB with visual comparisons every 5 epochs

**Relationship to project:**
This baseline addresses dense contact field estimation from sparse sensing. While the classification experiments (see `experiments/direction_B/`) ask *which grip type* will form from pre-contact signals, this model asks whether partial contact observations can recover the full spatial pressure distribution. Together they bracket the full prediction problem.

**To run:**
```bash
# Install dependencies
pip install torch torchvision wandb h5py Pillow plotly

# Set your WandB API key in pressure_impaint.py, then:
python baselines/pressure_inpainting/pressure_impaint.py
```

Note: update the `files` glob path in the script to point to your local OpenTouch HDF5 files.

---

## Pre-Contact Intent Classification (`direction_B/`)

**Author:** Cassandra He

Multimodal grip type classifier trained on pre-contact egocentric video and hand pose trajectories. Three modality configurations are evaluated to quantify the predictive value of each signal.

See `experiments/direction_B/results_summary.md` for full results, and `notebooks/opentouch_colab_training.ipynb` for the training pipeline.

**Quick results:**

| Modalities | Test Accuracy | Macro F1 |
|-----------|:-------------:|:--------:|
| Visual only | 31.9% | 0.169 |
| Visual + pose | 30.0% | 0.135 |
| Visual + pose + tactile | 28.9% | 0.144 |

*28 classes, random chance = 3.6%*