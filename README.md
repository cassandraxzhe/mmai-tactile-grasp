# Multimodal Predictive Models for Tactile Manipulation
### 6.S985 Modeling Multimodal AI — Spring 2026

**Team:** Cassandra He · Liane Xu · Edward Chen · Akshata Tiwari

---

## Overview

Humans anticipate the tactile consequences of their actions before contact occurs — they know what picking up a coffee cup will feel like before their fingers touch it. This predictive ability enables fluid manipulation, yet in robotics, tactile information enters the pipeline only after contact as reactive correction.

We propose a **cross-modal predictive world model** that generates anticipated full-hand tactile pressure maps from pre-contact egocentric video and 3D hand pose trajectories. Our work is grounded in the [OpenTouch dataset](https://github.com/OpenTouch-MIT/opentouch), which provides synchronized egocentric RGB, full-hand tactile pressure maps, hand pose, eye tracking, and head motion data across 5.1 hours of naturalistic human interaction.

---

## Research Directions

| Direction | Description | Status |
|-----------|-------------|--------|
| **A** | Dense contact field estimation from sparse tactile sensing | Planned |
| **B** | Pre-contact intent prediction from gaze and trajectory | 🔄 In progress |
| **C** | Tactile sensor domain alignment for hardware-agnostic transfer | Planned |
| **D** | Physics-informed hidden state refinement | Planned |

Our midterm focus is **Direction B**: predicting grasp type from pre-contact visual and hand trajectory signals, before contact occurs.

---

## Key Findings (Midterm)

- **Pre-contact signals are informative**: fingertip spread decreases monotonically in the ~15 frames before contact, reflecting the hand closing into its grip preshape. Different grip types show distinct preshaping patterns.
- **Vision alone is insufficient**: fine-tuning on EgoDex (HW3) showed that center-frame visual features cannot capture action-defining motion — motivating the multimodal approach.
- **Dataset**: 2,958 annotated clips with `onset_idx`, `peak_idx`, and `post_idx` timing annotations. 1,344 clips have meaningful pre-contact windows (≥5 frames before contact onset).

---

## Repository Structure

```
mmai-tactile-grasp/
├── README.md
├── assignments/
│   ├── hw1/
│   ├── hw2/
│   └── hw3/                        # EgoDex fine-tuning (Qwen2.5-VL + LoRA)
├── notebooks/
│   └── opentouch_exploration.ipynb # OpenTouch data exploration (Direction B)
├── experiments/
│   └── direction_B/                # Classification ablation results
├── data/
│   └── splits/                     # train/val/test CSVs
└── reports/
    ├── proposal/
    └── midterm/
```

---

## Setup

### 1. Clone and download data

```bash
git clone https://github.com/OpenTouch-MIT/opentouch
cd opentouch
pip install gdown
bash scripts/download_data.sh
cd data && unzip final_annotations.zip && cd ..
```

### 2. Create environment

```bash
conda create -n opentouch python=3.10
conda activate opentouch
pip install -e .
pip install h5py numpy pandas matplotlib seaborn tqdm scikit-learn opencv-python Pillow jupyter
```

### 3. Run data exploration notebook

```bash
jupyter notebook notebooks/opentouch_exploration.ipynb
```

---

## Experiments

### Grip type classification ablations (Direction B)

We run three modality configurations to quantify the value of pre-contact trajectory signals:

| Run | Modalities | Purpose |
|-----|-----------|---------|
| 1 | `visual` | Vision-only baseline |
| 2 | `visual pose` | Direction B core — pre-contact only |
| 3 | `visual pose tactile` | Upper bound with contact signal |

```bash
# Preprocess
python build_label_data.py \
    --input-dir data \
    --output-dir preprocessed_data/classification_peak \
    --label-mapping-path data/final_annotations \
    --label-column grip_type \
    --frame-index-column peak_idx \
    --temporal-radius 10

# Train (replace --modalities for each run)
python -m opentouch_train.classification_main \
    --train-data preprocessed_data/classification_peak \
    --model OpenTouch-DINOv3-B16-Classify \
    --task grip --modalities visual pose \
    --batch-size 64 --lr 3e-3 --epochs 100
```

Results will be posted to `experiments/direction_B/` as they complete.

---

## Links

- 📄 [Project Proposal](reports/proposal/)
- 📓 [Data Exploration Notebook](notebooks/opentouch_exploration.ipynb)
- 🗂️ [OpenTouch Dataset](https://github.com/OpenTouch-MIT/opentouch)
- 📋 [Project Issues](../../issues)

---

## Team & Contributions

| Member | Focus |
|--------|-------|
| Cassandra He | Data exploration, experiments, EgoDex baseline |
| Liane Xu | Related work, robotics hardware expertise, downstream real-robot evaluation |
| Edward Chen | OpenTouch visualizations, unimodal tactile next-frame prediction |
| Akshata Tiwari | Egocentric video next-frame prediction, diffusion/world model backbone |

*Contributions will be updated after team sync on April 5, 2026.*