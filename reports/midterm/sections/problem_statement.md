# Problem Statement

## Notation and Setup

We formalize the pre-contact tactile prediction problem as follows. Let $\mathbf{V} = \{v_1, \ldots, v_T\}$ denote a sequence of $T$ egocentric RGB frames captured from a head-mounted camera, and let $\mathbf{P} = \{p_1, \ldots, p_T\}$ denote the corresponding 3D hand pose trajectories, where each $p_t \in \mathbb{R}^{21 \times 3}$ represents the positions of 21 hand landmarks at frame $t$ in camera coordinates. We define contact onset as the frame index $\tau$ at which tactile pressure first exceeds a threshold, annotated in the dataset as `onset_idx`. The pre-contact observation window is thus $\{1, \ldots, \tau - 1\}$.

Let $\mathbf{M} \in \mathbb{R}^{16 \times 16}$ denote the full-hand tactile pressure map at peak contact (frame $\tau^*$, where $\tau^* > \tau$, annotated as `peak_idx`), and let $y \in \mathcal{Y}$ denote the discrete grip type label, where $|\mathcal{Y}| = 28$ in the OpenTouch dataset.

## Pre-Contact Intent Prediction (Classification)

We address the following classification objective: given only pre-contact visual and pose observations, predict the grip type the hand will form at contact:

$$f_\theta : (\mathbf{V}_{1:\tau-1},\ \mathbf{P}_{1:\tau-1}) \rightarrow \hat{y} \in \mathcal{Y}$$

where $f_\theta$ is a multimodal classifier parameterized by $\theta$, consisting of a frozen DINOv3 visual encoder, a learned pose encoder, a fusion module, and a linear classification head. We optimize $f_\theta$ using cross-entropy loss over the grip type labels:

$$\mathcal{L}_{\text{cls}} = -\sum_{c=1}^{|\mathcal{Y}|} \mathbb{1}[y = c] \log f_\theta(\mathbf{V}_{1:\tau-1},\ \mathbf{P}_{1:\tau-1})_c$$

The classification task serves as an empirical validation that pre-contact signals are predictive of contact outcome — a necessary precondition for the full generative objective described below.

## Full Generative Objective (Longer-Term Goal)

The broader goal of this project is to predict the peak tactile pressure map from pre-contact signals alone, without requiring any physical contact at inference time:

$$g_\phi : (\mathbf{V}_{1:\tau-1},\ \mathbf{P}_{1:\tau-1}) \rightarrow \hat{\mathbf{M}} \in \mathbb{R}^{16 \times 16}$$

where $g_\phi$ is a conditional diffusion model parameterized by $\phi$. We optimize $g_\phi$ via a combination of pixel-level reconstruction loss and structural similarity:

$$\mathcal{L}_{\text{gen}} = \|\mathbf{M} - \hat{\mathbf{M}}\|_2^2 + \lambda \cdot \mathcal{L}_{\text{SSIM}}(\mathbf{M},\ \hat{\mathbf{M}})$$

where $\lambda$ balances the two terms and $\mathcal{L}_{\text{SSIM}}$ is the structural similarity index measure loss, which captures perceptual differences in pressure map structure beyond pixel-level error.

## Relationship Between Objectives

The classification objective $\mathcal{L}_{\text{cls}}$ and the generative objective $\mathcal{L}_{\text{gen}}$ are related through the grip type label $y$: grip type is a coarse discrete summary of the pressure map $\mathbf{M}$, capturing the overall hand configuration at contact. A model that can reliably predict $y$ from pre-contact signals has learned a representation of the pre-contact window that encodes intent — the same representation that the generative model $g_\phi$ will need to condition on. The accuracy gap between the vision-only baseline $f_\theta(\mathbf{V}_{1:\tau-1})$ and the full pre-contact model $f_\theta(\mathbf{V}_{1:\tau-1}, \mathbf{P}_{1:\tau-1})$ quantifies the additional predictive value of hand trajectory, motivating its inclusion as a conditioning signal in $g_\phi$.

## Notation Summary

| Symbol | Description |
|--------|-------------|
| $\mathbf{V} = \{v_1, \ldots, v_T\}$ | Egocentric RGB video sequence |
| $\mathbf{P} = \{p_1, \ldots, p_T\}$ | Hand pose trajectory, $p_t \in \mathbb{R}^{21 \times 3}$ |
| $\tau$ | Contact onset frame (`onset_idx`) |
| $\tau^*$ | Peak pressure frame (`peak_idx`) |
| $\mathbf{M} \in \mathbb{R}^{16 \times 16}$ | Full-hand tactile pressure map at peak contact |
| $y \in \mathcal{Y}$ | Grip type label, $\|\mathcal{Y}\| = 28$ |
| $f_\theta$ | Multimodal intent classifier |
| $g_\phi$ | Conditional diffusion model for tactile prediction |
| $\mathcal{L}_{\text{cls}}$ | Cross-entropy classification loss |
| $\mathcal{L}_{\text{gen}}$ | Generative reconstruction loss |