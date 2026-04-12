# Proposed Approach

## Overview

Our approach proceeds in two stages. In the first stage, presented here as our primary midterm contribution, we train a multimodal classifier to predict grip type from pre-contact egocentric video and hand pose trajectories. This classification task serves as an empirical probe: if pre-contact signals cannot predict even a coarse discrete summary of contact outcome, they cannot support a full pressure map prediction model. In the second stage, which we outline as our next steps, we extend this framework to a conditional diffusion model that generates full-hand tactile pressure maps from the same pre-contact conditioning signals.

## Stage 1: Pre-Contact Intent Classification

### Input Representation

Given a clip with contact onset at frame $\tau$, we extract the pre-contact observation window $\{1, \ldots, \tau - 1\}$ and construct two input streams:

**Visual stream**: We sample up to $L = 20$ frames from the pre-contact window, resizing each to $224 \times 224$ pixels. Frames are encoded using a frozen DINOv3 ViT-B/16 backbone [CITE], which processes each frame independently and extracts the CLS token embedding $h_t \in \mathbb{R}^{768}$. The sequence of per-frame embeddings is temporally aggregated via mean pooling:

$$\bar{h}_V = \frac{1}{L} \sum_{t=1}^{L} h_t$$

and projected to a $d = 64$ dimensional embedding via a learned linear layer:

$$z_V = W_V \bar{h}_V + b_V, \quad z_V \in \mathbb{R}^{64}$$

**Pose stream**: Each frame's hand pose $p_t \in \mathbb{R}^{21 \times 3}$ is flattened to $\mathbb{R}^{63}$ and the sequence is processed by a 4-layer MLP with BatchNorm, GELU activations, and dropout ($p = 0.1$). The MLP operates on the temporally pooled pose sequence and projects to the same $d = 64$ dimensional space:

$$z_P = \text{MLP}_P(p_{1:\tau-1}), \quad z_P \in \mathbb{R}^{64}$$

### Fusion and Classification

When both modalities are present, the embeddings are concatenated and fused via a learned linear projection:

$$z = W_f [z_V \| z_P] + b_f, \quad z \in \mathbb{R}^{64}$$

where $[\cdot \| \cdot]$ denotes concatenation. The fused representation is passed through a linear classification head:

$$\hat{y} = \text{softmax}(W_c z + b_c), \quad \hat{y} \in \mathbb{R}^{|\mathcal{Y}|}$$

where $|\mathcal{Y}| = 28$ grip type classes. The model is optimized end-to-end using cross-entropy loss $\mathcal{L}_{\text{cls}}$ as defined in the problem statement, with the DINOv3 backbone kept frozen throughout training.

### Modality Ablations

To quantify the contribution of each input signal, we train three configurations of the model:

**Visual only** ($f_\theta^V$): only the visual stream is active; $z = z_V$ and the fusion layer is replaced by an identity mapping.

**Visual + pose** ($f_\theta^{VP}$): both streams are active as described above. This is our core pre-contact intent prediction baseline — it uses only signals available before contact occurs.

**Visual + pose + tactile** ($f_\theta^{VPT}$): a third tactile encoder, symmetric in architecture to the pose encoder, processes the flattened $16 \times 16$ pressure map at peak contact. This configuration serves as an upper bound, as it uses the tactile signal we ultimately aim to predict.

The accuracy gap $\Delta_{VP} = \text{acc}(f_\theta^{VP}) - \text{acc}(f_\theta^V)$ quantifies the additional predictive value of hand trajectory over vision alone. The gap $\Delta_{VPT} = \text{acc}(f_\theta^{VPT}) - \text{acc}(f_\theta^{VP})$ represents the headroom our generative model aims to close.

## Stage 2: Conditional Diffusion Model for Tactile Prediction (Planned)

The classification results motivate and inform a full generative model. Rather than predicting a discrete grip label, the generative model $g_\phi$ directly synthesizes the peak tactile pressure map $\hat{\mathbf{M}} \in \mathbb{R}^{16 \times 16}$ conditioned on the same pre-contact signals:

$$g_\phi : (\mathbf{V}_{1:\tau-1},\ \mathbf{P}_{1:\tau-1}) \rightarrow \hat{\mathbf{M}}$$

We plan to adopt a **diffusion forcing** backbone [CITE], which extends standard denoising diffusion to variable-length temporal sequences and enables flexible conditioning on partial observations. The forward process gradually corrupts the target pressure map $\mathbf{M}$ with Gaussian noise:

$$q(\mathbf{M}_k \mid \mathbf{M}_0) = \mathcal{N}(\mathbf{M}_k;\ \sqrt{\bar{\alpha}_k} \mathbf{M}_0,\ (1 - \bar{\alpha}_k)\mathbf{I})$$

where $k$ indexes the diffusion timestep and $\bar{\alpha}_k$ follows a cosine noise schedule. The reverse process learns to denoise conditioned on the pre-contact visual and pose embeddings:

$$p_\phi(\mathbf{M}_{k-1} \mid \mathbf{M}_k,\ z_V,\ z_P) = \mathcal{N}(\mathbf{M}_{k-1};\ \mu_\phi(\mathbf{M}_k, k, z_V, z_P),\ \sigma_k^2 \mathbf{I})$$

The conditioning embeddings $z_V$ and $z_P$ are injected into the denoising network via cross-attention at each diffusion timestep, following the architecture of latent diffusion models [CITE]. The model is trained to minimize the simplified diffusion objective:

$$\mathcal{L}_{\text{diff}} = \mathbb{E}_{k, \mathbf{M}_0, \epsilon}\left[\|\epsilon - \epsilon_\phi(\mathbf{M}_k, k, z_V, z_P)\|_2^2\right]$$

where $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ is the noise added at step $k$ and $\epsilon_\phi$ is the learned denoising network.

The classification results from Stage 1 directly inform Stage 2 in two ways. First, the accuracy of $f_\theta^{VP}$ establishes whether pre-contact signals carry sufficient information to warrant a generative model — if grip type is not predictable from pre-contact cues, neither is the pressure map. Second, the magnitude of $\Delta_{VPT}$ determines how much the generative model can potentially improve over the pre-contact baseline by synthesizing an approximation of the missing tactile signal.

## Connection to Complementary Experiments

In parallel with the classification experiments, two complementary baselines provide important context for the broader prediction task.

**Dense contact field estimation** (Edward Chen): A multimodal inpainting network (`PressureInpaintNet`) reconstructs the full $16 \times 16$ pressure map from sparse fingertip-only sensor readings, conditioned on egocentric RGB and hand landmarks. The architecture uses a U-Net style encoder-decoder with skip connections, injecting visual and pose features at the bottleneck via spatial conditioning. The model is trained with a composite loss combining MSE on the full map and a weighted MAE term specifically on unobserved (non-fingertip) cells:

$$\mathcal{L}_{\text{inpaint}} = \text{MSE}(\hat{\mathbf{M}}, \mathbf{M}) + \lambda \cdot \frac{\sum_{(i,j) \notin \mathcal{F}} |\hat{M}_{ij} - M_{ij}|}{|\{(i,j) \notin \mathcal{F}\}|}$$

where $\mathcal{F}$ is the set of observed fingertip cell locations. This directly addresses Direction A (dense contact field estimation from sparse sensing) and is complementary to the classification baseline — while our work predicts grip type from pre-contact signals, this work reconstructs the spatial pressure distribution from partial contact observations.

**Egocentric video prediction** (Akshata Tiwari): A next-frame prediction model trained on OpenTouch RGB sequences establishes that future egocentric frames are predictable from prior frames, validating the temporal structure of the visual signal and providing a unimodal reference point for the cross-modal generation task.