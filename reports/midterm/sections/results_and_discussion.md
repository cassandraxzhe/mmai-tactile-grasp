# Results and Discussion

## Data Exploration Findings

Before presenting classification results, we highlight two empirical findings from dataset exploration that motivate the broader project.

**Pre-contact window availability.** Of 2,958 annotated clips, only 1,344 (45.4%) have a meaningful pre-contact window (onset\_idx $\geq 5$ frames). The distribution of window lengths is heavily right-skewed: median onset is frame 5, mean is 23.3 frames, and maximum is 683 frames. Three grip types — Sphere 4 Finger, Large Diameter, and Lateral Tripod — have no usable pre-contact windows at all. Even among dominant classes, approximately 45–50% of Tip Pinch, Small Diameter, and Prismatic 2 Finger clips are excluded. This suggests that pre-contact prediction is an inherently harder problem for some grip types than others, and that dataset coverage is a meaningful constraint on what the model can learn.

**Anticipatory grip preshaping.** Figure 1 shows the fingertip spread — the standard deviation of fingertip positions across the five finger tips — in the pre-contact window for a representative clip. Fingertip spread decreases monotonically in the ~15 frames before contact, reflecting the hand closing into its grip preshape before any tactile feedback is received. This is consistent with the neuroscience literature on anticipatory motor control, and provides direct evidence that pre-contact hand trajectories encode grasp intent. Different grip types exhibit distinct preshaping profiles, suggesting that trajectory features carry discriminative information for grip type prediction.

## Classification Ablation Results

Table 1 presents test accuracy and macro F1 for the three modality configurations. All models significantly outperform random chance (3.6% for 28 classes), confirming that pre-contact signals carry meaningful grip type information.

**Table 1: Grip type classification results on the test split (263 samples, 28 classes).**

| Run | Modalities | Test Accuracy | Macro F1 | Train Loss (ep. 29) |
|-----|-----------|:-------------:|:--------:|:-------------------:|
| 1 | Visual only | **31.9%** | **0.169** | 1.041 |
| 2 | Visual + pose | 30.0% | 0.135 | 0.773 |
| 3 | Visual + pose + tactile | 28.9% | 0.144 | 0.758 |
| — | Random chance | 3.6% | — | — |

## Analysis

**Overfitting with additional modalities.** The most striking result is that adding modalities consistently *decreases* test accuracy despite *decreasing* training loss. The visual-only model achieves the best generalization (31.9%) while the full visual+pose+tactile model has the lowest training loss (0.758) but worst test accuracy (28.9%). This is a clear overfitting signature: the frozen DINOv3 visual encoder provides strong, generalizable features out of the box by virtue of large-scale pretraining, while the learned pose and tactile encoders — trained from scratch on 1,979 samples across 28 classes — overfit rather than generalize at 30 epochs.

**The tactile encoder is effectively dormant.** Figure 2 shows the training loss curves for all three runs. The visual+pose and visual+pose+tactile curves are nearly identical throughout training, converging to 0.773 and 0.758 respectively. This near-overlap suggests the tactile encoder contributes almost nothing to the training dynamics, and that the fusion layer learns to largely ignore it. One explanation is that the peak tactile signal, while highly informative in principle, provides redundant information to the visual and pose signals at the resolution of our 64-dimensional embedding space — the model may have found that visual features are already sufficient to minimize training loss at this data scale, leaving no gradient signal to train the tactile encoder effectively.

**Convergence is incomplete.** All three curves in Figure 2 are still decreasing at epoch 29, indicating none of the models have fully converged. This is a significant confound: the visual-only model's advantage may partly reflect that it has fewer parameters to optimize and converges faster within 30 epochs, rather than a fundamental advantage of the visual modality. The multimodal models have more to gain from additional training — the benefit of each added modality likely only becomes visible once the model has seen enough data for cross-modal interactions to emerge. Longer training runs are necessary to draw conclusions about the relative value of each modality at convergence.

**Visual features encode grip-relevant information.** Despite these caveats, the 31.9% visual-only baseline — nearly 9× random chance on a 28-class problem — demonstrates that DINOv3 features extracted from pre-contact egocentric frames carry substantial grip type information. This is consistent with the broader observation that large vision models trained at scale develop rich internal representations that implicitly encode physical world properties well beyond what their training objectives explicitly optimize for — the frozen DINOv3 backbone was never trained on tactile or manipulation data, yet its features are highly predictive of contact outcome. The question of how much additional signal pose and tactile add remains open pending longer training runs.

**Modality complementarity vs. redundancy.** A key assumption underlying naive multimodal fusion is that the shared information between modalities provides a useful training signal — that vision and pose, for instance, offer redundant views of the same underlying grip intent. Our results suggest this assumption may not hold here. Visual features capture object geometry and hand appearance, while pose captures finger configuration independently of appearance. These modalities may carry largely *complementary* rather than redundant information about grip type, which means a simple fusion strategy that treats them symmetrically will struggle to leverage both effectively — each modality's unique signal gets diluted rather than amplified. This motivates a fusion architecture that explicitly models what each modality contributes uniquely, rather than assuming shared information dominates.

## EgoDex Preliminary Experiments

Prior to the OpenTouch experiments, we fine-tuned Qwen2.5-VL-3B with LoRA on EgoDex egocentric manipulation clips for action description. This preliminary study revealed two findings directly relevant to the current work. First, center-frame visual features are fundamentally insufficient for action-defined tasks — twisting a bottle cap, pressing a ball into a tube, and snapping LEGO bricks are defined by motion and force rather than appearance at any single frame, and no prompt engineering strategy recovered this missing information. Second, fine-tuning on 88 samples across 111 classes produced mode collapse — the model defaulted to a single label, demonstrating that data efficiency is a critical constraint for multimodal manipulation models. Both findings motivate the OpenTouch approach: richer pre-contact signals (trajectory rather than single frames) and a larger, more balanced dataset.

## Complementary Results: Dense Contact Field Estimation

In parallel, Edward Chen developed `PressureInpaintNet`, a U-Net inpainting model that reconstructs the full $16 \times 16$ pressure map from sparse fingertip-only sensor readings, conditioned on egocentric RGB and hand landmarks. The model is trained with a composite loss that explicitly penalizes errors in unobserved (non-fingertip) cells:

$$\mathcal{L}_{\text{inpaint}} = \text{MSE}(\hat{\mathbf{M}}, \mathbf{M}) + \lambda \cdot \frac{\sum_{(i,j) \notin \mathcal{F}} |\hat{M}_{ij} - M_{ij}|}{|\{(i,j) \notin \mathcal{F}\}|}$$

*(Results and visualization to be added once Edward shares training metrics.)*

This work is complementary to the classification baseline in an important way: while our classification model asks whether pre-contact signals predict *which* grip type will form, the inpainting model asks whether partial contact observations can recover the *spatial distribution* of pressure across the full hand. Together they bracket the full prediction problem — from pre-contact coarse intent to at-contact dense spatial reconstruction.

## Failure Analysis

Several failure modes are worth noting for future work. The long-tailed label distribution — where Tip Pinch, Small Diameter, and Prismatic 2 Finger dominate — likely causes the model to over-predict common classes, explaining the substantial gap between accuracy (31.9%) and macro F1 (0.169) in Run 1. Grip types with short pre-contact windows (onset\_idx $< 5$) are systematically underrepresented in training, which may bias the model against grip types that tend to begin contact quickly. Finally, the fixed sequence length of 20 frames treats all pre-contact windows uniformly regardless of their actual length — a variable-length sequence model may better exploit the rich pre-contact windows available for some grip types while gracefully handling short ones.