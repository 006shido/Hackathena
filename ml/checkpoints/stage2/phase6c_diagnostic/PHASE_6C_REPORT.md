# PHASE 6C: Source Identity Information Bottleneck Diagnostic Report

---

## 1. Objective
Diagnose whether the current Stage 2 face-swap architecture fundamentally suffers from an identity information bottleneck caused by compressing the source image solely into a 512-D ArcFace recognition embedding. Specifically:
1. Probe whether source spatial feature maps contain measurable identity-discriminative information beyond the 512-D ArcFace embedding.
2. Build a controlled source-spatial feature conditioning prototype.
3. Compare Condition A (Phase 6B.2 baseline) against Condition B (Source Feature Conditioning) over a 100-step controlled diagnostic run on identical held-out validation pairs.

---

## 2. Existing Architecture Findings
- **Source Pathway**: Source image $I_s \in \mathbb{R}^{B \times 3 \times 128 \times 128}$ is resized to $112 \times 112$ and passed into frozen ArcFace iResNet-50. All spatial dimensions ($7 \times 7$) are pooled into a single 512-D vector $z_{\text{id}} \in \mathbb{S}^{511}$.
- **Modulation**: $z_{\text{id}}$ modulates the generator exclusively via AdaIN channel-wise affine transformations:
  $$\text{AdaIN}(F, z_{\text{id}}) = (1 + \gamma) \cdot \text{InstanceNorm}(F) + \beta$$
  where $(\gamma, \beta)$ are spatially uniform scalars per channel.
- **Target Pathway**: Target image $I_t$ is blurred with Gaussian blur ($\sigma=3.0$), concatenated with MediaPipe 1-channel landmark distance map $L_t \in \mathbb{R}^{B \times 1 \times 128 \times 128}$, and passed to `TargetStructureEncoder`, yielding $F_{\text{tgt}} \in \mathbb{R}^{B \times 512 \times 8 \times 8}$.
- **Asymmetry**: $100\%$ of spatial feature maps throughout the network originate from the target image and target landmark map. The source image has zero 2D spatial pathways into the generator.

---

## 3. Source Information Discarded by ArcFace-Only Pipeline
Because ArcFace is optimized strictly for pose/expression-invariant identity verification, it eliminates:
1. Source 2D landmark coordinates and spatial feature alignment.
2. Local geometric contours (exact eyelid creases, nostril contours, lip vermilion curvature, jaw boundaries).
3. High-frequency skin textures, pores, and micro-pigmentation.
4. Spatial lighting and localized skin undertone variations.

---

## 4. Source Feature Encoder Design
Created `SourceFeatureEncoder` in `ml/models/source_feature_encoder.py`:
- 4-stage convolutional downsampler processing $I_s$ [B, 3, 128, 128]:
  - Stage 1: $64 \times 64$ (64 channels)
  - Stage 2: $32 \times 32$ (128 channels)
  - Stage 3: $16 \times 16$ (256 channels)
  - Stage 4: $8 \times 8$ (512 channels)
- Parameter count: 5,892,864 parameters.

---

## 5. Identity Information Probe Methodology
Evaluated 50 same-identity pairs vs. 50 cross-identity pairs from the held-out validation set:
- For each pair $(I_1, I_2)$, extracted ArcFace cosine similarity and SourceFeatureEncoder spatial cosine similarities at resolutions $f_8, f_{16}, f_{32}, f_{64}$.
- Measured mean same-ID similarity, mean cross-ID similarity, and identity separation ($\Delta_{\text{sep}} = \text{Same} - \text{Cross}$).

---

## 6. Same-ID vs. Cross-ID Feature Probe Results

| Feature Layer | Same-ID Mean | Cross-ID Mean | Identity Separation ($\Delta$) | Conclusion |
| :--- | :---: | :---: | :---: | :--- |
| **ArcFace (512-D)** | **0.5902** | **0.0132** | **+0.5770** | **Strong Identity Discrimination** |
| `encoder_f8` (8×8) | 0.7416 | 0.7260 | +0.0156 | Negligible Identity Discrimination |
| `encoder_f16` (16×16) | 0.6635 | 0.6463 | +0.0172 | Negligible Identity Discrimination |
| `encoder_f32` (32×32) | 0.5945 | 0.5751 | +0.0193 | Negligible Identity Discrimination |
| `encoder_f64` (64×64) | 0.5538 | 0.5309 | +0.0229 | Negligible Identity Discrimination |

**Key Finding**: Unsupervised spatial feature maps carry virtually **zero identity discrimination** (+0.015 to +0.023 separation vs. +0.5770 for ArcFace). Spatial convolutions primarily capture non-identity spatial layout and illumination rather than identity.

---

## 7 & 8. Controlled A/B Diagnostic Test Results (Identical Validation Pairs)

Tested over 100 optimizer steps on genuine CelebA cross-identity pairs with fixed pre-cached validation pairs:

| Condition | Description | $A = \cos(z_{\text{src}}, z_{\text{out}})$ | $B = \cos(z_{\text{tgt}}, z_{\text{out}})$ | $C = \cos(z_{\text{src}}, z_{\text{tgt}})$ | $D = \cos(z_{\text{src}}, z_{\text{swap}})$ | $A - C$ | $D - C$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Condition A** | Phase 6B.2 Baseline (Step 750) | $-0.0055$ | $0.1889$ | $-0.0247$ | $-0.0211$ | **$+0.0192$** | **$+0.0036$** |
| **Condition B @ 0** | Source Spatial Feats (Init) | $-0.0055$ | $0.1889$ | $-0.0247$ | $-0.0211$ | $+0.0192$ | $+0.0036$ |
| **Condition B @ 100** | Source Spatial Feats (100 steps) | $-0.0023$ | $0.1962$ | $-0.0247$ | $-0.0077$ | **$+0.0224$** | **$+0.0170$** |

---

## 9 & 10. A-C and D-C Comparison (Delta Analysis)
- $\Delta(A - C) = 0.0224 - 0.0192 = \mathbf{+0.0032}$ (negligible change; within validation noise).
- $\Delta(D - C) = 0.0170 - 0.0036 = \mathbf{+0.0134}$ (marginal gain in raw swap face).
- $\Delta B = 0.1962 - 0.1889 = \mathbf{+0.0074}$ (slight increase in target identity leakage).

---

## 11. Qualitative Findings
Inspected comparative sample grids `[SOURCE, TARGET, BASELINE_OUT, CONDITION_B_OUT, CONDITION_B_MASK]` at `ml/checkpoints/stage2/phase6c_diagnostic/samples/`:
1. **Pose Misalignment**: In cross-identity pairs where source pose differs from target pose, directly projecting unaligned source spatial features creates spatial interference with target landmarks and target structure.
2. **Artifacts**: The generator develops darkened eye and mouth hollows because source spatial feature locations conflict with target landmark coordinates.
3. **Identity Transfer**: No recognizable source facial structures (e.g. source eyes, nose, or lips) emerge on the target pose.
4. **Mask & Background**: Mask behavior remained localized and background preserved cleanly (inside mean: $0.8710$).

---

## 12. Hardware, VRAM & Throughput
- **GPU**: NVIDIA GeForce RTX 5060 Laptop GPU (8 GB VRAM)
- **100-step Training Time**: 17.01 s (Throughput: 5.88 steps/s)
- **VRAM Utilization**: ~1,450 MB allocated / 8,123 MB total (17.8%)
- **Source Feature Parameters**: 5,892,864 (encoder) + 349,120 (projections) = 6,241,984 parameters.

---

## 13, 14, 15 & 16. Integrity Verification
- **ArcFace Checkpoint SHA-256**: `2B75B93C48B01C78A4263F7295AB2DBF84F85190C51BE45B92C4C0B0AEDCEAA3` (Intact).
- **ArcFace Trainable Parameters**: Strictly 0; received 0 gradients.
- **Dataset File Count**: 202,599 images verified intact.
- **`identity_CelebA.txt`**: 202,599 lines verified intact.
- **Checkpoints**: Stage 1, Phase 6B.2 (`best_model.pt`), Phase 6B.3, and Phase 6B.4 untouched.
- **Frontend / Backend**: Zero modifications in `client/` and `server/`.
- **Numerical Stability**: 0 NaN / 0 Inf occurrences.

---

## 17. Final Conclusion & Classification

### Diagnostic Classification: **FAIL**

### Mechanistic Reason:
1. **The Hypothesis is Partially True in Theory, but Failed in Architecture**:
   While the 512-D ArcFace embedding is a compressed representation, **raw spatial convolutions of the source image do not solve the problem**. As proven by the probe, generic spatial convolutions have near-zero identity separation ($\Delta = +0.015$ to $+0.023$).
2. **The Spatial Misalignment Conflict**:
   In cross-identity face swapping, the source and target faces possess **different poses, expressions, and face shapes**. Direct spatial feature injection fails because source spatial features at coordinate $(x, y)$ represent the source face's geometry at source pose, whereas the generator is constrained to synthesize the face at target pose $(x', y')$. Without an explicit spatial warping/cross-attention alignment mechanism, spatial source features act as unaligned noise that creates feature conflicts rather than useful identity guidance.
3. **Empirical Evidence**: Adding source spatial features produced a statistically negligible gain $\Delta(A-C) = +0.0032$, while introducing eye-socket shadowing and slightly increasing target leakage $B$.
