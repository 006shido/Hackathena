# Phase 6E: Correspondence-Aware Neural Prototype Report

## 1. Objective
Following the validation of Phase 6D (where deterministic MediaPipe 478-landmark piecewise-affine warping achieved an ArcFace identity retention of $L = 0.6393$ with a 70.1% reduction in landmark reprojection error), Phase 6E tested the hypothesis:

> **Hypothesis**: Injecting geometrically aligned source spatial features into the target-conditioned generator via an Aligned Source Feature Encoder and Gated Residual Fusion will produce a sustained and measurable increase in source identity transfer compared to the Phase 6B.2 baseline.

Phase 6E was executed as a controlled, 100-step prototype experiment without long training, using 100% genuine cross-identity CelebA pairs.

---

## 2. Baseline Model
- **Checkpoint**: Initialized from `ml/checkpoints/stage2/best_model.pt` (Phase 6B.2 Step 750).
- **Phase 6B.2 Baseline Metrics**:
  - Source Identity Gain ($A - C$): $\approx \mathbf{+0.0651}$ (peak observed during Phase 6B.2 extended training).
  - Target Identity Correlation ($B$): $\approx 0.18 - 0.22$.
  - Anti-collapse mask: inside face $\approx 0.89$, outside face $\approx 0.001$, active coverage $\approx 44.5\%$.
- **ArcFace Extractor**: Frozen MS1MV2 iResNet-50 (`ms1mv2_iresnet50.pth`).

---

## 3. Architecture Change
1. **Source Spatial Pre-Alignment**:
   - For every cross-identity pair $(I_s, I_t)$, MediaPipe 478 landmarks are extracted.
   - Using the Phase 6D local piecewise-affine Delaunay warp, source pixels are resampled into target geometry, producing $I_{s \to t}^{\text{aligned}} \in \mathbb{R}^{B \times 3 \times 128 \times 128}$.
2. **Aligned Source Feature Encoder**:
   - 4-stage convolutional downsampler ($3 \to 64 \to 128 \to 256 \to 512$) with non-affine `InstanceNorm2d`, outputting $F_{\text{src}} \in \mathbb{R}^{B \times 512 \times 8 \times 8}$.
3. **Geometric Confidence Modulation**:
   - Triangle distortion metric rasterized into a full-resolution confidence map $C_{\text{geom}} \in \mathbb{R}^{B \times 1 \times 128 \times 128}$, downsampled to $8 \times 8$, modulating $F_{\text{src}}$:
     $$F_{\text{src,mod}} = F_{\text{src}} \odot C_8$$
4. **Gated Residual Fusion**:
   - At the $8 \times 8$ bottleneck:
     $$\text{Gate} = \sigma\left(\text{Conv}_{3 \times 3}([F_{\text{tgt}}, F_{\text{src,mod}}])\right) \in \mathbb{R}^{B \times 512 \times 8 \times 8}$$
     $$\Delta_{\text{src}} = \text{Conv}_{3 \times 3}(F_{\text{src,mod}}) \in \mathbb{R}^{B \times 512 \times 8 \times 8}$$
     $$F_{\text{fused}} = F_{\text{tgt}} + \text{Gate} \odot \Delta_{\text{src}}$$
   - Initialized with zero weights on $\Delta_{\text{src}}$, guaranteeing an exact start from Phase 6B.2 baseline.
5. **Preserved Global Conditioning**:
   - Frozen ArcFace 512-D $z_{\text{id}}$ retained via AdaIN across all generator bottleneck and upsampling blocks.

---

## 4. Tensor Shapes at Every Stage
- $I_s, I_t, I_{s \to t}^{\text{aligned}}$: $[B, 3, 128, 128]$
- $C_{\text{geom}}$: $[B, 1, 128, 128]$
- $z_{\text{id}}$: $[B, 512]$
- $F_{\text{tgt}}, F_{\text{src}}, F_{\text{src,mod}}, \Delta_{\text{src}}, F_{\text{fused}}$: $[B, 512, 8, 8]$
- $\text{Gate}$: $[B, 512, 8, 8]$
- $I_{\text{swap}}, I_{\text{composite}}$: $[B, 3, 128, 128]$
- $M_{\text{pred}}$: $[B, 1, 128, 128]$

---

## 5. Correspondence Method
- Vectorized Delaunay triangulation on target landmarks plus 8 peripheral boundary anchors.
- Backward pixel mapping using barycentric coordinates and bilinear resampling (`cv2.remap`).
- Complete full-frame coverage with zero boundary clipping or border seams.

---

## 6. Confidence Map Derivation
- For each Delaunay triangle $i$:
  - Area ratio: $R_i = \max(A_T / A_S, A_S / A_T)$
  - Affine condition number: $\kappa_i = \sigma_{\max} / \sigma_{\min}$
  - Local planar distortion: $D_i = \sqrt{R_i \cdot \kappa_i}$
  - Confidence: $\text{conf}_i = \text{clip}(\exp(-0.25 \cdot \max(0, D_i - 1.0)), 0.10, 1.0)$
- Rasterized into full $128 \times 128$ image, smoothed with $5 \times 5$ Gaussian kernel.

---

## 7. Training Configuration
- **Total Optimizer Steps**: Exactly 100 steps.
- **Batch Size**: 1 (Gradient Accumulation = 4 $\to$ Effective Batch Size 4).
- **Optimizers**: Adam ($lr_g = 1 \times 10^{-4}, lr_d = 1 \times 10^{-4}, \beta = (0.5, 0.999)$).
- **Precision**: Mixed Precision (AMP `GradScaler` on CUDA).
- **Loss Weights**: Verified Phase 6B.2 values:
  - $\lambda_{\text{id}} = 10.0, \lambda_{\text{id,swap}} = 8.0, \lambda_{\text{struct}} = 5.0, \lambda_{\text{bg}} = 5.0, \lambda_{\text{mask}} = 5.0, \lambda_{\text{adv}} = 0.5$.
- **Validation Schedule**: Evaluated at steps 0, 25, 50, 75, 100.

---

## 8. Validation Setup
- Evaluated on 10 fixed, held-out cross-identity validation pairs from CelebA val split.
- Dual evaluation per checkpoint:
  - **Mode A (Enabled)**: Full correspondence-aware pathway active.
  - **Mode B (Disabled / Ablation)**: Source pathway bypassed ($\Delta_{\text{src}} = \mathbf{0}$).

---

## 9. Identity Metrics Progression

| Step | ArcFace $A$ (Source $\cdot$ Output) | ArcFace $B$ (Target $\cdot$ Output) | ArcFace $C$ (Cross-ID Base) | ArcFace $D$ (Source $\cdot$ Swap) | Source Identity Gain ($A - C$) | $D - C$ | $A - B$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | $-0.0233 \pm 0.0851$ | $0.2535$ | $-0.0345$ | $-0.0119$ | **$+0.0112$** | $+0.0226$ | $-0.2768$ |
| **25** | $-0.0118 \pm 0.0969$ | $0.2196$ | $-0.0345$ | $-0.0023$ | **$+0.0227$** | $+0.0322$ | $-0.2314$ |
| **50** | $+0.0119 \pm 0.0852$ | $0.2336$ | $-0.0345$ | $-0.0024$ | **$+0.0464$** | $+0.0321$ | $-0.2217$ |
| **75** | $+0.0088 \pm 0.0867$ | $0.2191$ | $-0.0345$ | $-0.0016$ | **$+0.0433$** | $+0.0329$ | $-0.2103$ |
| **100** | $-0.0177 \pm 0.0997$ | $0.2531$ | $-0.0345$ | $-0.0272$ | **$+0.0168$** | $+0.0073$ | $-0.2708$ |

Identity gain peaked at step 50 ($A - C = +0.0464$) before collapsing back down to $+0.0168$ at step 100.

---

## 10. Ablation Metrics: Direct Pathway Contribution
Evaluating whether the newly added pathway actually affects the output:

| Step | $A - C$ (Pathway ENABLED) | $A - C$ (Pathway DISABLED) | $\Delta(A - C)$ (Pathway Net Gain) |
| :---: | :---: | :---: | :---: |
| **0** | $+0.0112$ | $+0.0112$ | **$+0.0000$** |
| **25** | $+0.0227$ | $+0.0237$ | **$-0.0010$** |
| **50** | $+0.0464$ | $+0.0448$ | **$+0.0016$** |
| **75** | $+0.0433$ | $+0.0420$ | **$+0.0012$** |
| **100** | $+0.0168$ | $+0.0159$ | **$+0.0010$** |

### Key Finding:
Across all evaluation steps, **the net contribution of the aligned source pathway $\Delta(A - C)$ never exceeded $+0.0016$**. Disabling the pathway produces virtually identical identity scores ($+0.0159$ vs $+0.0168$ at step 100). The generator decoder fails to route features from the newly added pathway to the output pixels.

---

## 11. Spatial Metrics

| Step | Mask Inside Face | Mask Outside Face | Active Mask % | Background L1 Error | Landmark Re-Detection | Mean Landmark Error |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | $0.893$ | $0.001$ | $44.6\%$ | $0.0046$ | **90.0%** | $3.79$ px |
| **25** | $0.899$ | $0.001$ | $45.3\%$ | $0.0055$ | **100.0%** | $4.31$ px |
| **50** | $0.876$ | $0.003$ | $44.9\%$ | $0.0064$ | **20.0%** | $6.07$ px |
| **75** | $0.893$ | $0.001$ | $44.9\%$ | $0.0048$ | **60.0%** | $3.37$ px |
| **100** | $0.882$ | $0.005$ | $44.9\%$ | $0.0061$ | **90.0%** | $3.62$ px |

At step 50, independent face re-detection collapsed to 20% due to horizontal raster striations before recovering at step 100 as the generator reverted to predicting smooth target-like skin.

---

## 12. Pose-Stratified Performance
Performance on the 10 validation pairs partitioned by pose disparity at Step 100:
- **Low Pose Disparity (Similar Pose)**:
  - $A - C = +0.0241$
  - Re-detection rate: 100%
- **Medium Pose Disparity (Moderate Pose)**:
  - $A - C = +0.0182$
  - Re-detection rate: 100%
- **High Pose Disparity (Large Pose)**:
  - $A - C = -0.0042$
  - Re-detection rate: 66.7%

---

## 13. Qualitative Findings
Validation sample grids generated at each interval:
- [step_000_val_grid.png](file:///c:/files/forgitclone/Hackathena/ml/checkpoints/stage2/phase6e_prototype/samples/step_000_val_grid.png)
- [step_050_val_grid.png](file:///c:/files/forgitclone/Hackathena/ml/checkpoints/stage2/phase6e_prototype/samples/step_050_val_grid.png)
- [step_100_val_grid.png](file:///c:/files/forgitclone/Hackathena/ml/checkpoints/stage2/phase6e_prototype/samples/step_100_val_grid.png)

![Step 100 Validation Grid](C:\Users\shido\.gemini\antigravity-ide\brain\633ef793-a7b5-42c1-b4cc-c2c615279238\step_100_val_grid.png)

### Observations:
1. **Col 3 (6D Aligned Source)** vs **Col 4 (6E Neural Output)**:
   - In Column 3 (the deterministic Phase 6D geometric warp), source identity, eyes, nose, teeth, and skin texture are crisp, natural, and recognizable.
   - In Column 4 (the neural generator output), the face suffers from severe horizontal raster striations, blurred eye sockets, and smeared features.
2. **Horizontal Striation Artifacts**:
   - Similar to Phase 6B.5, injecting multi-pathway spatial gradients into the decoder causes high-frequency ringing and horizontal banding across the synthesized facial region.

---

## 14. Hardware Performance & Telemetry
- **Hardware**: NVIDIA GeForce RTX 5060 Laptop GPU (8,123.4 MB VRAM).
- **Execution Speed**: 2.41 steps/second.
- **Total Runtime**: **41.42 seconds** for 100 optimizer steps (400 forward/backward passes).
- **Peak VRAM Allocated**: **1,604.9 MB** (well below 8 GB limit).
- **Peak VRAM Reserved**: **1,666.0 MB**.

---

## 15. Post-Run Integrity Verification
- **ArcFace Checkpoint**: `ml/models/weights/ms1mv2_iresnet50.pth` SHA-256 confirmed:
  `2B75B93C48B01C78A4263F7295AB2DBF84F85190C51BE45B92C4C0B0AEDCEAA3`.
- **Dataset**: `ml/data/celeba/img_align_celeba` (202,599 files) and `identity_CelebA.txt` (202,599 lines) verified untouched.
- **Baseline Checkpoints**:
  - `ml/checkpoints/stage1/best_model.pt` (559,186,268 bytes) untouched.
  - `ml/checkpoints/stage2/best_model.pt` (592,389,456 bytes) untouched.
- **Frontend / Backend**: `client/` and `server/` untouched.
- **Stability**: Zero NaNs, zero Infs, zero OOMs.

---

## 16. Comparison Against Prior Phases

| Phase | Core Mechanism | Peak / Final $A - C$ Gain | Key Observed Failure Mode |
| :--- | :--- | :---: | :--- |
| **Phase 6B.2** | Baseline AdaIN (Step 750) | **$+0.0651$** | Strong baseline; modest identity dominance |
| **Phase 6B.3** | Amplified Identity Loss ($\lambda_{\text{id}}=20$) | $+0.0436$ | Unstable training, yellow hue shift, mask collapse |
| **Phase 6B.4** | Strong Target Blur ($\sigma=7.0$) | $+0.0110$ | Smeared identity-neutral faces, lost facial structure |
| **Phase 6B.5** | Multi-Scale AdaIN Injection | $+0.0385$ | Severe horizontal striations, fluctuating metrics |
| **Phase 6C** | Raw Source CNN Feature Injection | $+0.0090$ | Pose conflict between raw source and target, ghost shadows |
| **Phase 6D** | Deterministic Landmark Warp | **$+0.6301$** | Planar stretch on large yaw; validated geometric correspondence |
| **Phase 6E** | Aligned Source Encoder + Gated Fusion | **$+0.0168$** (Peak: $+0.0464$) | **Ablation net gain only $+0.0010$**; horizontal raster striations |

---

## 17. Comparison Against Phase 6C
- **Improvement over 6C**:
  - In Phase 6C, raw source CNN features caused immediate spatial collision (ghost eyes, double mouths) because source and target had different poses.
  - In Phase 6E, pre-aligning source pixels into target geometry eliminated the double-pose collision. The mask correctly aligned with the target head silhouette.
- **Shared Limitation**:
  - In both phases, feeding spatial source features into a decoder whose upsampling blocks are heavily conditioned on AdaIN resulted in the decoder ignoring the bottleneck spatial injection ($\Delta A - C < 0.002$) while developing horizontal deconvolution striations.

---

## 18. Failure Modes Identified in Phase 6E
1. **Bottleneck Decoupling / Pathway Neglect**:
   - The generator decoder is overwhelmingly dominated by the AdaIN modulation at every upsampling layer ($16 \to 32 \to 64 \to 128$). Because AdaIN directly modulates channel mean and variance at all scales, perturbations introduced at the $8 \times 8$ bottleneck are largely washed out during upsampling.
   - This directly explains why the ablation showed $\Delta(A - C) \approx +0.0010$.
2. **Horizontal Striation Ringing**:
   - Updating the generator decoder weights while optimizing the new encoder and gate induces high-frequency grid artifacts during nearest-neighbor + conv upsampling.
3. **Contrast with Deterministic Warp**:
   - The deterministic 6D warp in Column 3 preserves crisp facial features ($L = 0.6393$), whereas passing that warped face through an encoder-generator pipeline degrades the signal down to $A - C = +0.0168$.

---

## 19. Architectural Interpretation
The results of Phases 6C, 6D, and 6E yield an unmistakable conclusion:
- **Phase 6D proved that deterministic landmark-guided warping provides genuine identity transfer ($+0.6301$) in target pose**.
- **Phase 6E proved that attempting to compress that warped image through an $8 \times 8$ CNN bottleneck and re-synthesize it via an AdaIN generator destroys the fine identity detail that the geometric warp established**.
- Rather than forcing the generator to reconstruct source identity from scratch through an $8 \times 8$ bottleneck, future architectures must directly blend or warp source facial textures into target coordinates at higher resolutions, using the neural network primarily for seamless boundary harmonization, relighting, and occlusion handling.

---

## 20. Final Classification

### **CLASSIFICATION: FAIL**

### Rationale:
1. **Identity Transfer Criterion Not Met**:
   - Phase 6E achieved a final source identity gain of $A - C = \mathbf{+0.0168}$ (peaking at $+0.0464$ at step 50), which is **significantly inferior to the Phase 6B.2 baseline of $+0.0651$**.
2. **Ablation Proves Pathway Ineffectiveness**:
   - The net gain attributed to the correspondence-aware source pathway is only $\Delta(A - C) = \mathbf{+0.0010}$. Disabling the pathway yields the same identity score ($+0.0159$ disabled vs $+0.0168$ enabled).
3. **Severe Visual Artifacts**:
   - The generator outputs degraded, striation-ridden faces that fail to capture the rich source likeness evident in the 6D geometric warp.

**Phase 6E is complete. Execution stopped per instructions.**
