# Phase 6E: Correspondence-Aware Neural Prototype Design Document

## 1. Executive Summary
Phase 6E designs and trains a controlled neural prototype that incorporates **landmark-guided source→target geometric correspondence** directly into the neural face-swapping architecture.

In Phase 6C, injecting raw source spatial features failed because source and target faces possessed different poses, yaw angles, and expressions, causing severe spatial collision. In Phase 6D, deterministic MediaPipe 478-landmark piecewise-affine warping demonstrated an ArcFace identity retention of $L = 0.6393$ (+0.6301 gain over baseline) and reduced landmark reprojection error by 70.1% ($4.71 \to 1.41$ px).

Phase 6E introduces an **Aligned Source Feature Encoder** and **Gated Residual Fusion** mechanism operating on the warped source image, bridging source identity textures into target coordinates without replacing the proven global ArcFace AdaIN conditioning.

---

## 2. Review of Current Phase 6B.2 Architecture
In the Phase 6B.2 model (`AdaINFaceSwapModel`):
1. **Source Pathway**:
   - Source RGB $I_s \in \mathbb{R}^{B \times 3 \times 128 \times 128}$ is resized to $112 \times 112$ and passed through the frozen official ArcFace backbone (`ms1mv2_iresnet50.pth`).
   - Output: 512-D L2-normalized identity embedding $z_{\text{id}} \in \mathbb{R}^{B \times 512}$.
2. **Target Pathway**:
   - Target RGB $I_t \in \mathbb{R}^{B \times 3 \times 128 \times 128}$ undergoes 2D Gaussian blur ($K=9, \sigma=3.0$) to suppress high-frequency textures, producing $I_{t,\text{low}}$.
   - $I_{t,\text{low}}$ is concatenated with the precomputed structural landmark distance map $L_t \in \mathbb{R}^{B \times 1 \times 128 \times 128}$ to form $X_{\text{tgt}} \in \mathbb{R}^{B \times 4 \times 128 \times 128}$.
   - $X_{\text{tgt}}$ is downsampled via `TargetStructureEncoder` through 4 convolutional stages with non-affine `InstanceNorm2d`:
     $$128 \times 128 \xrightarrow{/2} 64 \times 64 \xrightarrow{/2} 32 \times 32 \xrightarrow{/2} 16 \times 16 \xrightarrow{/2} 8 \times 8$$
     producing target spatial bottleneck representation $F_{\text{tgt}} \in \mathbb{R}^{B \times 512 \times 8 \times 8}$.
3. **Generator Decoder**:
   - $F_{\text{tgt}}$ passes through 4 bottleneck `AdaINResBlock2d` blocks at $8 \times 8$, modulated by $z_{\text{id}}$.
   - Progressively upsampled $8 \to 16 \to 32 \to 64 \to 128$ with AdaIN conditioning at each resolution.
   - Dual output heads produce synthesized face $I_{\text{swap}} \in \mathbb{R}^{B \times 3 \times 128 \times 128}$ and soft blend mask $M_{\text{pred}} \in \mathbb{R}^{B \times 1 \times 128 \times 128}$.
4. **Compositing**:
   - Soft compositor blends: $I_{\text{composite}} = M_{\text{pred}} \odot I_{\text{swap}} + (1 - M_{\text{pred}}) \odot I_t$.

---

## 3. Where Source Identity Currently Enters vs Where the New Pathway Enters
- **Current Identity Pathway**:
  - Purely global: $z_{\text{id}}$ modulates channels in `AdaIN2d` via an affine scale/shift:
    $$\text{AdaIN}(x, z_{\text{id}}) = (1 + \gamma(z_{\text{id}})) \cdot \text{InstanceNorm}(x) + \beta(z_{\text{id}})$$
  - It controls global facial appearance (skin tone, eye color, identity semantics) but lacks fine spatial identity details (distinctive nose shape, eye contour, lip shape, mole/freckle placement).
- **New Aligned Source Feature Pathway**:
  - Operates on $I_{s \to t}^{\text{aligned}} \in \mathbb{R}^{B \times 3 \times 128 \times 128}$, which is the source image pre-warped into the target's exact 478-landmark spatial configuration.
  - Aligned source features $F_{\text{src}} \in \mathbb{R}^{B \times 512 \times 8 \times 8}$ are extracted by a dedicated `AlignedSourceEncoder`.
  - Injected directly into the $8 \times 8$ bottleneck alongside $F_{\text{tgt}}$ via **Gated Residual Fusion**.
  - The two pathways are complementary:
    - ArcFace $z_{\text{id}}$ provides the global biometric anchor.
    - $F_{\text{src}}$ provides geometrically aligned spatial textures and structural nuances.

---

## 4. Phase 6D Correspondence & Confidence Mask Computation
For each training pair $(I_s, I_t)$:
1. **MediaPipe Landmark Extraction**:
   - Source landmarks $S \in \mathbb{R}^{478 \times 2}$, Target landmarks $T \in \mathbb{R}^{478 \times 2}$.
   - 8 peripheral boundary anchors added to both sets: corners `(0,0), (127,0), (0,127), (127,127)` and edge centers `(63,0), (0,63), (127,63), (63,127)`.
2. **Delaunay Piecewise-Affine Warp**:
   - 2D Delaunay triangulation constructed on $T_{\text{all}}$.
   - Dense backward mapping via vectorized barycentric coordinates:
     $$\mathbf{b} = \mathbf{T}_i \cdot (\mathbf{x} - \mathbf{r}_i), \quad \mathbf{x}_{\text{src}} = b_0 \mathbf{s}_0 + b_1 \mathbf{s}_1 + b_2 \mathbf{s}_2$$
   - Resampled via bilinear interpolation to produce $I_{s \to t}^{\text{aligned}} \in \mathbb{R}^{B \times 3 \times 128 \times 128}$.
3. **Geometric Correspondence Confidence Mask**:
   - For each triangle $i$, area ratio $R_i = \max(A_T / A_S, A_S / A_T)$ and affine condition number $\kappa_i = \sigma_{\max} / \sigma_{\min}$ measure local planar distortion.
   - Confidence per triangle: $\text{conf}_i = \text{clip}(\exp(-0.25 \cdot (\sqrt{R_i \cdot \kappa_i} - 1.0)), 0.10, 1.0)$.
   - Rasterized and Gaussian-smoothed into full-resolution confidence map $C_{\text{geom}} \in \mathbb{R}^{B \times 1 \times 128 \times 128}$.
   - High confidence ($\approx 1.0$) on frontal, low-distortion regions; smoothly attenuates ($\to 0.10$) over laterally stretched, self-occluded cheeks under large yaw.

---

## 5. Tensor Shapes at Every Stage

| Component / Layer | Input Tensor Shape | Output Tensor Shape | Notes |
| :--- | :---: | :---: | :--- |
| **Source Image ($I_s$)** | — | $[B, 3, 128, 128]$ | In $[-1.0, 1.0]$ |
| **Target Image ($I_t$)** | — | $[B, 3, 128, 128]$ | In $[-1.0, 1.0]$ |
| **6D Piecewise Warp** | $I_s, S, T$ | $I_{s \to t}^{\text{aligned}}$: $[B, 3, 128, 128]$ | In $[-1.0, 1.0]$ |
| **Confidence Map** | $S, T$ | $C_{\text{geom}}$: $[B, 1, 128, 128]$ | In $[0.10, 1.0]$ |
| **ArcFace Backbone** | $I_s$ (interpolated to $112 \times 112$) | $z_{\text{id}}$: $[B, 512]$ | Frozen, L2-normalized |
| **Target Blur + Landmark**| $I_t, L_t$ | $X_{\text{tgt}}$: $[B, 4, 128, 128]$ | Blur $\sigma=3.0$ |
| **TargetStructureEncoder**| $X_{\text{tgt}}$: $[B, 4, 128, 128]$ | $F_{\text{tgt}}$: $[B, 512, 8, 8]$ | Non-affine IN2d |
| **AlignedSourceEncoder** | $I_{s \to t}^{\text{aligned}}$: $[B, 3, 128, 128]$ | $F_{\text{src}}$: $[B, 512, 8, 8]$ | Non-affine IN2d |
| **Confidence Downsample**| $C_{\text{geom}}$: $[B, 1, 128, 128]$ | $C_8$: $[B, 1, 8, 8]$ | Bilinear interpolation |
| **Modulated Source Feats**| $F_{\text{src}} \odot C_8$ | $F_{\text{src,mod}}$: $[B, 512, 8, 8]$ | Modulated by geometry |
| **Gate Conv ($G$)** | $[F_{\text{tgt}}, F_{\text{src,mod}}]$: $[B, 1024, 8, 8]$ | $\text{Gate}$: $[B, 512, 8, 8]$ | Sigmoid activation |
| **Residual Projection ($P$)**| $F_{\text{src,mod}}$: $[B, 512, 8, 8]$ | $\Delta_{\text{src}}$: $[B, 512, 8, 8]$ | $3 \times 3$ Conv |
| **Fused Bottleneck** | $F_{\text{tgt}} + \text{Gate} \odot \Delta_{\text{src}}$ | $F_{\text{fused}}$: $[B, 512, 8, 8]$ | Fed into generator |
| **Generator Bottleneck** | $F_{\text{fused}}$: $[B, 512, 8, 8], z_{\text{id}}$ | $X_8$: $[B, 512, 8, 8]$ | 4 AdaIN ResBlocks |
| **Generator Upsampling** | $X_8, z_{\text{id}}$ | $X_{128}$: $[B, 64, 128, 128]$ | 4 stages w/ AdaIN |
| **Output Heads** | $X_{128}$ | $I_{\text{swap}}$: $[B, 3, 128, 128]$, $M_{\text{pred}}$: $[B, 1, 128, 128]$ | Tanh / Sigmoid |
| **Soft Compositor** | $I_{\text{swap}}, M_{\text{pred}}, I_t$ | $I_{\text{composite}}$: $[B, 3, 128, 128]$ | Alpha blend |

---

## 6. Gated Residual Fusion Architecture
The fusion at the $8 \times 8$ bottleneck is formulated as:
$$\text{Gate} = \sigma\left(\text{Conv}_{3 \times 3}(\text{Concat}(F_{\text{tgt}}, F_{\text{src,mod}}))\right) \in \mathbb{R}^{B \times 512 \times 8 \times 8}$$
$$\Delta_{\text{src}} = \text{Conv}_{3 \times 3}(F_{\text{src,mod}}) \in \mathbb{R}^{B \times 512 \times 8 \times 8}$$
$$F_{\text{fused}} = F_{\text{tgt}} + \text{Gate} \odot \Delta_{\text{src}}$$

### Initialization Guarantees:
- `proj_source` final conv weights and biases are initialized to zero ($\mathbf{W} = 0, \mathbf{b} = 0$).
- Consequently, at step 0: $\Delta_{\text{src}} = \mathbf{0}$, which ensures:
  $$F_{\text{fused}}\big|_{t=0} = F_{\text{tgt}}$$
- The model starts **identically** from the verified Phase 6B.2 baseline, preventing any initial distribution shock.

---

## 7. Reused vs Newly Initialized Parameters

### Reused Parameters (Loaded from `ml/checkpoints/stage2/best_model.pt`):
1. `target_encoder` (All 4 convolutional downsampling layers, 100% weights intact).
2. `generator` (All 4 bottleneck AdaIN blocks, 4 upsampling stages, RGB head, Mask head, 100% weights intact).
3. `compositor` (Parameter-free).
4. `discriminator` (Multiscale PatchGAN discriminator, 100% weights intact).
5. `arcface` (Official MS1MV2 iResNet-50, frozen).

### Newly Initialized Parameters:
1. `aligned_source_encoder`:
   - 4 downsampling convolutional stages mirroring the target encoder structure ($3 \to 64 \to 128 \to 256 \to 512$).
   - Standard He/Kaiming normal initialization.
2. `gate_conv`:
   - $1024 \to 512$ convolution, kernel size 3, padding 1.
3. `proj_source`:
   - $512 \to 512$ convolution, kernel size 3, padding 1, zero-initialized.

---

## 8. Training Configuration & Loss Protocol
- **Optimizer**: Adam ($\text{lr}_g = 1 \times 10^{-4}, \text{lr}_d = 1 \times 10^{-4}, \beta_1 = 0.5, \beta_2 = 0.999$).
- **Batch Size**: 1 (gradient accumulation = 4 steps $\to$ effective batch size 4).
- **Duration**: **Exactly 100 optimizer steps**.
- **Pairing**: 100% genuine cross-identity validation/training pairs.
- **Losses**: Baseline Phase 6B.2 anti-collapse loss suite:
  - $\lambda_{\text{id}} = 10.0$ (Composite ArcFace loss)
  - $\lambda_{\text{id,swap}} = 8.0$ (Swap-image ArcFace loss)
  - $\lambda_{\text{struct}} = 5.0$ (Reconstruction L1 + perceptual)
  - $\lambda_{\text{bg}} = 5.0$ (Background L1)
  - $\lambda_{\text{mask}} = 5.0$ (Anti-collapse mask coverage)
  - $\lambda_{\text{adv}} = 0.5$ (PatchGAN hinge loss)
- **Validation**: Every 25 steps on 10 fixed held-out cross-identity validation pairs.
- **Ablation Protocol**: At every validation step, evaluate each pair with the new aligned source pathway **Enabled** vs **Disabled** ($\Delta_{\text{src}} = \mathbf{0}$) to measure the exact differential gain $\Delta (A-C)$.
