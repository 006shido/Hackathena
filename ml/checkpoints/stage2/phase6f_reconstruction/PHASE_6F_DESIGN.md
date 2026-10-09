# Phase 6F: Aligned-Source Neural Reconstruction Diagnostic Design Document

## 1. Executive Summary
Phase 6D proved that deterministic MediaPipe 478-landmark local piecewise-affine warping creates a high-fidelity source-to-target correspondence, achieving an ArcFace source similarity of $L = 0.6393 \pm 0.1324$ (+0.6301 over baseline) and reducing landmark reprojection error by 70.1%.

However, Phase 6E demonstrated that injecting this aligned source into the existing generator architecture produced an identity gain of only $A - C = +0.0168$ at step 100 (peak $+0.0464$ at step 50), with the direct pathway ablation contributing a negligible $\Delta(A - C) = +0.0010$. Furthermore, the neural generator suffered from severe horizontal raster striations and blurred high-frequency facial textures.

**Core Diagnostic Hypothesis for Phase 6F**:
> The standard neural encoder/decoder bottleneck ($128 \times 128 \to 8 \times 8 \to 128 \times 128$) fundamentally discards the high-frequency spatial identity details present in the 6D aligned source. A multi-scale skip architecture (U-Net style) will preserve these spatial details and prevent deconvolution ringing, whereas bottleneck-only architectures will fail.

Phase 6F tests this purely as an **aligned-source autoencoding diagnostic** across three controlled conditions (A, B, and C) without target conditioning, GAN loss, or ArcFace loss.

---

## 2. Review of Existing Architectures

### 1. Existing Phase 6B.2 Generator Architecture
The Phase 6B.2 model (`AdaINFaceSwapModel`) comprises:
- `TargetStructureEncoder`: 4 convolutional stages downsampling $128 \times 128 \to 8 \times 8$.
- `IdentityConditionedGenerator`: 4 bottleneck residual blocks at $8 \times 8$ followed by 4 upsampling stages ($8 \to 16 \to 32 \to 64 \to 128$) and dual output heads (`rgb_head`, `mask_head`).
- `SoftCompositor`: Blends synthesized face with target image using predicted soft mask.

### 2. Existing Target Encoder
- Input: $X_{\text{tgt}} \in \mathbb{R}^{B \times 4 \times 128 \times 128}$ (3 channels Gaussian blurred target RGB $\sigma=3.0$, plus 1 channel landmark distance map).
- Downsampler: 4 strided $4 \times 4$ convolutions (stride 2, padding 1) with channel expansion: $4 \to 64 \to 128 \to 256 \to 512$.
- Output: Bottleneck spatial features $F_{\text{tgt}} \in \mathbb{R}^{B \times 512 \times 8 \times 8}$.

### 3. Existing Decoder
- 4 bottleneck blocks at $8 \times 8$ (`AdaINResBlock2d`).
- Upsampling stages:
  - $8 \times 8 \to 16 \times 16$: `F.interpolate(scale=2, mode='nearest')` $\to$ `Conv2d(512, 256)` $\to$ `AdaINResBlock2d(256)`
  - $16 \times 16 \to 32 \times 32$: `F.interpolate(scale=2, mode='nearest')` $\to$ `Conv2d(256, 128)` $\to$ `AdaINResBlock2d(128)`
  - $32 \times 32 \to 64 \times 64$: `F.interpolate(scale=2, mode='nearest')` $\to$ `Conv2d(128, 64)` $\to$ `AdaINResBlock2d(64)`
  - $64 \times 64 \to 128 \times 128$: `F.interpolate(scale=2, mode='nearest')` $\to$ `Conv2d(64, 64)` $\to$ `AdaINResBlock2d(64)`
- RGB Head: `Conv2d(64, 3, 3, 1, 1)` $\to$ `Tanh()` $\to I_{\text{swap}} \in [-1.0, 1.0]$.
- Mask Head: `Conv2d(64, 1, 3, 1, 1)` $\to$ `Sigmoid()` $\to M_{\text{pred}} \in [0.0, 1.0]$.

### 4. Existing Normalization
- `InstanceNorm2d(channels, affine=False)` is used in the encoders. Non-affine InstanceNorm strips per-instance mean and variance to suppress lighting and contrast variations.

### 5. Existing AdaIN Conditioning
- Formula: $\text{AdaIN}(x, z_{\text{id}}) = (1 + \gamma(z_{\text{id}})) \cdot \text{InstanceNorm}(x) + \beta(z_{\text{id}})$.
- A 2-layer MLP maps the 512-D ArcFace vector $z_{\text{id}}$ to channel-wise scale $\gamma$ and shift $\beta$.
- Applied at every residual block throughout the decoder.

### 6. Existing Upsampling Mechanism
- Nearest-neighbor interpolation (`mode='nearest'`) followed by $3 \times 3$ convolution and AdaIN residual blocks.

### 7. Location Where Phase 6E Source Information Entered
- In Phase 6E, $I_{s \to t}^{\text{aligned}}$ was encoded by a separate `AlignedSourceEncoder` into $F_{\text{src}} \in \mathbb{R}^{B \times 512 \times 8 \times 8}$.
- It was modulated by a geometric confidence map $C_8$ and injected via a gated residual:
  $$F_{\text{fused}} = F_{\text{tgt}} + \text{Gate} \odot \Delta_{\text{src}} \quad \in \mathbb{R}^{B \times 512 \times 8 \times 8}$$
- This injection occurred **strictly at the $8 \times 8$ bottleneck**, before the decoder.

### 8. Why That Information Was Lost
1. **Extreme Spatial Downsampling Factor (16×)**:
   - Compressing a $128 \times 128$ face into $8 \times 8$ reduces spatial dimensionality by $256\times$ ($16,384 \to 64$ spatial locations).
   - High-frequency identity cues (pupil edges, iris texture, eyelid fold, nose bridge sharpness, lip fissures, skin pores) cannot survive in an $8 \times 8$ grid.
2. **Decoder AdaIN Overwrite**:
   - The decoder applies AdaIN at resolutions $8 \times 8, 16 \times 16, 32 \times 32, 64 \times 64, 128 \times 128$.
   - Each AdaIN block re-normalizes the feature map (stripping its spatial variance) and rescales it using global vector $z_{\text{id}}$. Any residual spatial texture injected at $8 \times 8$ is smoothed out during upsampling.
3. **Deconvolution / Nearest-Neighbor Grid Artifacts**:
   - Forcing the decoder to invent spatial texture from an $8 \times 8$ latent with strong loss gradients creates horizontal and vertical phase ringing (raster striations).

---

## 3. Phase 6F Diagnostic Architectures

To determine whether the information bottleneck is architectural, Phase 6F evaluates three distinct autoencoding conditions on the **exact same 6D aligned source image**:

### Condition A: Direct Bottleneck Autoencoder (No Skips, Single Bottleneck)
- **Encoder**:
  - $128 \times 128 \to 64 \times 64$: `Conv2d(3, 64, 4, 2, 1)` $\to$ `InstanceNorm(64)` $\to$ `LeakyReLU(0.2)`
  - $64 \times 64 \to 32 \times 32$: `Conv2d(64, 128, 4, 2, 1)` $\to$ `InstanceNorm(128)` $\to$ `LeakyReLU(0.2)`
  - $32 \times 32 \to 16 \times 16$: `Conv2d(128, 256, 4, 2, 1)` $\to$ `InstanceNorm(256)` $\to$ `LeakyReLU(0.2)`
  - $16 \times 16 \to 8 \times 8$: `Conv2d(256, 512, 4, 2, 1)` $\to$ `InstanceNorm(512)` $\to$ `LeakyReLU(0.2)`
- **Bottleneck Latent**: $Z \in \mathbb{R}^{B \times 512 \times 8 \times 8}$.
- **Decoder**:
  - $8 \times 8 \to 16 \times 16$: `Bilinear(scale=2)` $\to$ `Conv2d(512, 256, 3, 1, 1)` $\to$ `InstanceNorm(256)` $\to$ `LeakyReLU(0.2)`
  - $16 \times 16 \to 32 \times 32$: `Bilinear(scale=2)` $\to$ `Conv2d(256, 128, 3, 1, 1)` $\to$ `InstanceNorm(128)` $\to$ `LeakyReLU(0.2)`
  - $32 \times 32 \to 64 \times 64$: `Bilinear(scale=2)` $\to$ `Conv2d(128, 64, 3, 1, 1)` $\to$ `InstanceNorm(64)` $\to$ `LeakyReLU(0.2)`
  - $64 \times 64 \to 128 \times 128$: `Bilinear(scale=2)` $\to$ `Conv2d(64, 64, 3, 1, 1)` $\to$ `InstanceNorm(64)` $\to$ `LeakyReLU(0.2)`
  - Output Head: `Conv2d(64, 3, 3, 1, 1)` $\to$ `Tanh()`.
- Tests the baseline capacity of a pure $8 \times 8$ latent bottleneck.

### Condition B: Multi-Scale Skip Autoencoder (U-Net Style)
- **Encoder**: Same hierarchical stages as Condition A, but preserves intermediate feature maps:
  - $E_1 \in \mathbb{R}^{B \times 64 \times 64 \times 64}$
  - $E_2 \in \mathbb{R}^{B \times 128 \times 32 \times 32}$
  - $E_3 \in \mathbb{R}^{B \times 256 \times 16 \times 16}$
  - $E_4 = Z \in \mathbb{R}^{B \times 512 \times 8 \times 8}$
- **Decoder with Multi-Scale Skip Connections**:
  - $8 \times 8 \to 16 \times 16$: `Bilinear(scale=2)` $\to$ Concat with $E_3$ ($256 + 256 = 512$ channels) $\to$ `Conv2d(512, 256, 3, 1, 1)` $\to$ `InstanceNorm(256)` $\to$ `LeakyReLU(0.2)`
  - $16 \times 16 \to 32 \times 32$: `Bilinear(scale=2)` $\to$ Concat with $E_2$ ($256 + 128 = 384$ channels) $\to$ `Conv2d(384, 128, 3, 1, 1)` $\to$ `InstanceNorm(128)` $\to$ `LeakyReLU(0.2)`
  - $32 \times 32 \to 64 \times 64$: `Bilinear(scale=2)` $\to$ Concat with $E_1$ ($128 + 64 = 192$ channels) $\to$ `Conv2d(192, 64, 3, 1, 1)` $\to$ `InstanceNorm(64)` $\to$ `LeakyReLU(0.2)`
  - $64 \times 64 \to 128 \times 128$: `Bilinear(scale=2)` $\to$ `Conv2d(64, 64, 3, 1, 1)` $\to$ `InstanceNorm(64)` $\to$ `LeakyReLU(0.2)`
  - Output Head: `Conv2d(64, 3, 3, 1, 1)` $\to$ `Tanh()`.
- Tests whether preserving multi-scale spatial representations eliminates information loss.

### Condition C: Phase 6E-Style Bottleneck Architecture
- Encodes $I_{s \to t}^{\text{aligned}}$ to $8 \times 8$ latent.
- Passes through 4 residual blocks at $8 \times 8$ (matching the Phase 6E bottleneck residual structure).
- Decodes without skips using nearest-neighbor interpolation (replicating 6E decoder upsampling).
- Tests whether the specific residual bottleneck of Phase 6E is responsible for feature degradation and striations.

---

## 4. Tensor Shapes Table

| Module / Operation | Condition A Shape | Condition B Shape (U-Net) | Condition C Shape (6E-Style) |
| :--- | :---: | :---: | :---: |
| **Input ($I_{s \to t}^{\text{aligned}}$)** | $[B, 3, 128, 128]$ | $[B, 3, 128, 128]$ | $[B, 3, 128, 128]$ |
| **Encoder Stage 1 ($64 \times 64$)** | $[B, 64, 64, 64]$ | $[B, 64, 64, 64]$ (Skip $E_1$) | $[B, 64, 64, 64]$ |
| **Encoder Stage 2 ($32 \times 32$)** | $[B, 128, 32, 32]$ | $[B, 128, 32, 32]$ (Skip $E_2$) | $[B, 128, 32, 32]$ |
| **Encoder Stage 3 ($16 \times 16$)** | $[B, 256, 16, 16]$ | $[B, 256, 16, 16]$ (Skip $E_3$) | $[B, 256, 16, 16]$ |
| **Encoder Stage 4 ($8 \times 8$)** | $[B, 512, 8, 8]$ | $[B, 512, 8, 8]$ ($E_4$) | $[B, 512, 8, 8]$ |
| **Bottleneck Processing** | Direct | Direct | 4 ResBlocks at $[B, 512, 8, 8]$ |
| **Decoder Up 1 ($16 \times 16$)** | $[B, 256, 16, 16]$ | $[B, 256, 16, 16]$ (concat w/ $E_3$) | $[B, 256, 16, 16]$ (Nearest) |
| **Decoder Up 2 ($32 \times 32$)** | $[B, 128, 32, 32]$ | $[B, 128, 32, 32]$ (concat w/ $E_2$) | $[B, 128, 32, 32]$ (Nearest) |
| **Decoder Up 3 ($64 \times 64$)** | $[B, 64, 64, 64]$ | $[B, 64, 64, 64]$ (concat w/ $E_1$) | $[B, 64, 64, 64]$ (Nearest) |
| **Decoder Up 4 ($128 \times 128$)** | $[B, 64, 128, 128]$ | $[B, 64, 128, 128]$ | $[B, 64, 128, 128]$ (Nearest) |
| **Reconstruction Output ($\hat{I}$)**| $[B, 3, 128, 128]$ | $[B, 3, 128, 128]$ | $[B, 3, 128, 128]$ |

---

## 5. Experimental Protocol & Metrics
- **Optimizer**: Adam ($\text{lr} = 5 \times 10^{-4}, \beta = (0.9, 0.999)$), AMP on CUDA.
- **Duration**: Exactly **300 optimizer steps** per condition.
- **Validation**: Every 50 steps on a fixed set of 10 held-out validation pairs.
- **Metrics Evaluated**:
  1. Total L1 Reconstruction Loss: $\| \hat{I} - I \|_1$
  2. Face Region L1 (inside target face oval mask)
  3. Background L1 (outside face mask)
  4. PSNR ($20 \log_{10}(2.0 / \text{RMSE})$)
  5. SSIM (Structural Similarity Index)
  6. Edge/Gradient Error: L1 error of horizontal & vertical finite differences
  7. ArcFace Similarity Retention $R = \cos(z(\hat{I}), z(I))$
  8. MediaPipe Landmark Re-Detection & Reprojection Error
  9. Horizontal vs Vertical Gradient Ratio (Frequency artifact index)
- **Multi-Scale Ablation (Condition B)**:
  - B1: Full skips ($64 + 32 + 16$)
  - B2: Skip 64 disabled
  - B3: Skip 32 disabled
  - B4: Skip 16 disabled
