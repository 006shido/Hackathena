# Phase 6C: Architecture & Information Bottleneck Analysis

## 1. Objective & Scope
This analysis investigates whether the current Stage 2 face-swap architecture fundamentally loses too much source visual identity information when compressing the source image solely into a 512-D ArcFace recognition embedding.

---

## 2. Findings on Current Information Pathways

### A. How the Source Image Becomes the 512-D ArcFace Embedding
1. **Input**: Source image $I_s \in \mathbb{R}^{B \times 3 \times 128 \times 128}$ in range $[-1.0, 1.0]$.
2. **Preprocessing & Alignment**:
   - Resampled via bilinear interpolation to canonical resolution: $(128, 128) \to (112, 112)$.
3. **Backbone Flow** (`iresnet50`):
   - Input layer: $3 \times 3$ conv $\to 64$ channels.
   - 4 residual stages with $(3, 4, 14, 3)$ modified bottleneck blocks:
     - Stage 1: $64$ channels, $56 \times 56$
     - Stage 2: $128$ channels, $28 \times 28$
     - Stage 3: $256$ channels, $14 \times 14$
     - Stage 4: $512$ channels, $7 \times 7$
   - Dense projection: Flatten $(512 \times 7 \times 7 = 25,088) \to \text{Linear} \to \text{BatchNorm1d} \to 512\text{-D}$.
   - L2 Normalization: $z_{\text{id}} = \frac{e}{\|e\|_2 + \epsilon}$.
4. **Conclusion**: ArcFace collapses all 2D spatial dimensions ($7 \times 7$) into a single 512-dimensional vector on the unit hypersphere $\mathbb{S}^{511}$.

---

### B. Where $z_{\text{id}}$ Enters the Generator
- In `IdentityConditionedGenerator`:
  - **Bottleneck Blocks**: Applied across 4 residual blocks at $8 \times 8$ resolution (512 channels) via `AdaIN2d`.
  - **Upsampling Blocks**: Applied across 4 upsampling residual blocks:
    - Stage 1: $16 \times 16$ (256 channels)
    - Stage 2: $32 \times 32$ (128 channels)
    - Stage 3: $64 \times 64$ (64 channels)
    - Stage 4: $128 \times 128$ (64 channels)
- **Modulation Mechanism**:
  $$\text{AdaIN}(F, z_{\text{id}}) = (1 + \gamma(z_{\text{id}})) \cdot \text{InstanceNorm}(F) + \beta(z_{\text{id}})$$
  where $(\gamma, \beta)$ are predicted by a 2-layer MLP from $z_{\text{id}}$.
- **Limitation**: $z_{\text{id}}$ applies only **spatially uniform**, channel-wise scale and bias. It cannot convey spatial variations (e.g. eye position vs mouth position).

---

### C. Which Target Features Enter the Generator
1. **Target Image**: $I_t \in \mathbb{R}^{B \times 3 \times 128 \times 128}$ filtered with Gaussian blur ($\sigma=3.0, k=9$) $\to I_{t,\text{low}}$.
2. **Target Landmark Map**: $L_t \in \mathbb{R}^{B \times 1 \times 128 \times 128}$ (dense distance field from MediaPipe 468 landmarks).
3. **Concatenation**: $X_{\text{tgt}} = [I_{t,\text{low}} \,\|\, L_t] \in \mathbb{R}^{B \times 4 \times 128 \times 128}$.
4. **Target Structure Encoder**:
   - Downsamples $X_{\text{tgt}}$ via 4 convolutional layers ($128 \to 64 \to 32 \to 16 \to 8$).
   - Yields dense spatial feature representation $F_{\text{tgt}} \in \mathbb{R}^{B \times 512 \times 8 \times 8}$.
5. **Compositing**: $I_t$ is injected directly into `SoftCompositor`:
   $$I_{\text{composite}} = M_{\text{pred}} \cdot I_{\text{swap}} + (1 - M_{\text{pred}}) \cdot I_t$$

---

### D. Information Discarded After ArcFace Encoding
Because ArcFace is optimized strictly for identity verification (cosine margin classification), it deliberately discards information unnecessary or invariant for classification:
1. **Source Spatial Coordinates & Layout**: Absolute/relative 2D landmark coordinates of source features.
2. **Local Geometric Contours**: Exact eye shape, eyelid fold structure, nose bridge width/contour, lip curvature, jaw contour.
3. **Local High-Frequency Texture**: Skin texture, micro-shading, unique pigmentation, facial hair patterns.
4. **Source Appearance Gradients**: Spatial variation in illumination, skin undertones, and specular highlights across the face.

---

### E. Assessment of Current Generator Architecture
- **Source Spatial Pathway**: **NON-EXISTENT**.
- $100\%$ of spatial feature maps throughout the entire network originate from the target image and target landmark map.
- The generator is tasked with synthesizing source facial geometry purely from a single 512-D global style vector, while constantly constrained by a dense 2D spatial feature grid derived entirely from the target.
- This creates an asymmetrical conditioning bottleneck: target spatial conditioning overwhelmingly dominates the 512-D style modulation.
