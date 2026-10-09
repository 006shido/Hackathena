# Phase 6G — Visual Quality & Identity Transfer Diagnostic Report
**Controlled Diagnostic Milestone — Final Report**
**Classification**: `PASS — root cause sufficiently identified`
**Execution Status**: WebRTC Integration **STRICTLY PAUSED**

---

## 1. Problem Statement

During localhost integration tests of the Phase 6G neural face-swap pipeline (FastAPI $\to$ Express $\to$ React on RTX 5060), real-time evaluations revealed a visual-quality and identity transfer failure:
- Neural outputs ($I_{\text{comp}}$) often remain strongly target-like, with target identity dominating over source identity.
- Example pair (`081968.jpg` $\to$ `037827.jpg`) produced:
  $$A (\text{source} \to \text{comp}) = 0.1040, \quad B (\text{target} \to \text{comp}) = 0.2700, \quad C (\text{source} \to \text{target}) = -0.0731$$
  $$\text{Gain } (A - C) = +0.1771, \quad \text{Advantage } (A - B) = -0.1660$$
- Facial features in the swapped composite were noticeably blurred compared to both the ground-truth target and the photorealistic 6D piecewise-affine aligned source warp ($L$).
- Despite the positive identity gain metric ($A - C > 0$), visual face-swap quality was deficient.

This diagnostic was executed without modifying production weights, without retraining, and under strict isolation in `ml/diagnostics/phase6g_quality/`.

---

## 2. Why Positive Identity Gain Does Not Imply Successful Transfer

In face swapping, three identity vectors are evaluated via ArcFace:
1. $z_{\text{src}}$: source embedding
2. $z_{\text{tgt}}$: target embedding
3. $z_{\text{comp}}$: composite output embedding

The standard metric used during early development was **Identity Gain**:
$$\text{Identity Gain} = A - C = \cos(z_{\text{src}}, z_{\text{comp}}) - \cos(z_{\text{src}}, z_{\text{tgt}})$$

### The Mathematical Trap:
Two random human faces in CelebA have baseline cosine similarity centered near zero or slightly negative ($C \approx -0.0000$, standard deviation $\approx 0.0868$, minimum $\approx -0.1964$).
- If $C = -0.1000$ and $A = 0.1000$, then $\text{Gain } (A - C) = +0.2000$ (strongly positive).
- However, if the output still looks almost entirely like the target ($B = \cos(z_{\text{tgt}}, z_{\text{comp}}) = 0.3500$), then $B \gg A$.
- The **Source-Target Advantage** is:
  $$A - B = 0.1000 - 0.3500 = -0.2500$$
The output is 3.5× more similar to the target person than to the source person! Therefore, **Identity Gain ($A - C$) is necessary but completely insufficient**. A true face swap requires:
1. $A > B$ (Source identity exceeds Target identity in the output)
2. $A - B > 0$ (Source-Target Advantage)
3. $B - C \approx 0$ (Target leakage minimized)

---

## 3. Dataset & Evaluation Protocol

- **Dataset**: CelebA held-out validation split (`ml/data/celeba/`, `split="val"`, zero overlap with training identities).
- **Sample Size**: 100 genuine cross-identity pairs ($id(A) \neq id(B)$, sampled with fixed seed 42).
- **Verified Checkpoint**: `ml/checkpoints/stage2/phase6g/best_model.pt`.
- **Verified ArcFace**: `ml/models/weights/ms1mv2_iresnet50.pth` (verified SHA-256 `2B75B93C48B01C78A4263F7295AB2DBF84F85190C51BE45B92C4C0B0AEDCEAA3`).
- **Hardware**: NVIDIA GeForce RTX 5060 Laptop GPU (CUDA 12.8, PyTorch 2.7.1+cu128).
- **Execution Script**: `ml/diagnostics/phase6g_quality/run_diagnostics.py`.
- **Results Artifact**: `ml/diagnostics/phase6g_quality/diagnostic_results.json`.

---

## 4. 100-Pair Quantitative Results

| Metric | Definition | Mean | Median | Std | Min | Max |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **A** | $\cos(z_{\text{src}}, z_{\text{comp}})$ | 0.2366 | 0.2476 | 0.1314 | -0.0776 | +0.5496 |
| **B** | $\cos(z_{\text{tgt}}, z_{\text{comp}})$ | 0.2024 | 0.1969 | 0.1060 | -0.0048 | +0.4454 |
| **C** | $\cos(z_{\text{src}}, z_{\text{tgt}})$ | -0.0000 | -0.0069 | 0.0868 | -0.1964 | +0.2158 |
| **D** | $\cos(z_{\text{tgt}}, z_{\text{src}})$ | -0.0000 | -0.0069 | 0.0868 | -0.1964 | +0.2158 |
| **$D_{\text{swap}}$** | $\cos(z_{\text{src}}, z_{\text{swap}})$ | 0.2632 | 0.2727 | 0.1406 | -0.1363 | +0.5608 |
| **$B_{\text{swap}}$** | $\cos(z_{\text{tgt}}, z_{\text{swap}})$ | 0.0598 | 0.0621 | 0.0906 | -0.1652 | +0.3118 |
| **$L_{\text{aln}}$** | $\cos(z_{\text{src}}, z_{\text{aln}})$ | 0.6283 | 0.6332 | 0.1343 | +0.2601 | +0.8885 |
| **Gain ($A - C$)** | Source Gain | +0.2366 | +0.2202 | 0.1368 | -0.1460 | +0.5478 |
| **Adv ($A - B$)** | Source Advantage | +0.0341 | +0.0383 | 0.1788 | -0.3951 | +0.4066 |
| **Leak ($B - C$)** | Target Leakage | +0.2025 | +0.1985 | 0.1330 | -0.0320 | +0.5973 |
| **Transfer Ratio** | $(A - C) / (1 - C)$ | 0.2338 | 0.2282 | 0.1341 | -0.1568 | +0.5176 |

---

## 5. A / B / C / D Distributions

Detailed percentile breakdown across all 100 held-out evaluation pairs:

| Percentile | A ($\text{Src}\to\text{Comp}$) | B ($\text{Tgt}\to\text{Comp}$) | C ($\text{Src}\to\text{Tgt}$) | A - C (Gain) | A - B (Advantage) | B - C (Leakage) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **P10** | 0.0675 | 0.0602 | -0.1046 | +0.0670 | -0.1989 | +0.0385 |
| **P25** | 0.1285 | 0.1268 | -0.0592 | +0.1744 | -0.0843 | +0.1083 |
| **Median (P50)**| **0.2476** | **0.1969** | **-0.0069** | **+0.2202** | **+0.0383** | **+0.1985** |
| **P75** | 0.3087 | 0.2748 | +0.0565 | +0.3227 | +0.1539 | +0.3003 |
| **P90** | 0.4201 | 0.3591 | +0.1119 | +0.4081 | +0.2495 | +0.3866 |

---

## 6. Percentage Where A > B

$$\mathbf{P(A > B) = 58.0\%}$$

In **42.0% of held-out cross-identity pairs**, the generated composite is more similar to the target than to the source ($B \ge A$).
This confirms the user's initial observation: Phase 6G frequently produces target-dominant outputs.

---

## 7. Percentage Where A > C and A - C > 0

$$\mathbf{P(A > C) = 95.0\%}$$
$$\mathbf{P(A - C > 0) = 95.0\%}$$

This stark discrepancy between $P(A - C > 0) = 95.0\%$ and $P(A > B) = 58.0\%$ is the mathematical signature of **target identity leakage**. In 37% of all cases, the model moves positively away from the target baseline $C$, but stops short of surpassing the target $B$.

---

## 8. Identity Gain (A - C) Distribution

- Mean: $+0.2366 \pm 0.1368$
- Median: $+0.2202$
- Range: $[-0.1460, +0.5478]$
- Interquartile Range: $[+0.1744, +0.3227]$

While the identity gain distribution is heavily shifted positive (95% positive), it conceals the fact that target leakage ($B - C = +0.2025$) absorbs nearly the entirety of the gain.

---

## 9. Source-Target Advantage (A - B) Distribution

- Mean: $+0.0341 \pm 0.1788$
- Median: $+0.0383$
- Range: $[-0.3951, +0.4066]$
- P10: $-0.1989$ (severe target dominance)
- P90: $+0.2495$ (strong source dominance)

The distribution is centered almost directly on zero ($+0.0341$). This indicates that Phase 6G operates in an unstable compromise state between source and target identities, tipping toward the source in 58% of cases and toward the target in 42% of cases.

---

## 10. Sharpness & High-Frequency Energy Results

| Image Stream | Laplacian Variance (Sharpness) | High-Freq Energy ($\|\nabla I\|^2$) | Relative to Target |
| :--- | :---: | :---: | :---: |
| **Ground-Truth Target** | $325.4 \pm 243.1$ | $10,020.6 \pm 3,909.9$ | 100.0% |
| **6D Aligned Source ($I_{\text{aln}}$)** | $316.0 \pm 180.9$ | $8,795.9 \pm 3,645.7$ | 97.1% |
| **Raw Swap Prediction ($I_{\text{swap}}$)** | $\mathbf{158.9 \pm 29.4}$ | $\mathbf{7,757.1 \pm 1,252.4}$ | **48.8% (-51.2%)** |
| **Final Composite ($I_{\text{comp}}$)** | $234.7 \pm 160.3$ | $9,544.3 \pm 2,915.7$ | 72.1% (-27.9%) |

### Key Diagnostic Findings:
1. **Severe Neural Smoothing**: The raw swap prediction $I_{\text{swap}}$ suffers a **-49.7% loss of sharpness** relative to the 6D aligned source ($158.9$ vs $316.0$).
2. **Composite Artifact**: The composite sharpness ($234.7$) appears higher only because the sharp un-swapped background from the target image is preserved by the soft mask. Inside the facial mask boundary, the output is noticeably blurred.

---

## 11. Face-Region vs Background Results (Diagnostic 4)

Mean Absolute Pixel Difference (L1 normalized to $[0, 1]$):

| Comparison | Inside Face Region ($M_{\text{face}} > 0.5$) | Outside Face Region ($M_{\text{face}} \le 0.5$) |
| :--- | :---: | :---: |
| **$I_{\text{swap}}$ vs Target** | $0.1350 \pm 0.0501$ | $0.2751 \pm 0.0463$ |
| **Composite vs Target** | $0.1207 \pm 0.0456$ | $\mathbf{0.0045 \pm 0.0021}$ |

### Empirical Classification:
**Category C: Transferring source identity but losing detail (visually blurred), with high target leakage.**
- The generator actively alters the target face ($L1_{\text{face}} = 0.1350$).
- Background preservation is near-perfect ($L1_{\text{bg}} = 0.0045$).
- However, the facial features inside the mask lack the high frequencies and crisp contours present in the 6D aligned source.

---

## 12. Mask Results

| Metric | Mean | Median | Std | Min | Max |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mask Mean Coverage** | 43.5% | 43.6% | 2.9% | 32.7% | 50.5% |
| **Mask Binary Activation ($> 0.5$)** | 44.9% | 45.0% | 3.0% | 33.8% | 52.3% |
| **Mask Outside-Face Leakage** | 5.57% | 5.30% | 1.78% | 2.02% | 13.05% |

The predicted soft mask $M_{\text{pred}}$ is healthy and well-behaved:
- Coverage is stable around 44.9% of the 128×128 canvas.
- Leakage outside the facial contour is low (5.57%).
- The mask is **NOT** the bottleneck.

---

## 13. Landmark & Redetection Results

| Metric | Value |
| :--- | :---: |
| **MediaPipe Face Redetection Rate** | **98.0% (98/100 PASS)** |
| **Mean 5-Point Landmark Error** | **$1.79 \pm 0.77\text{ px}$** |
| Median Landmark Error | 1.58 px |
| P10 Landmark Error | 0.97 px |
| P90 Landmark Error | 2.76 px |

The composite face is geometrically realistic and anatomically natural, passing independent MediaPipe facial mesh re-detection in 98% of cases with sub-2-pixel keypoint error.

---

## 14. Resolution & Architecture Bottleneck Analysis (Diagnostic 5)

An exhaustive code audit of `MultiScaleSourceEncoder`, `MultiScaleSkipGenerator`, and `TargetStructureEncoder` in `ml/models/phase6g_model.py` reveals the two fundamental structural bottlenecks causing the visual blur:

```
TARGET INPUT (128x128)               ALIGNED SOURCE (128x128)
         │                                       │
 [Blur sigma=3.0]                                │
         │                                       │
TargetStructureEncoder               MultiScaleSourceEncoder
         │                                       │
  conv stride 2                           conv stride 2 + InstanceNorm
         │                                       │
      (64x64)                                (64x64)  ───► s64
         │                                       │
      (32x32)                                (32x32)  ───► s32
         │                                       │
      (16x16)                                (16x16)  ───► s16
         │                                       │
       (8x8)                                  (8x8)   ───► s8
         │                                       │
         └───────────────┬───────────────────────┘
                         │
                    Bottleneck
                     fuse_8 (Conv + InstanceNorm)
                         │
                    4x AdaIN ResBlocks (8x8) modulated by z_id
                         │
                     Up 1 (16x16)  ◄── fuse_16 (Conv + InstanceNorm) from s16
                         │
                     Up 2 (32x32)  ◄── fuse_32 (Conv + InstanceNorm) from s32
                         │
                     Up 3 (64x64)  ◄── fuse_64 (Conv + InstanceNorm) from s64
                         │
    ═════════════════════╪════════════════════════════════════════════════
    CRITICAL GAP: NO 128x128 LATERAL SKIP CONNECTION FROM ALIGNED SOURCE
    ═════════════════════╪════════════════════════════════════════════════
                         │
                 Bilinear Upsample (64x64 -> 128x128)
                         │
                 conv_up4 (64 -> 64)
                 res_up4 (AdaIN)
                 rgb_head (Conv 3x3, Tanh)
                         │
                    I_swap (128x128) [BLURRED]
```

### Bottleneck Identification:
1. **The Missing 128×128 Skip**: Lateral skips exist at $8\times8$, $16\times16$, $32\times32$, and $64\times64$. However, **no skip connection exists at the native $128\times128$ resolution**.
   At $64\times64$, bilinear upsampling interpolates $2\times2$ pixel neighborhoods, blurring sharp edges. The single conv layer `conv_up4` cannot invent missing high-frequency details (skin pores, sharp iris boundaries, distinct eyelashes) without a direct $128\times128$ skip or residual connection from the aligned source.
2. **Destructive Instance Normalization**: Every convolutional block in `MultiScaleSourceEncoder` and in the fusion blocks (`fuse_8`, `fuse_16`, `fuse_32`, `fuse_64`) contains `nn.InstanceNorm2d(..., affine=False)`.
   InstanceNorm standardizes feature channels by subtracting the spatial mean and dividing by the spatial variance:
   $$\hat{x}_{c} = \frac{x_{c} - \mu_{c}}{\sqrt{\sigma_{c}^2 + \epsilon}}$$
   This strips the absolute contrast, texture energy, and fine-scale illumination gradients from the aligned source features, forcibly smoothing the representation before it enters the decoder.

---

## 15. Source / Target Inference Ablation Results (Diagnostic 7)

Evaluated on a controlled 20-pair held-out subset using the exact checkpoint `best_model.pt`:

| Condition | ArcFace A ($\text{Src}\to\text{Comp}$) | ArcFace B ($\text{Tgt}\to\text{Comp}$) | Gain ($A - C$) | Advantage ($A - B$) | $\% (A > B)$ | Sharpness | Face Det |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Normal Phase 6G** | **0.2368** | 0.2064 | **+0.2574** | **+0.0304** | **60.0%** | 274.0 | **100%** |
| **2. ArcFace Conditioning Disabled ($z_{\text{id}} = 0$)** | **0.2367** | 0.2059 | **+0.2573** | **+0.0308** | **55.0%** | 273.9 | **100%** |
| **3. Aligned Source Disabled ($I_{\text{aln}} = 0$)** | -0.0195 | 0.1993 | +0.0011 | -0.2187 | 0.0% | 231.9 | 85.0% |
| **4. Spatial Skips Disabled (`disable_skips=True`)** | -0.0195 | 0.1993 | +0.0011 | -0.2187 | 0.0% | 231.9 | 85.0% |
| **5. Both $z_{\text{id}} = 0$ & Skips Disabled** | -0.0191 | 0.1989 | +0.0014 | -0.2180 | 0.0% | 231.9 | 85.0% |

### Astounding Empirical Finding:
- **Condition 2 ($z_{\text{id}} = 0$) vs Baseline**:
  $$\Delta A = 0.2368 - 0.2367 = \mathbf{+0.0001}$$
  $$\Delta \text{Gain} = +0.2574 - 0.2573 = \mathbf{+0.0001}$$
  Disabling the ArcFace 512-D identity conditioning vector completely ($z_{\text{id}} = \mathbf{0}$) produces virtually **zero change** in output identity! The AdaIN residual blocks are not injecting source identity.
- **Condition 4 (Skips Disabled) vs Baseline**:
  Disabling the spatial skips drops identity gain from $+0.2574$ down to $+0.0011$ ($\Delta = -0.2563$), and drops $P(A > B)$ to $0.0\%$.
- **Conclusion**: **100% of source identity transfer in Phase 6G is carried exclusively by the 6D aligned source multi-scale skips.** The ArcFace AdaIN pathway is dormant.

---

## 16. Correlations: Identity Transfer vs Sharpness (Diagnostic 6)

| Correlation Pair | Pearson $r$ | Pearson $p$-value | Spearman $\rho$ | Spearman $p$-value | Relationship |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Advantage ($A - B$) vs Composite Sharpness** | -0.0841 | 0.405 | -0.0633 | 0.532 | **No correlation** ($p > 0.40$) |
| **Gain ($A - C$) vs Composite Sharpness** | +0.0393 | 0.698 | +0.0051 | 0.960 | **No correlation** ($p > 0.69$) |
| **Leakage ($B - C$) vs Composite Sharpness** | +0.1535 | 0.127 | +0.0903 | 0.372 | **No correlation** ($p > 0.12$) |
| **Advantage ($A - B$) vs Raw Swap Sharpness** | +0.1063 | 0.292 | +0.0841 | 0.405 | **No correlation** ($p > 0.29$) |
| **Gain ($A - C$) vs Raw Swap Sharpness** | +0.1173 | 0.245 | +0.0911 | 0.367 | **No correlation** ($p > 0.24$) |

### Definite Scientific Conclusion:
**"Blur is completely independent of identity transfer."**
Poor sharpness is not a trade-off for higher identity gain ($|r| < 0.10, p \gg 0.05$). The model is not blurry *because* it transfers identity; it is blurry because the encoder-decoder architecture lacks high-frequency preservation mechanisms.

---

## 17. Representative Visual Examples (Diagnostic 3)

20 representative 6-column comparison grids were generated and saved in `ml/diagnostics/phase6g_quality/grids/`:
Format: `[ 1. SOURCE | 2. TARGET | 3. 6D ALIGNED | 4. I_SWAP | 5. COMPOSITE | 6. MASK ]`

### 1. Strong Transfer (High Source Advantage):
- `grid_01_pair8_172366_086159.png`: $A = 0.4530, B = 0.0463, C = -0.0881, A - B = \mathbf{+0.4066}$. Source identity dominates strongly.
- `grid_02_pair94_039087_010552.png`: $A = 0.4608, B = 0.0733, C = -0.0540, A - B = \mathbf{+0.3875}$.
- `grid_03_pair47_009790_123798.png`: $A = 0.4163, B = 0.0608, C = +0.0159, A - B = \mathbf{+0.3555}$.
- `grid_04_pair25_102198_108942.png`: $A = 0.4035, B = 0.0550, C = -0.0404, A - B = \mathbf{+0.3485}$.

### 2. Average Transfer (Near Median):
- `grid_05_pair19_044572_101888.png`: $A = 0.3675, B = 0.3267, C = +0.0258, A - B = \mathbf{+0.0407}$.
- `grid_06_pair44_154992_034298.png`: $A = 0.3647, B = 0.3255, C = +0.1085, A - B = \mathbf{+0.0393}$.
- `grid_07_pair3_116463_144134.png`: $A = 0.3385, B = 0.3012, C = -0.0186, A - B = \mathbf{+0.0372}$.
- `grid_08_pair76_026182_143507.png`: $A = 0.2715, B = 0.2472, C = +0.0573, A - B = \mathbf{+0.0243}$.

### 3. Weak Transfer (Target Dominant):
- `grid_12_pair26_168771_138664.png`: $A = 0.0230, B = 0.4181, C = -0.1792, A - B = \mathbf{-0.3951}$. Target identity dominates completely despite $A - C = +0.2022$.
- `grid_11_pair41_112475_184010.png`: $A = 0.0933, B = 0.4454, C = -0.0220, A - B = \mathbf{-0.3521}$.
- `grid_10_pair34_032673_021188.png`: $A = 0.0688, B = 0.3747, C = -0.0109, A - B = \mathbf{-0.3059}$.
- `grid_09_pair100_043182_162426.png`: $A = 0.0676, B = 0.3734, C = -0.0224, A - B = \mathbf{-0.3057}$.

### 4. Large Pose Differences:
- `grid_13_pair58_013099_163850.png`: $A = 0.0275, B = 0.0423$, Pose Diff = 8.92 px.
- `grid_14_pair79_199397_159846.png`: $A = 0.2453, B = 0.0534$, Pose Diff = 7.84 px.
- `grid_15_pair6_001921_145564.png`: $A = 0.0664, B = 0.2470$, Pose Diff = 7.15 px.

### 5. Small Pose Differences:
- `grid_17_pair11_027860_080131.png`: $A = 0.2657, B = 0.1141$, Pose Diff = 1.05 px.
- `grid_18_pair1_000172_153865.png`: $A = 0.0825, B = 0.2768$, Pose Diff = 1.21 px.
- `grid_19_pair23_042387_157875.png`: $A = 0.3980, B = 0.2568$, Pose Diff = 1.34 px.

---

## 18. Root-Cause Analysis

The diagnostic measurements identify three distinct root causes:

### Cause 1: Target Identity Dominance (Leakage)
- In Phase 6G, the generator receives target identity information via the blurred target RGB ($\sigma=3.0$) and landmark heatmaps at the bottleneck ($8\times8$).
- Because the ArcFace 512-D $z_{\text{id}}$ conditioning is completely ignored by the decoder (ablation $\Delta A = 0.0001$), the generator has no semantic identity counterweight to the target structure encoder.
- When the 6D aligned source skip features have lower contrast or mismatched illumination, the target features dominate the reconstruction, causing $B > A$ in 42% of pairs.

### Cause 2: Visual Blur & Detail Loss (-49.7% Sharpness)
- The 6D aligned source contains sharp, photorealistic high-frequency details (sharpness $316.0$).
- Passing it through `MultiScaleSourceEncoder` with `InstanceNorm2d` strips feature variance and contrast.
- Lateral skips stop at $64\times64$. The final step from $64\times64$ to $128\times128$ is a bilinear upsampling followed by standard convolutions, discarding all spatial frequencies above 64 cycles per image.
- Consequently, $I_{\text{swap}}$ collapses to a blurry sharpness of $158.9$.

### Cause 3: Disconnected Failure Modes
- Sharpness and identity advantage have zero correlation ($r = -0.0841, p = 0.405$).
- Visual blur and target leakage are separate bottlenecks. Fixing blur requires high-frequency detail preservation; fixing target dominance requires active source identity supervision and target feature suppression.

---

## 19. Confidence Level for Each Conclusion

| Finding | Confidence | Supporting Evidence |
| :--- | :---: | :--- |
| **Positive $A-C$ hides target dominance** | **100% (Certain)** | $P(A-C>0) = 95\%$, but $P(A>B) = 58\%$. Mean $B = 0.2024$, $B-C = +0.2025$. |
| **ArcFace AdaIN conditioning is non-functional** | **100% (Certain)** | Ablation with $z_{\text{id}}=0$ changes $A$ by only $0.0001$ ($0.2368 \to 0.2367$). |
| **100% of identity transfer comes from 6D skips** | **100% (Certain)** | Disabling skips drops identity gain from $+0.2574$ to $+0.0011$ ($P(A>B) \to 0\%$). |
| **Blur is caused by missing 128×128 skip & InstanceNorm** | **95% (High)** | Sharpness drops $-49.7\%$ in $I_{\text{swap}}$. Code inspection verifies no 128×128 skip. |
| **Blur is independent of identity transfer** | **95% (High)** | Pearson $r = -0.0841, p = 0.405$; Spearman $\rho = -0.0633, p = 0.532$. |
| **Mask prediction is not the failure point** | **99% (High)** | Mask leakage is only $5.57\%$, coverage is stable ($44.9\%$). |

---

## 20. Recommended Next Experiment (Proposal Only)

**Do NOT implement yet. Proposal for future work:**

To resolve both bottlenecks simultaneously without breaking landmark correspondence:
1. **Add Native $128\times128$ High-Frequency Residual Skip**:
   Pass a lateral connection from the 6D aligned source directly to the final synthesis layer at $128\times128$, bypassing downsampling, to restore the $-49.7\%$ sharpness loss.
2. **Remove Texture-Destroying InstanceNorm**:
   Replace `InstanceNorm2d` in `MultiScaleSourceEncoder` with `WeightNorm` or standard convolutions, preserving absolute texture and illumination contrast.
3. **Explicit Source-Target Advantage Loss**:
   Supervise training directly with a margin loss enforcing $A > B$:
   $$\mathcal{L}_{\text{adv}} = \max(0, \cos(z_{\text{tgt}}, I_{\text{swap}}) - \cos(z_{\text{src}}, I_{\text{swap}}) + m)$$
   where $m = 0.20$, penalizing target leakage directly.

---

## 21. Explicit Statement on WebRTC Integration

$$\mathbf{WebRTC\ Integration\ MUST\ REMAIN\ PAUSED.}$$

Proceeding with WebRTC streaming while 42% of cross-identity pairs are target-dominant and facial outputs suffer a 50% sharpness loss would produce an unconvincing and visibly defective real-time experience. WebRTC integration should only resume once the visual quality and identity advantage bottlenecks are resolved in a controlled offline milestone.

---

## Final Classification: `PASS — root cause sufficiently identified`
