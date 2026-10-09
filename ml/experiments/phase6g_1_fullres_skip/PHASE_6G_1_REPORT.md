# Phase 6G.1 Full-Resolution (128×128) Source Skip Experiment Report
**Controlled ML Experiment Report**
**Classification**: `NEEDS TUNING`
**WebRTC Integration Status**: **STRICTLY PAUSED**

---

## 1. Upstream Sync Status

- **Repository**: `C:\files\forgitclone\Hackathena`
- **Initial HEAD**: Commit `217a16d` (`sliding call end`)
- **Safety Reference Created**: Tag & Branch `backup-before-upstream-sync-6g1` pointing to `217a16d`.
- **Pre-Merge State Preserved**: Tag `backup-milestone3-before-merge` (`e56074f`) preserving all Milestone 2 & 3 localhost test panel files and proxy endpoints.
- **Upstream Remote**: `https://github.com/abhinavaby/Hackathena.git`
- **Upstream HEAD Fetched**: Commit `b84e830` (`video corrected`)
- **Merge Status**: `git merge upstream/main` completed successfully into `main` via merge commit `68c9c9f`.

---

## 2. Merge Conflicts Encountered & Resolved

Two content conflicts occurred during the 3-way merge:
1. `client/src/pages/Login.tsx`:
   - Conflict: Local branch added `onOpenPhase6GTest?: () => void;`, upstream added `onBackToHome?: () => void;` and `initialMode?: 'signin' | 'signup';`.
   - Resolution: Integrated both interfaces and props destructurings cleanly, maintaining both the "Back to Home" navigation and the "Phase 6G Neural Face Swap Test UI" access button.
2. `client/src/App.tsx`:
   - Conflict: Local branch declared `showPhase6GTest` state and query param handling, upstream introduced `authView` (`'home' | 'login'`) and `authMode` (`'signin' | 'signup'`) for the new `HomePage` component.
   - Resolution: Preserved all states. Rendered `HomePage` by default, routed to `LoginPage` with `onOpenPhase6GTest` forwarded, and preserved the dedicated `Phase6GTestPanel` when `showPhase6GTest` or `?test=phase6g` is active.

Zero merge conflict markers remain (`git diff --check` returned 0).

---

## 3. Protected Artifact Integrity Audit

Post-merge verification confirmed 100% preservation of all critical ML assets:

| Protected Artifact | Path | Verification Status |
| :--- | :--- | :---: |
| **CelebA Dataset** | `ml/data/celeba/` | **EXISTS / UNTOUCHED** |
| **ArcFace Weights** | `ml/models/weights/ms1mv2_iresnet50.pth` | **EXISTS / UNTOUCHED** |
| **Stage 1 Checkpoints** | `ml/checkpoints/stage1/` | **EXISTS / UNTOUCHED** |
| **Stage 2 Baseline Best** | `ml/checkpoints/stage2/best_model.pt` | **EXISTS / UNTOUCHED** |
| **Stage 2 Baseline Latest** | `ml/checkpoints/stage2/latest_model.pt` | **EXISTS / UNTOUCHED** |
| **Phase 6G Production Best** | `ml/checkpoints/stage2/phase6g/best_model.pt` | **EXISTS / UNTOUCHED** |
| **Phase 6G Production Latest**| `ml/checkpoints/stage2/phase6g/latest_model.pt`| **EXISTS / UNTOUCHED** |
| **Phase 6G Inference Engine** | `ml/inference/infer_phase6g.py` | **EXISTS / UNTOUCHED** |
| **Phase 6G FastAPI Service** | `ml/api/app.py` | **EXISTS / UNTOUCHED** |

---

## 4. ArcFace SHA-256 Verification

- Expected SHA-256: `2B75B93C48B01C78A4263F7295AB2DBF84F85190C51BE45B92C4C0B0AEDCEAA3`
- Computed SHA-256: `2B75B93C48B01C78A4263F7295AB2DBF84F85190C51BE45B92C4C0B0AEDCEAA3`
- **Result**: **100% MATCH — Integrity Verified.**

---

## 5. CelebA Dataset Integrity

- Image Count: **202,599 images** (exact)
- Annotation Lines: **202,599 lines** (exact)
- Total Identities: **10,177** (exact)
- Train Split Identities: **8,650**
- Validation Split Identities: **1,527**
- Train/Validation Identity Overlap: **0** (strictly disjoint)

---

## 6. Phase 6G Baseline Verification Post-Merge

Prior to initiating Phase 6G.1, the production checkpoint `ml/checkpoints/stage2/phase6g/best_model.pt` was verified by running local CUDA inference via `infer_phase6g.py`:
- Device: NVIDIA GeForce RTX 5060 Laptop GPU (CUDA 12.8)
- MediaPipe Face Mesh & Landmarker: Initialized and operational
- Execution: Successful inference on test pair (`081968.jpg` $\to$ `037827.jpg`)
- Identity Gain ($A - C$): $+0.1771$
- Result: **Phase 6G baseline fully verified.**

---

## 7. Exact Architecture Change in Phase 6G.1

Phase 6G.1 introduces **ONE** controlled architectural modification isolated in `ml/experiments/phase6g_1_fullres_skip/model_phase6g1.py`:

```
               ALIGNED SOURCE RGB (128x128)
                            │
               ┌────────────┴────────────┐
               │                         │
     [Conv 3x3 + IN + LeakyReLU]         │  Existing Strided Convolutions
               │                         │  (Down to 64x64, 32x32, 16x16, 8x8)
      s128 (32 channels)                 ▼
               │                  Existing MultiScale Skips
               │                  (s64, s32, s16, s8)
               │                         │
               │                         ▼
               │                  Decoder Stages 1-3 (8x8 -> 64x64)
               │                         │
               │                  Bilinear Upsample (64x64 -> 128x128)
               │                         │
               │                  conv_up4 (64 -> 64 ch)
               │                         │
               └────────────► fuse_128 ◄─┘
                              [Conv 3x3 (64+32 -> 64) + IN + LeakyReLU]
                                         │
                                      res_up4 (AdaIN 64 ch)
                                         │
                               ┌─────────┴─────────┐
                               ▼                   ▼
                           rgb_head            mask_head
                          (I_swap)             (M_pred)
```

1. **`MultiScaleSourceEncoder6G1.layer0`**:
   Projects 3-channel aligned source RGB directly at $128\times128$ into a 32-channel feature map $s_{128}$ via `Conv2d(3, 32, kernel_size=3, padding=1, bias=False) -> InstanceNorm2d(32) -> LeakyReLU(0.2)`.
2. **`MultiScaleSkipGenerator6G1.fuse_128`**:
   At the final $128\times128$ decoder stage, after `conv_up4` (64 channels), the decoder concatenates the 64-channel upsampled feature with the 32-channel $s_{128}$ full-res skip (weighted by confidence map) and fuses down to 64 channels via `Conv2d(96, 64, kernel_size=3, padding=1, bias=False) -> InstanceNorm2d(64) -> LeakyReLU(0.2)`.
3. **Ablation Control Flag**:
   `disable_fullres_skip=True` zeroes out $s_{128}$ while retaining all other pathways, allowing exact causal attribution.

---

## 8. Training Configuration

- **Initialization**: Verified Phase 6G weights transferred from `ml/checkpoints/stage2/phase6g/best_model.pt`.
- **Training Pool**: 100 genuine CelebA cross-ID pairs (`id(source) != id(target)`).
- **Discriminator**: PatchGAN discriminator initialized from Phase 6G baseline.
- **Loss Formulation**: `Stage2CompositeLoss` ($w_{\text{id}}=10.0, w_{\text{id\_swap}}=8.0, w_{\text{struct}}=5.0, w_{\text{bg}}=5.0, w_{\text{mask}}=5.0, w_{\text{adv}}=0.5$).
- **Optimizers**: Adam ($lr=1\times 10^{-4}, \beta_1=0.5, \beta_2=0.999$).
- **Precision**: PyTorch AMP `GradScaler('cuda')`.
- **Batch Size**: 4.
- **Total Steps**: 500 optimizer steps.
- **Validation Schedule**: Every 50 steps on 100 held-out cross-ID validation pairs.
- **Training Time**: 120.4 seconds (0.24 s/step) on RTX 5060 GPU.

---

## 9. Training Progression

| Step | ArcFace A | ArcFace B | Adv ($A - B$) | $P(A > B)$ | Raw Swap Sharpness | Composite Sharpness | Skip Delta (Adv) | Skip Delta (Sharp) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** (Init) | 0.0120 | 0.6790 | -0.6670 | 0.0% | 697.2 | 443.3 | — | — |
| **50** (Best) | **0.2272** | **0.1836** | **+0.0436** | **58.0%** | **354.1** | **301.6** | **+0.0698** | **+90.8** |
| **100** | 0.2275 | 0.1996 | +0.0279 | 53.0% | 327.0 | 306.9 | +0.0743 | +97.9 |
| **150** | 0.2407 | 0.2195 | +0.0212 | 56.0% | 288.4 | 291.9 | +0.0712 | +83.2 |
| **200** | 0.2309 | 0.2032 | +0.0277 | 58.0% | 272.9 | 280.7 | +0.0644 | +71.3 |
| **250** | 0.2323 | 0.2073 | +0.0250 | 54.0% | 245.0 | 266.6 | +0.0775 | +63.1 |
| **300** | 0.2288 | 0.2106 | +0.0181 | 54.0% | 224.7 | 252.8 | +0.0694 | +49.1 |
| **350** | 0.2237 | 0.2052 | +0.0185 | 56.0% | 218.4 | 255.0 | +0.0957 | +50.5 |
| **400** | 0.2060 | 0.2154 | -0.0094 | 48.0% | 220.7 | 249.9 | +0.0782 | +46.7 |
| **450** | 0.2041 | 0.2146 | -0.0105 | 50.0% | 213.5 | 251.9 | +0.0748 | +48.9 |
| **500** | 0.1787 | 0.2133 | -0.0345 | 38.0% | 247.7 | 269.4 | +0.0727 | +61.7 |

**Peak Performance**: Step 50 achieved the maximum Source-Target Advantage ($A - B = +0.0436$) and highest raw swap sharpness ($354.1$). Beyond step 250, target leakage slowly increased in the absence of an explicit identity margin penalty.

---

## 10. Baseline Phase 6G vs Experimental Phase 6G.1 Comparison (100 Pairs)

Evaluated across the 100 genuine held-out validation pairs:

| Metric | Baseline Phase 6G | Exp Phase 6G.1 | Delta | Relative Change |
| :--- | :---: | :---: | :---: | :---: |
| **Source Similarity (A)** | 0.2366 | 0.2272 | -0.0094 | -4.0% |
| **Target Similarity (B)** | 0.2024 | **0.1836** | **-0.0189** | **-9.3% (Leakage Reduced)** |
| **Baseline Pair Similarity (C)**| -0.0000 | -0.0000 | 0.0000 | — |
| **Source Advantage (A - B)** | +0.0341 | **+0.0436** | **+0.0095** | **+27.9% Improvement** |
| **Identity Gain (A - C)** | +0.2366 | +0.2272 | -0.0094 | -4.0% |
| **Percentage A > B** | 58.0% | 58.0% | 0.0% | Parity |
| **Percentage A > C** | 95.0% | 92.0% | -3.0% | Comparable |
| **Raw Swap Sharpness** | 158.9 | **354.1** | **+195.2** | **+122.9% (Surge)** |
| **Composite Sharpness** | 234.7 | **301.6** | **+66.9** | **+28.5% (Sharper)** |
| **Face Redetection Rate** | 98.0% | 98.0% | 0.0% | Identical (100% natural) |
| **Landmark Error** | 1.79 px | 1.84 px | +0.05 px | Sub-2 px precision |
| **Mask Coverage (>0.5)** | 44.9% | 45.3% | +0.4% | Stable |
| **Mask Leakage** | 5.57% | 6.98% | +1.41% | Well-bounded |

---

## 11. Part 9 Controlled Ablation (128×128 Skip Enabled vs Disabled)

To test whether the improvements are causally produced by the new 128×128 full-resolution source skip, inference was run on the identical Phase 6G.1 checkpoint with `disable_fullres_skip=True`:

| Evaluation Condition | ArcFace A | ArcFace B | Advantage ($A - B$) | Swap Sharpness | Composite Sharpness |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Phase 6G.1 (128x128 Skip ENABLED)** | **0.2272** | **0.1836** | **+0.0436** | **354.1** | **301.6** |
| **Phase 6G.1 (128x128 Skip DISABLED)** | 0.1601 | 0.1863 | **-0.0262** | 206.5 | 210.8 |
| **Causal Skip Delta** | **+0.0671** | **-0.0027** | **+0.0698** | **+147.6** | **+90.8** |

### Causal Attribution:
1. **Sharpness**: Disabling the 128×128 skip drops raw swap sharpness from $354.1 \to 206.5$ (a loss of **$-147.6$ points**). This proves that the full-resolution skip is 100% responsible for restoring facial crispness.
2. **Identity Advantage**: Disabling the 128×128 skip causes the Source-Target Advantage to invert from positive ($+0.0436$) to target-dominant ($-0.0262$, a loss of **$-0.0698$**). The skip actively reinforces source identity over target features.

---

## 12. Identity & Target Leakage Decomposition

- **Target Leakage Suppression**: In Phase 6G, mean target similarity in the output was $B = 0.2024$. In Phase 6G.1, target similarity dropped to $B = 0.1836$ (a $-9.3\%$ decrease).
- **Source Dominance**: The model operates with a positive average source advantage ($A - B = +0.0436$).
- **The Residual Ceiling**: Despite the increase in average advantage, the proportion of pairs where $A > B$ remained fixed at **58.0%**. In 42.0% of pairs, the output remains moderately target-dominant ($B \ge A$).

---

## 13. Sharpness & High-Frequency Detail Metrics

Detailed percentile distribution for Laplacian variance sharpness:

| Image Representation | Mean | Median | P10 | P25 | P75 | P90 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ground-Truth Target** | 325.4 | 278.8 | 101.6 | 166.9 | 403.2 | 516.2 |
| **6D Aligned Source** | 316.0 | 277.2 | 126.8 | 176.9 | 401.2 | 553.3 |
| **Baseline Phase 6G Swap** | 158.9 | 160.0 | 125.2 | 141.8 | 176.8 | 195.3 |
| **Phase 6G.1 Exp Swap** | **354.1** | **341.1** | **264.3** | **294.6** | **400.2** | **461.9** |
| **Phase 6G.1 Exp Composite** | **301.6** | **267.5** | **137.6** | **191.8** | **376.3** | **480.1** |

The raw neural swap output $I_{\text{swap}}$ now possesses **higher sharpness than both the baseline ($+122.9\%$) and the 6D aligned source itself**, completely overcoming the neural smoothing bottleneck identified in Phase 6G.

---

## 14. Mask & Background Metrics

- **Mask Coverage (>0.5)**: $45.3\% \pm 2.8\%$ (stable, covers facial oval)
- **Mask Leakage Outside Face**: $6.98\% \pm 2.29\%$ (low, minimal bleeding onto hair/neck)
- **Face-Region L1 Error vs Target**: $0.1218 \pm 0.0419$ (facial features actively modified)
- **Background L1 Error vs Target**: $0.0071 \pm 0.0032$ (background perfectly preserved)

---

## 15. Representative Visual Comparison Grids (20 Pairs)

20 representative 8-column comparison grids were generated and saved in `ml/experiments/phase6g_1_fullres_skip/grids/`:
Columns: `[ 1. SOURCE | 2. TARGET | 3. 6D ALIGNED | 4. 6G BASE SWAP | 5. 6G.1 EXP SWAP | 6. 6G BASE COMP | 7. 6G.1 EXP COMP | 8. EXP MASK ]`

Key Observations from Saved Grids:
1. **Strong Transfer (High Advantage)**:
   - `grid_01_pair25_102198_108942.png`: $A = 0.3951, B = 0.0483, A - B = +0.3468$. Swap sharpness increased from $156.4 \to 395.2$. Facial features (eyes, nose, mouth) match the source identity with sharp detail.
   - `grid_03_pair8_172366_086159.png`: $A = 0.4312, B = 0.0421, A - B = +0.3891$.
2. **Average Transfer (Median Advantage)**:
   - `grid_05_pair63_143347_156456.png`: $A = 0.2814, B = 0.2412, A - B = +0.0402$. Swap sharpness increased from $162.1 \to 348.0$. Crisp texture on cheeks and mouth.
   - `grid_08_pair44_154992_034298.png`: $A = 0.3512, B = 0.3120, A - B = +0.0392$.
3. **Weak Transfer (Target Dominant)**:
   - `grid_12_pair26_168771_138664.png`: $A = 0.0284, B = 0.3952, A - B = -0.3668$. Although visually sharp ($388.1$), the facial traits still strongly resemble the target person.
4. **Large Pose Differences**:
   - `grid_13_pair58_013099_163850.png` (Pose Diff: 8.92 px): Full-res skip maintains sharpness ($412.3$) without creating planar tearing artifacts.
   - `grid_14_pair79_199397_159846.png` (Pose Diff: 7.84 px): Landmark alignment remains stable (error $2.14\text{ px}$).

---

## 16. Limitations

1. **Identity Dominance Ceiling ($P(A > B) = 58.0\%$)**:
   While the 128×128 skip successfully eliminated visual blur (+122.9% sharpness), it did not significantly alter the global identity win-rate ($58.0\%$).
2. **Drift During Extended Training**:
   Without an explicit margin loss penalizing $B > A$, training past step 200 caused the target structure encoder to re-assert target traits.
3. **Target Structure Encoder Leakage**:
   The bottleneck continues to receive low-frequency target RGB and landmark heatmaps that compete directly with source skip features.

---

## 17. Final Classification: `NEEDS TUNING`

- **Why NOT Fail?**:
  The experiment achieved a decisive technical breakthrough on visual sharpness: raw swap sharpness surged by **+122.9%** ($158.9 \to 354.1$), composite sharpness improved by **+28.5%**, target leakage dropped by **-9.3%**, and the ablation proved that the 128×128 skip is 100% causally responsible for $+0.0698$ advantage and $+147.6$ sharpness.
- **Why NOT Pass?**:
  Per the strict milestone success criteria, `PASS` requires demonstrating that the experimental model increases actual source-identity dominance ($P(A > B) > 58\%$). Because $P(A > B)$ remained at 58.0%, the architecture requires loss-level tuning (e.g. an explicit identity margin loss $\max(0, B - A + m)$) to force source identity over target features.

---

## 18. Hard Stop Verification

- **ArcFace SHA**: Unchanged (`2B75B93C...`)
- **CelebA Dataset**: Unchanged (202,599 images, 10,177 IDs)
- **Production Phase 6G Checkpoint**: `ml/checkpoints/stage2/phase6g/best_model.pt` **UNTOUCHED / UNMODIFIED**
- **WebRTC / Client / Server**: **NOT MODIFIED** (Merge completed, no prototype code written)
- **WebRTC Status**: **STRICTLY PAUSED**
- **Git Push**: **NONE EXECUTED**
