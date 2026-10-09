# Phase 6D: Source->Target Geometric Correspondence Diagnostic Report

## 1. Objective
Following the classification of Phase 6C as FAIL (where generic CNN spatial features failed to provide identity discrimination and naive spatial injection induced severe structural conflicts due to pose/expression disparity), Phase 6D evaluates the hypothesis:

> **Hypothesis**: Source facial visual features can be successfully mapped into target pose/geometry without copying source pose if they are geometrically guided via canonical facial correspondence.

Specifically, Phase 6D investigates whether existing **MediaPipe 478-landmark representations** can establish a stable, meaningful source-to-target facial correspondence for genuine CelebA cross-identity pairs **without training any neural network**.

---

## 2. MediaPipe Implementation Findings
A thorough audit of `ml/training/face_preprocessing.py` and `client/public/models/face_landmarker.task` established:
- **478 Landmark Coordinates**: Consists of 468 standard dense FaceMesh landmarks (indices 0–467) plus 10 iris contour/center landmarks (indices 468–477).
- **Coordinate Space**: Normalized coordinates $[0, 1]$ are mapped to absolute pixel coordinates $[0, 128]$ upon detection.
- **5-Point Alignment**: Based on left iris center (468), right iris center (473), nose tip (1), left mouth corner (61), and right mouth corner (291), aligned to `CANONICAL_5PTS_NORMALIZED` via Umeyama similarity transform.
- **Contour Mask**: Derived from 36 closed-loop landmarks in `MEDIAPIPE_FACE_OVAL_INDICES`.
- **Extraction Reliability**: Successful detection on >99% of CelebA validation images, providing identical topological order across all faces. Full details documented in [mediapipe_analysis.md](file:///c:/files/forgitclone/Hackathena/ml/checkpoints/stage2/phase6d_diagnostic/mediapipe_analysis.md).

---

## 3. Dataset & Pairing Methodology
- **Validation Split**: Held-out, identity-disjoint validation partition of CelebA (`identity_CelebA.txt`, seed=42, `same_identity_probability=0.0`).
- **Pairs Evaluated**: **100 genuine cross-identity pairs** ($N=100$).
- **Filtering**: Verified that both Source and Target images produced complete 478-landmark topologies before processing.
- **Baseline Cross-Identity Separation**:
  - ArcFace cosine similarity between original Source and Target: $C = 0.0092 \pm 0.0972$ (demonstrating true identity orthogonality).

---

## 4. Landmark Correspondence Method
Two deterministic geometric correspondence paradigms were implemented and evaluated on the exact same pairs:
1. **Global Similarity Alignment**:
   - Least-squares Umeyama 5-point similarity transform ($S_{5\text{pts}} \to T_{5\text{pts}}$) computing optimal rotation, uniform scale, and 2D translation.
2. **Local Piecewise-Affine Delaunay Warp**:
   - 2D Delaunay triangulation constructed over Target 478 landmarks augmented with 8 peripheral frame-boundary anchors (ensuring seamless, hole-free full-frame coverage).
   - Dense backward coordinate mapping computed via vectorized barycentric coordinates:
     $$\mathbf{b} = \mathbf{T} \cdot (\mathbf{x} - \mathbf{r})$$
     $$\mathbf{x}_{\text{src}} = b_0 \mathbf{s}_0 + b_1 \mathbf{s}_1 + b_2 \mathbf{s}_2$$
   - Source pixels resampled into target geometry via bilinear interpolation (`cv2.remap`).

---

## 5. Global Similarity-Transform Results
- **ArcFace Source Identity ($G$)**: $0.9838 \pm 0.0129$ ($G - C = +0.9746$)
- **ArcFace Target Similarity ($G_{\text{tgt}}$)**: $0.0090 \pm 0.0945$
- **Target Landmark Reprojection Error**:
  - Mean Error: **4.71 px** ($11.75\%$ of Inter-Ocular Distance / IOD)
  - Median Error: **3.91 px**
  - 95th Percentile: **11.51 px**
  - Jaw / Contour Error: **9.78 px**
  - Nose Error: **5.72 px**
- **Conclusion**: A global similarity transform preserves near-perfect source identity ($0.9838$) because it only rigidly rotates/scales the image, but it **completely fails to adapt to target facial geometry**, retaining massive reprojection error ($>11\%$ IOD).

---

## 6. Local Piecewise-Warp Results
- **ArcFace Source Identity ($L$)**: **$0.6393 \pm 0.1324$** ($L - C = \mathbf{+0.6301}$)
- **ArcFace Target Similarity ($L_{\text{tgt}}$)**: **$0.0314 \pm 0.0886$**
- **Independent Re-Detection Success**: **98.0%** (98 out of 100 warped images successfully detected by MediaPipe FaceLandmarker).
- **Target Landmark Reprojection Error (on Re-Detected Warped Faces)**:
  - Mean Error: **1.41 px** ($3.48\%$ IOD) — a **70.1% error reduction** compared to global alignment ($4.71 \to 1.41$ px).
  - Median Error: **1.23 px**
  - 95th Percentile: **3.12 px**
- **Conclusion**: The local piecewise affine warp successfully shifts the facial structure into target coordinates (reducing landmark error to $\approx 1.4$ px) while retaining substantial source identity ($L = 0.6393$ vs cross-ID baseline $0.0092$).

---

## 7. Landmark Reprojection Errors by Anatomical Region
Measured across all 100 validation pairs (values in pixels on $128 \times 128$ resolution):

| Anatomical Region | Points | Global Similarity Error | Local Piecewise Error | Error Reduction |
| :--- | :---: | :---: | :---: | :---: |
| **Left Eye** | 16 | 3.22 px | **1.02 px** | **-68.3%** |
| **Right Eye** | 16 | 3.07 px | **0.99 px** | **-67.8%** |
| **Left Eyebrow** | 10 | 4.71 px | **1.62 px** | **-65.6%** |
| **Right Eyebrow** | 10 | 5.10 px | **1.39 px** | **-72.7%** |
| **Nose** | 18 | 5.72 px | **1.41 px** | **-75.3%** |
| **Inner Lips** | 20 | 3.63 px | **1.37 px** | **-62.3%** |
| **Outer Lips** | 20 | 3.84 px | **1.41 px** | **-63.3%** |
| **Cheeks** | 12 | 4.89 px | **1.37 px** | **-72.0%** |
| **Central Face** | 14 | 5.61 px | **1.44 px** | **-74.3%** |
| **Contour / Jaw** | 36 | 9.78 px | **2.44 px** | **-75.1%** |
| **Overall Face** | **478** | **4.71 px (11.75% IOD)** | **1.41 px (3.48% IOD)** | **-70.1%** |

Both eyes, the nose, the cheeks, and the lips align to within $\approx 1.0 - 1.4$ pixels of target geometry.

---

## 8. Source Identity Preservation
- ArcFace identity embedding cosine similarities across the 100 pairs:
  - $C = \cos(\text{Source}, \text{Target}) = 0.0092 \pm 0.0972$
  - $G = \cos(\text{Source}, \text{Global Warp}) = 0.9838 \pm 0.0129$ ($G - C = +0.9746$)
  - $L = \cos(\text{Source}, \text{Local Warp}) = \mathbf{0.6393 \pm 0.1324}$ ($L - C = \mathbf{+0.6301}$)
- Critically, the similarity of the locally warped image to the target is near zero:
  - $L_{\text{tgt}} = \cos(\text{Target}, \text{Local Warp}) = 0.0314 \pm 0.0886$
- This confirms that geometric warping does **not** transfer target identity; rather, it reshapes source facial texture into target geometry while keeping identity strongly tied to the source ($L \gg C$).

---

## 9. Target Geometry Matching
Independent MediaPipe extraction on the warped images confirms:
- In 98% of pairs, standard facial feature detectors successfully recognize and track the warped image as a natural face.
- Eye centers, pupils, lip contours, and nose apex match the target position within sub-2-pixel accuracy.
- This demonstrates that landmark-guided correspondence provides a viable geometric bridge from source texture to target spatial configuration.

---

## 10. Similar-Pose Results ($N = 33$)
Pairs with low pose disparity ($\text{Disparity} \le 0.1800$):
- **Cross-ID Baseline ($C$)**: $0.0091$
- **Local Warp Identity ($L$)**: **$0.7025$** ($L - C = \mathbf{+0.6934}$)
- **Target Similarity ($L_{\text{tgt}}$)**: $0.0395$
- **Local Landmark Error**: **$1.04$ px** ($2.48\%$ IOD)
- **Eye Errors**: Left eye $0.75$ px, Right eye $0.80$ px
- **Nose Error**: $0.98$ px
- **Re-Detection Rate**: **100.0%**
- **Assessment**: Near-flawless geometric transfer with very high identity retention.

---

## 11. Moderate-Pose Results ($N = 34$)
Pairs with medium pose disparity ($0.1800 < \text{Disparity} \le 0.3292$):
- **Cross-ID Baseline ($C$)**: $0.0069$
- **Local Warp Identity ($L$)**: **$0.6649$** ($L - C = \mathbf{+0.6580}$)
- **Target Similarity ($L_{\text{tgt}}$)**: $0.0510$
- **Local Landmark Error**: **$1.19$ px** ($2.89\%$ IOD)
- **Eye Errors**: Left eye $0.76$ px, Right eye $0.94$ px
- **Nose Error**: $1.28$ px
- **Re-Detection Rate**: **100.0%**
- **Assessment**: Robust correspondence and stable identity preservation across moderate yaw/pitch differences.

---

## 12. Large-Pose Results ($N = 33$)
Pairs with high pose disparity ($\text{Disparity} > 0.3292$, e.g., semi-profile to frontal):
- **Cross-ID Baseline ($C$)**: $0.0116$
- **Local Warp Identity ($L$)**: **$0.5498$** ($L - C = \mathbf{+0.5383}$)
- **Target Similarity ($L_{\text{tgt}}$)**: $0.0030$
- **Local Landmark Error**: **$2.05$ px** ($5.19\%$ IOD)
- **Global Similarity Landmark Error**: **$6.93$ px** ($17.83\%$ IOD), with jaw error **$15.71$ px** and nose error **$9.08$ px**
- **Re-Detection Rate**: **93.9%** (31/33 detected)
- **Assessment**: Shows notable geometric strain. When a face looking hard left is mapped to a frontal target, occluded cheek regions undergo severe planar stretching.

---

## 13. Qualitative Samples
Comparative evaluation grids generated across validation pairs:

Columns: **[1] SOURCE | [2] TARGET | [3] GLOBAL SIMILARITY WARP | [4] LOCAL PIECEWISE WARP**

![Master Diagnostic Grid](C:\Users\shido\.gemini\antigravity-ide\brain\633ef793-a7b5-42c1-b4cc-c2c615279238\master_diagnostic_grid.png)

### Key Visual Observations:
1. **Rows 1 & 2 (Similar Pose)**:
   - Source facial skin tone, beard/eyes, and expressions are cleanly mapped to target mouth width, jaw shape, and eye positions.
   - Zero tearing, no holes, smooth continuity.
2. **Row 3 & 4 (Moderate Pose)**:
   - Eye gaze and mouth orientation follow target pose smoothly while maintaining source facial likeness.
3. **Row 5 (Extreme Profile vs Frontal)**:
   - Severe lateral stretching across the occluded cheek because 2D planar Delaunay triangulation cannot invent self-occluded facial surface texture.

---

## 14. Failure Cases & Geometric Limitations
1. **Self-Occlusion Under Yaw Rotation**:
   - In 2D piecewise affine warping, if the source is viewed at an angle ($\text{yaw} > 25^\circ$), half of the face is geometrically compressed or self-occluded. Mapping those triangles to a frontal target stretches a few compressed pixels across large target areas.
2. **Extreme Mouth Opening Disparity**:
   - When mapping an open-mouth source (teeth exposed) to a closed-mouth target, triangular interpolation compresses the teeth into a thin horizontal strip, creating visible banding.
3. **Out-of-Plane 3D Pose**:
   - A purely 2D affine mesh lacks 3D depth perception; extreme pitch differences cause foreshortening artifacts.

---

## 15. ArcFace Integrity
- Checkpoint: `ml/models/weights/ms1mv2_iresnet50.pth`
- SHA-256 Verified: `2B75B93C48B01C78A4263F7295AB2DBF84F85190C51BE45B92C4C0B0AEDCEAA3` (Unmodified).

---

## 16. Dataset Integrity
- CelebA Aligned Images: 202,599 files verified in `ml/data/celeba/img_align_celeba/`.
- Identity Annotations: 202,599 lines verified in `ml/data/celeba/identity_CelebA.txt`.

---

## 17. Checkpoint Integrity
All existing training checkpoints remain untouched:
- `ml/checkpoints/stage1/best_model.pt` (Verified intact)
- `ml/checkpoints/stage2/best_model.pt` (Phase 6B.2 Step 750 baseline verified intact, 592,389,456 bytes)
- `ml/checkpoints/stage2/identity_tuning/best_model.pt` (Verified intact)
- `ml/checkpoints/stage2/blur7_pilot/best_model.pt` (Verified intact)
- `ml/checkpoints/stage2/multiscale_identity_pilot/best_model.pt` (Verified intact)
- `ml/checkpoints/stage2/phase6c_diagnostic/` (Verified intact)

---

## 18. Frontend & Backend Integrity
- `client/` and `server/` directories are completely unmodified.

---

## 19. Numerical Stability
- Deterministic, vectorized barycentric coordinate calculation avoided all bounding box clipping and shape broadcasting errors.
- Zero NaN or Inf values across all 100 pairs.
- Full metrics logged to `ml/checkpoints/stage2/phase6d_diagnostic/metrics.json`.

---

## 20. Final Classification

### **CLASSIFICATION: NEEDS TUNING**

### Rationale:
1. **Strong Promise Satisfied**:
   - The landmark-based correspondence establishes that source facial appearance can be geometrically moved into target coordinates while strongly preserving source identity ($L = 0.6393$ vs cross-ID $C = 0.0092$, gain $+0.6301$).
   - The target geometry is adopted with high precision ($1.41$ px error, $70\%$ reduction vs global similarity).
   - In similar and moderate pose conditions (67% of cases), it works remarkably well ($L \approx 0.66 - 0.70$, error $\approx 1.0 - 1.2$ px, 100% re-detection).
2. **Why Not PASS**:
   - In large cross-pose differences ($\text{yaw} > 25^\circ$), a naive 2D planar Delaunay warp suffers from lateral texture stretching over occluded cheeks, causing identity drop ($L = 0.5498$) and re-detection drop ($93.9\%$).
3. **Why Not FAIL**:
   - Landmark correspondence unequivocally solves the spatial conflict that caused Phase 6C to fail. It proves that geometric correspondence is the missing link needed to inject source visual features into target generator coordinates.
   - The limitations under large pose differences can be addressed via deterministic pose-aware weighting, 3D landmark depth correction, or blending with global canonical features.

**Phase 6D is complete. Stopping per instructions.**
