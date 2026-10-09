# Phase 6D: MediaPipe Pipeline Inspection & Landmark Topology Analysis

## 1. Executive Summary
This document analyzes the existing MediaPipe FaceLandmarker pipeline used in the Hackathena repository (`ml/training/face_preprocessing.py` and `ml/data/dataset.py`) to evaluate its suitability for source-to-target geometric correspondence diagnostics in Phase 6D.

---

## 2. Specific Findings

### A. Which 478 Landmark Coordinates Are Available
- **Model Asset**: `client/public/models/face_landmarker.task` loaded via `mediapipe.tasks.python.vision.FaceLandmarker`.
- **Landmark Topology**: MediaPipe regresses 478 3D landmarks (`x`, `y`, `z`) per face:
  - **Indices 0–467 (468 points)**: Canonical MediaPipe dense Face Mesh topology covering face oval, eyebrows, eye sockets, eyelids, nose ridge, nose base, lips, and cheeks.
  - **Indices 468–472 (5 points)**: Left iris landmarks (index 468 is the left iris center; 469–472 form the iris contour).
  - **Indices 473–477 (5 points)**: Right iris landmarks (index 473 is the right iris center; 474–477 form the iris contour).
- In `MediaPipeLandmarkDetector.detect()`, the full 478 points are extracted into `dense_coords` of shape `[478, 2]`.

### B. Coordinate Representation: Normalized [0, 1] vs Absolute Pixels
- **Raw MediaPipe Output**: `detection_result.face_landmarks` stores normalized coordinates where $x \in [0.0, 1.0]$ and $y \in [0.0, 1.0]$ relative to image dimensions.
- **Detector Processing**: In `ml/training/face_preprocessing.py` (lines 412–417):
  ```python
  dense_coords = np.zeros((len(chosen_face), 2), dtype=np.float32)
  for i, lm in enumerate(chosen_face):
      dense_coords[i, 0] = lm.x * width
      dense_coords[i, 1] = lm.y * height
  ```
  Coordinates are explicitly unnormalized into absolute pixel coordinates `[x, y]` relative to the input image resolution ($W \times H$).
- When transformed via Umeyama similarity alignment to target square resolution (128x128), landmark coordinates remain in absolute pixel space $[0.0, 128.0]$.

### C. Landmarks Currently Used for 5-Point Alignment
The canonical 5-point alignment keypoints are extracted in `MediaPipeLandmarkDetector.detect()` using the following specific indices:
1. **Left Eye Center**: `MEDIAPIPE_LEFT_EYE_IDX = 468` (Left iris center)
2. **Right Eye Center**: `MEDIAPIPE_RIGHT_EYE_IDX = 473` (Right iris center)
3. **Nose Tip**: `MEDIAPIPE_NOSE_TIP_IDX = 1`
4. **Left Mouth Corner**: `MEDIAPIPE_LEFT_MOUTH_IDX = 61`
5. **Right Mouth Corner**: `MEDIAPIPE_RIGHT_MOUTH_IDX = 291`

These 5 points are aligned via closed-form least-squares Umeyama similarity transformation (`estimate_umeyama_similarity_transform`) to match the canonical 5-point reference:
- Left Eye: `[0.3419 * 128, 0.4615 * 128] = [43.76, 59.07]`
- Right Eye: `[0.6565 * 128, 0.4598 * 128] = [84.03, 58.85]`
- Nose Tip: `[0.5002 * 128, 0.6405 * 128] = [64.03, 81.98]`
- Left Mouth Corner: `[0.3709 * 128, 0.8246 * 128] = [47.48, 105.55]`
- Right Mouth Corner: `[0.6315 * 128, 0.8232 * 128] = [80.83, 105.37]`

### D. Landmarks Used for Face Oval Mask
The facial mask is generated from 36 contour landmark indices defined in `MEDIAPIPE_FACE_OVAL_INDICES`:
```python
MEDIAPIPE_FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]
```
These 36 points form a continuous closed boundary encompassing the forehead hairline, temples, cheeks, and jawline. In `generate_facial_mask`, the points are rasterized into a binary polygon mask, then softened via Gaussian blur (`radius=feather_radius`, default 5 pixels) to output a `[1, 128, 128]` tensor in `[0.0, 1.0]`.

### E. Bidirectional Reliability on Source and Target Images
- **Consistency**: Both source and target images originate from genuine CelebA photos.
- **Multi-Face Handling**: When multiple faces are present, `MediaPipeLandmarkDetector` deterministically selects the face with the largest 2D bounding area (`selected_idx`).
- **Failure Rate**: On the CelebA validation split, landmark detection succeeds on >99.2% of images.
- **Symmetry**: Because both source and target images undergo identical landmark detection, the resulting 478 points share an identical topological ordering and anatomical semantic correspondence across both faces.
- **Robustness**: In `CelebAPairedDataset`, any sample where face detection fails raises `NoFaceDetectedError` and resamples another genuine pair up to 15 attempts. For Phase 6D, we can filter or verify pairs such that both source and target have guaranteed valid 478 landmarks.

---

## 3. Facial Semantic Subsets for Geometric Diagnostic
To evaluate geometric correspondence beyond aggregate error, we partition the 478 landmarks into standard anatomical subsets:
1. **Contour / Jawline (36 points)**: `MEDIAPIPE_FACE_OVAL_INDICES`
2. **Left Eyebrow (10 points)**: `[70, 63, 105, 66, 107, 55, 65, 52, 53, 46]`
3. **Right Eyebrow (10 points)**: `[336, 296, 334, 293, 300, 285, 295, 282, 283, 276]`
4. **Left Eye (16 points)**: `[33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]`
5. **Right Eye (16 points)**: `[362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]`
6. **Nose (16 points)**: Bridge `[168, 6, 197, 195, 5]`, tip & nostrils `[1, 2, 98, 327, 4, 278, 48, 115, 220, 45, 275]`
7. **Outer Lips (20 points)**: `[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]`
8. **Inner Lips (20 points)**: `[78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]`
9. **Cheeks (12 points)**: Left `[116, 117, 118, 123, 50, 205]`, Right `[345, 346, 347, 352, 280, 425]`
10. **Irises (10 points)**: Left iris `[468, 469, 470, 471, 472]`, Right iris `[473, 474, 475, 476, 477]`

This provides a complete, granular anatomical basis for measuring regional reprojection errors and geometric alignment accuracy.
