"""
Real Face Preprocessing & Landmark Alignment Pipeline for Hackathena Face-Swap
Transforms real face images into canonically aligned, normalized tensors and real landmark-derived soft masks.

Pipeline:
  Real Image -> Face & Landmark Detection -> Real 5-Point / Dense Landmarks ->
  Umeyama Similarity Alignment -> Crop & Resize -> Normalization -> Real Landmark Mask
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Union, List, Dict, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.spatial import ConvexHull

# Canonical 5-point facial template coordinates normalized to [0.0, 1.0]
# Order: Left Eye, Right Eye, Nose Tip, Left Mouth Corner, Right Mouth Corner
CANONICAL_5PTS_NORMALIZED = np.array([
    [0.3419, 0.4615],  # Left Eye
    [0.6565, 0.4598],  # Right Eye
    [0.5002, 0.6405],  # Nose Tip
    [0.3709, 0.8246],  # Left Mouth Corner
    [0.6315, 0.8232],  # Right Mouth Corner
], dtype=np.float32)

# MediaPipe FaceMesh topological indices for 5 canonical alignment points
# Left eye center, right eye center, nose tip, left mouth corner, right mouth corner
MEDIAPIPE_LEFT_EYE_IDX = 468   # Left iris center (or mean of eye corners 33, 133)
MEDIAPIPE_RIGHT_EYE_IDX = 473  # Right iris center (or mean of eye corners 362, 263)
MEDIAPIPE_NOSE_TIP_IDX = 1     # Nose tip
MEDIAPIPE_LEFT_MOUTH_IDX = 61  # Left mouth corner
MEDIAPIPE_RIGHT_MOUTH_IDX = 291  # Right mouth corner

# MediaPipe 36-point Face Oval contour indices (tracing forehead hairline, temples, cheeks, jawline, chin)
MEDIAPIPE_FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]


class FaceDetectionError(Exception):
    """Base exception for face detection and preprocessing failures."""
    pass


class NoFaceDetectedError(FaceDetectionError):
    """Raised when no face is found in an image."""
    pass


class InvalidLandmarksError(FaceDetectionError):
    """Raised when detected landmarks are malformed, degenerate, or contain NaN/Inf."""
    pass


def get_canonical_landmarks(image_size: int = 128) -> np.ndarray:
    """Returns canonical 5-point facial landmarks scaled to the target image size."""
    return CANONICAL_5PTS_NORMALIZED * float(image_size)


def estimate_umeyama_similarity_transform(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """
    Computes least-squares 2D similarity transform (rotation, uniform scale, translation)
    mapping source points to destination points using the closed-form Umeyama algorithm.

    Args:
        src_pts: [N, 2] source coordinates
        dst_pts: [N, 2] target coordinates

    Returns:
        M: [2, 3] affine transformation matrix
    """
    src_pts = np.asarray(src_pts, dtype=np.float64)
    dst_pts = np.asarray(dst_pts, dtype=np.float64)

    num_pts = src_pts.shape[0]
    if num_pts < 3:
        raise ValueError(f"At least 3 landmark pairs required for similarity transform, got {num_pts}")

    mean_src = np.mean(src_pts, axis=0)
    mean_dst = np.mean(dst_pts, axis=0)

    src_centered = src_pts - mean_src
    dst_centered = dst_pts - mean_dst

    var_src = np.var(src_pts[:, 0]) + np.var(src_pts[:, 1])
    if var_src < 1e-8:
        raise ValueError("Degenerate source landmarks with zero variance")

    # Covariance matrix
    cov = np.dot(dst_centered.T, src_centered) / num_pts
    u, d, vt = np.linalg.svd(cov)

    s = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[1, 1] = -1

    r = np.dot(u, np.dot(s, vt))
    scale = 1.0 if var_src == 0 else (1.0 / var_src) * np.trace(np.dot(np.diag(d), s))
    t = mean_dst - scale * np.dot(r, mean_src)

    m = np.zeros((2, 3), dtype=np.float32)
    m[0:2, 0:2] = scale * r
    m[0:2, 2] = t
    return m


def align_face_similarity(
    image: Image.Image,
    detected_landmarks: np.ndarray,
    target_size: int = 128,
    extra_landmarks: Optional[np.ndarray] = None
) -> Tuple[Image.Image, np.ndarray, Optional[np.ndarray]]:
    """
    Aligns and crops a facial image using 5-point landmark similarity transformation.

    Args:
        image: PIL RGB Image
        detected_landmarks: [5, 2] key landmarks (left eye, right eye, nose, left mouth, right mouth)
        target_size: Target square resolution (128 or 256)
        extra_landmarks: Optional [N, 2] dense/contour landmarks to also project into aligned space

    Returns:
        aligned_image: Aligned PIL Image of size (target_size, target_size)
        aligned_5pts: [5, 2] key landmarks in aligned coordinate space
        aligned_extra: [N, 2] extra landmarks in aligned coordinate space (or None)
    """
    if detected_landmarks.shape != (5, 2):
        raise ValueError(f"Expected [5, 2] key landmarks for alignment, got {detected_landmarks.shape}")

    canonical_pts = get_canonical_landmarks(target_size)
    matrix = estimate_umeyama_similarity_transform(detected_landmarks, canonical_pts)

    # Invert affine matrix for PIL image.transform
    # PIL affine transform expects matrix mapping destination pixel (x, y) to source pixel (u, v)
    m_3x3 = np.vstack([matrix, [0, 0, 1]])
    inv_m = np.linalg.inv(m_3x3)[0:2, :]
    pil_matrix = (inv_m[0, 0], inv_m[0, 1], inv_m[0, 2],
                  inv_m[1, 0], inv_m[1, 1], inv_m[1, 2])

    aligned_img = image.transform(
        (target_size, target_size),
        Image.AFFINE,
        pil_matrix,
        resample=Image.BILINEAR
    )

    # Project 5 key landmarks into aligned coordinate frame
    ones_5 = np.ones((5, 1), dtype=np.float32)
    homo_5 = np.hstack([detected_landmarks, ones_5])
    aligned_5pts = np.dot(homo_5, matrix.T)

    aligned_extra = None
    if extra_landmarks is not None:
        ones_extra = np.ones((extra_landmarks.shape[0], 1), dtype=np.float32)
        homo_extra = np.hstack([extra_landmarks, ones_extra])
        aligned_extra = np.dot(homo_extra, matrix.T)

    return aligned_img, aligned_5pts, aligned_extra


def generate_facial_mask(
    contour_landmarks: np.ndarray,
    image_size: int = 128,
    feather_radius: int = 5,
    use_convex_hull: bool = False
) -> np.ndarray:
    """
    Generates a genuine landmark-derived soft facial region mask.
    The mask boundary is computed directly from detected facial contour landmark coordinates
    (e.g., jawline, cheeks, forehead hairline) without hardcoded anatomical offsets.

    Args:
        contour_landmarks: [N, 2] detected facial contour landmarks (N >= 3)
        image_size: Image width and height (square resolution)
        feather_radius: Gaussian blur feathering radius in pixels (applied AFTER real mask creation)
        use_convex_hull: If True, computes the 2D convex hull of contour_landmarks

    Returns:
        mask: [1, H, W] float32 numpy array with values in [0.0, 1.0]
    """
    pts = np.asarray(contour_landmarks, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
        raise ValueError(f"Expected [N, 2] contour points with N >= 3, got shape {pts.shape}")

    if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
        raise InvalidLandmarksError("Contour landmarks contain NaN or Inf values")

    if use_convex_hull or pts.shape[0] > 36:
        # Use 2D convex hull of the landmark coordinates
        hull = ConvexHull(pts)
        hull_pts = pts[hull.vertices]
        polygon_points = [(float(p[0]), float(p[1])) for p in hull_pts]
    else:
        # Ordered contour points (e.g. MediaPipe face oval contour loop)
        polygon_points = [(float(p[0]), float(p[1])) for p in pts]

    # Rasterize binary polygon mask derived directly from genuine detected landmark coordinates
    mask_img = Image.new('L', (image_size, image_size), 0)
    draw = ImageDraw.Draw(mask_img)
    draw.polygon(polygon_points, fill=255)

    # Apply soft boundary Gaussian feathering strictly AFTER creating the real landmark mask
    if feather_radius > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=feather_radius))

    mask_array = np.array(mask_img, dtype=np.float32) / 255.0
    mask_array = np.clip(mask_array, 0.0, 1.0)

    # Shape: [1, H, W]
    return np.expand_dims(mask_array, axis=0)


def generate_landmark_map(
    landmarks: np.ndarray,
    image_size: int = 128,
    radius: float = 1.0,
    sigma: float = 1.0
) -> np.ndarray:
    """
    Generates a single-channel landmark conditioning map [1, H, W] from genuine detected coordinates.
    Values are strictly bounded in [0.0, 1.0].

    Args:
        landmarks: [N, 2] facial landmark coordinates in target coordinate space
        image_size: Target square resolution (e.g. 128)
        radius: Radius of rendered landmark points
        sigma: Optional Gaussian smoothing parameter

    Returns:
        landmark_map: [1, H, W] float32 array in [0.0, 1.0]
    """
    map_img = Image.new('L', (image_size, image_size), 0)
    draw = ImageDraw.Draw(map_img)

    pts = np.asarray(landmarks, dtype=np.float32)
    for p in pts:
        x, y = float(p[0]), float(p[1])
        if 0 <= x < image_size and 0 <= y < image_size:
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=255)

    if sigma > 0:
        map_img = map_img.filter(ImageFilter.GaussianBlur(radius=sigma))

    arr = np.array(map_img, dtype=np.float32) / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    return np.expand_dims(arr, axis=0)


def normalize_image_tensor(image: Union[Image.Image, np.ndarray]) -> np.ndarray:
    """
    Converts RGB image to normalized float32 tensor array [3, H, W] in range [-1.0, 1.0].

    Args:
        image: PIL Image or [H, W, 3] uint8 numpy array

    Returns:
        tensor: [3, H, W] float32 array in [-1.0, 1.0]
    """
    if isinstance(image, Image.Image):
        arr = np.array(image.convert('RGB'), dtype=np.float32)
    else:
        arr = np.asarray(image, dtype=np.float32)

    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected 3-channel RGB image, got shape {arr.shape}")

    # Scale from [0, 255] to [-1.0, 1.0]
    normalized = (arr / 127.5) - 1.0

    # Transpose [H, W, C] to [C, H, W]
    tensor = np.transpose(normalized, (2, 0, 1))
    return np.ascontiguousarray(tensor, dtype=np.float32)


def denormalize_image_tensor(tensor: np.ndarray) -> Image.Image:
    """
    Converts normalized float32 tensor [3, H, W] in [-1.0, 1.0] back to PIL RGB Image [0, 255].
    """
    if tensor.ndim == 4:
        tensor = tensor[0]
    arr = np.transpose(tensor, (1, 2, 0))
    arr = (arr + 1.0) * 127.5
    arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode='RGB')


class LandmarkDetectionResult:
    """Container for genuine facial landmark detection results."""
    def __init__(
        self,
        key_landmarks_5pts: np.ndarray,
        contour_landmarks: np.ndarray,
        dense_landmarks: Optional[np.ndarray] = None,
        confidence: float = 1.0,
        num_faces_detected: int = 1,
        selected_face_index: int = 0
    ):
        self.key_landmarks_5pts = key_landmarks_5pts  # [5, 2] in pixel coordinates
        self.contour_landmarks = contour_landmarks    # [K, 2] facial contour landmarks
        self.dense_landmarks = dense_landmarks        # [478, 2] or None
        self.confidence = confidence
        self.num_faces_detected = num_faces_detected
        self.selected_face_index = selected_face_index


class MediaPipeLandmarkDetector:
    """
    Genuine facial landmark detector using Google MediaPipe FaceLandmarker.
    Regresses 478 dense 3D facial landmarks from real image pixels.
    Extracts 5 canonical keypoints for Umeyama similarity alignment and 36 contour landmarks for face mask.
    """
    def __init__(
        self,
        model_asset_path: str = "client/public/models/face_landmarker.task",
        min_detection_confidence: float = 0.5
    ):
        self.model_asset_path = model_asset_path
        self.min_detection_confidence = min_detection_confidence
        self._landmarker = None

    def _get_landmarker(self):
        if self._landmarker is not None:
            return self._landmarker

        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError:
            raise ImportError(
                "Genuine facial landmark detection requires 'mediapipe'. "
                "Please install: pip install mediapipe>=0.10.0\n"
                "Automatic package installation is disabled to prevent unverified changes. "
                "No synthetic, estimated, or placeholder landmarks will be generated."
            )

        # Validate that the model asset exists
        model_path = Path(self.model_asset_path)
        if not model_path.is_file():
            # Check relative to repo root if run from subfolder
            repo_root_candidate = Path(__file__).resolve().parent.parent.parent / self.model_asset_path
            if repo_root_candidate.is_file():
                model_path = repo_root_candidate
            else:
                raise FileNotFoundError(
                    f"MediaPipe Face Landmarker model asset not found at '{self.model_asset_path}' "
                    f"or '{repo_root_candidate}'.\n"
                    "The model asset 'face_landmarker.task' is required for real landmark detection. "
                    "Automatic model downloading is disabled. "
                    "Ensure the model file is placed in 'client/public/models/face_landmarker.task'."
                )

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=5,
            min_face_detection_confidence=self.min_detection_confidence,
            min_face_presence_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_detection_confidence
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        return self._landmarker

    def detect(self, image: Image.Image) -> LandmarkDetectionResult:
        """
        Detects genuine facial landmarks on a real image.

        Args:
            image: PIL RGB Image

        Returns:
            LandmarkDetectionResult with genuine detected coordinates.

        Raises:
            NoFaceDetectedError: if 0 faces are detected (NEVER fabricates landmarks)
            InvalidLandmarksError: if landmarks are degenerate or malformed
        """
        import mediapipe as mp
        landmarker = self._get_landmarker()

        rgb_image = image.convert('RGB')
        width, height = rgb_image.size
        np_image = np.array(rgb_image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_image)

        detection_result = landmarker.detect(mp_image)

        num_faces = len(detection_result.face_landmarks)
        if num_faces == 0:
            raise NoFaceDetectedError("No face detected in image by MediaPipe FaceLandmarker")

        # Deterministic multi-face selection rule:
        # If multiple faces are detected, select the face with the largest 2D bounding area.
        selected_idx = 0
        if num_faces > 1:
            max_area = -1.0
            for i, face in enumerate(detection_result.face_landmarks):
                xs = [lm.x for lm in face]
                ys = [lm.y for lm in face]
                area = (max(xs) - min(xs)) * (max(ys) - min(ys))
                if area > max_area:
                    max_area = area
                    selected_idx = i

        chosen_face = detection_result.face_landmarks[selected_idx]

        # Extract all dense landmarks in pixel coordinates [478, 2]
        dense_coords = np.zeros((len(chosen_face), 2), dtype=np.float32)
        for i, lm in enumerate(chosen_face):
            dense_coords[i, 0] = lm.x * width
            dense_coords[i, 1] = lm.y * height

        # Validate coordinates
        if np.any(np.isnan(dense_coords)) or np.any(np.isinf(dense_coords)):
            raise InvalidLandmarksError("Detected facial landmarks contain NaN or Inf values")

        # Extract 5 canonical alignment keypoints
        left_eye = dense_coords[MEDIAPIPE_LEFT_EYE_IDX]
        right_eye = dense_coords[MEDIAPIPE_RIGHT_EYE_IDX]
        nose_tip = dense_coords[MEDIAPIPE_NOSE_TIP_IDX]
        left_mouth = dense_coords[MEDIAPIPE_LEFT_MOUTH_IDX]
        right_mouth = dense_coords[MEDIAPIPE_RIGHT_MOUTH_IDX]
        key_5pts = np.vstack([left_eye, right_eye, nose_tip, left_mouth, right_mouth])

        # Extract facial contour landmarks (36 points of the face oval boundary)
        contour_coords = dense_coords[MEDIAPIPE_FACE_OVAL_INDICES]

        return LandmarkDetectionResult(
            key_landmarks_5pts=key_5pts,
            contour_landmarks=contour_coords,
            dense_landmarks=dense_coords,
            confidence=1.0,
            num_faces_detected=num_faces,
            selected_face_index=selected_idx
        )


class RealFacePreprocessor:
    """
    Face preprocessing coordinator for loading, detecting genuine landmarks,
    aligning via Umeyama similarity transform, masking, and normalizing faces.
    """
    def __init__(
        self,
        image_size: int = 128,
        feather_radius: int = 5,
        model_asset_path: str = "client/public/models/face_landmarker.task",
        min_detection_confidence: float = 0.5
    ):
        if image_size not in (128, 256):
            raise ValueError(f"image_size must be 128 or 256, got {image_size}")
        self.image_size = image_size
        self.feather_radius = feather_radius
        self.canonical_5pts = get_canonical_landmarks(image_size)
        self.detector = MediaPipeLandmarkDetector(
            model_asset_path=model_asset_path,
            min_detection_confidence=min_detection_confidence
        )

    def detect_landmarks(self, image: Image.Image) -> LandmarkDetectionResult:
        """
        Detects genuine facial landmarks on the provided real image.
        Raises an explicit error if detection fails. Never fabricates coordinates.
        """
        return self.detector.detect(image)

    def preprocess_image(
        self,
        image_path: str
    ) -> Tuple[np.ndarray, np.ndarray, LandmarkDetectionResult]:
        """
        Full genuine preprocessing pipeline for any real image:
        1. Load real image
        2. Detect genuine facial landmarks (raises NoFaceDetectedError if absent)
        3. Umeyama similarity alignment using detected 5 keypoints to canonical frame
        4. Derive soft face mask directly from aligned real contour landmarks
        5. Normalize aligned image to float32 [-1.0, 1.0]

        Returns:
            image_tensor: [3, H, W] float32 in [-1.0, 1.0]
            mask_tensor:  [1, H, W] float32 in [0.0, 1.0]
            detection_result: LandmarkDetectionResult with genuine coordinates & metadata
        """
        img = Image.open(image_path).convert('RGB')
        det_result = self.detect_landmarks(img)

        aligned_img, aligned_5pts, aligned_contour = align_face_similarity(
            image=img,
            detected_landmarks=det_result.key_landmarks_5pts,
            target_size=self.image_size,
            extra_landmarks=det_result.contour_landmarks
        )

        # Construct soft face mask from actual aligned contour landmark coordinates
        mask = generate_facial_mask(
            contour_landmarks=aligned_contour,
            image_size=self.image_size,
            feather_radius=self.feather_radius
        )

        tensor = normalize_image_tensor(aligned_img)
        return tensor, mask, det_result

    def preprocess_aligned_celeba_image(
        self,
        image_path: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocesses a CelebA image using genuine landmark detection,
        5-point Umeyama alignment, real landmark-derived contour mask, and normalization.
        """
        tensor, mask, _ = self.preprocess_image(image_path)
        return tensor, mask

    def preprocess_sample(
        self,
        image_path: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, LandmarkDetectionResult]:
        """
        Complete preprocessing extracting tensor, soft mask, structural landmark map, and detection metadata.
        """
        img = Image.open(image_path).convert('RGB')
        det_result = self.detect_landmarks(img)

        extra_to_align = det_result.dense_landmarks if det_result.dense_landmarks is not None else det_result.contour_landmarks
        aligned_img, aligned_5pts, aligned_dense = align_face_similarity(
            image=img,
            detected_landmarks=det_result.key_landmarks_5pts,
            target_size=self.image_size,
            extra_landmarks=extra_to_align
        )

        if det_result.dense_landmarks is not None:
            aligned_contour = aligned_dense[MEDIAPIPE_FACE_OVAL_INDICES]
        else:
            aligned_contour = aligned_dense

        mask = generate_facial_mask(
            contour_landmarks=aligned_contour,
            image_size=self.image_size,
            feather_radius=self.feather_radius
        )
        landmark_map = generate_landmark_map(
            landmarks=aligned_dense,
            image_size=self.image_size
        )
        tensor = normalize_image_tensor(aligned_img)
        return tensor, mask, landmark_map, det_result

    def preprocess_unaligned_image(
        self,
        image_path: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocesses an unaligned raw image using genuine landmark detection,
        5-point Umeyama alignment, real landmark-derived contour mask, and normalization.
        """
        tensor, mask, _ = self.preprocess_image(image_path)
        return tensor, mask
