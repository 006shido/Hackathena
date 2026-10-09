"""Shared face alignment and correspondence for inference and research training."""
from typing import Tuple
import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial import Delaunay
from ml.training.face_preprocessing import (
    RealFacePreprocessor, align_face_similarity, generate_facial_mask,
    generate_landmark_map, normalize_image_tensor, MEDIAPIPE_FACE_OVAL_INDICES,
)

def compute_piecewise_affine_map(src_pts: np.ndarray, dst_pts: np.ndarray, width: int = 128, height: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes dense backward coordinate mapping (map_x, map_y) from destination to source
    using Delaunay triangulation and barycentric coordinates.
    Tiled with boundary points to ensure complete coverage without holes.
    """
    boundary_pts = np.array([
        [0.0, 0.0], [width - 1.0, 0.0], [0.0, height - 1.0], [width - 1.0, height - 1.0],
        [(width - 1.0) / 2.0, 0.0], [0.0, (height - 1.0) / 2.0],
        [width - 1.0, (height - 1.0) / 2.0], [(width - 1.0) / 2.0, height - 1.0]
    ], dtype=np.float32)

    src_all = np.vstack([src_pts, boundary_pts])
    dst_all = np.vstack([dst_pts, boundary_pts])

    tri = Delaunay(dst_all)
    y, x = np.mgrid[0:height, 0:width]
    dst_coords = np.column_stack([x.ravel(), y.ravel()])

    simplex_indices = tri.find_simplex(dst_coords)
    valid_mask = simplex_indices >= 0
    src_map_x = np.zeros(dst_coords.shape[0], dtype=np.float32)
    src_map_y = np.zeros(dst_coords.shape[0], dtype=np.float32)

    simps = simplex_indices[valid_mask]
    pts = dst_coords[valid_mask]

    T = tri.transform[simps, :2, :]
    r = tri.transform[simps, 2, :]

    diff = pts - r
    b0 = T[:, 0, 0] * diff[:, 0] + T[:, 0, 1] * diff[:, 1]
    b1 = T[:, 1, 0] * diff[:, 0] + T[:, 1, 1] * diff[:, 1]
    b2 = 1.0 - b0 - b1

    tri_verts = tri.simplices[simps]
    v0 = src_all[tri_verts[:, 0]]
    v1 = src_all[tri_verts[:, 1]]
    v2 = src_all[tri_verts[:, 2]]

    src_mapped = b0[:, None] * v0 + b1[:, None] * v1 + b2[:, None] * v2
    src_map_x[valid_mask] = src_mapped[:, 0]
    src_map_y[valid_mask] = src_mapped[:, 1]

    map_x = src_map_x.reshape((height, width))
    map_y = src_map_y.reshape((height, width))
    return map_x, map_y


def compute_triangle_confidence(src_pts: np.ndarray, dst_pts: np.ndarray, width: int = 128, height: int = 128) -> np.ndarray:
    """
    Computes a continuous geometric correspondence confidence map [128, 128] in [0.10, 1.0].
    Evaluates area distortion ratio and condition number per Delaunay triangle.
    """
    boundary_pts = np.array([
        [0.0, 0.0], [width - 1.0, 0.0], [0.0, height - 1.0], [width - 1.0, height - 1.0],
        [(width - 1.0) / 2.0, 0.0], [0.0, (height - 1.0) / 2.0],
        [width - 1.0, (height - 1.0) / 2.0], [(width - 1.0) / 2.0, height - 1.0]
    ], dtype=np.float32)

    src_all = np.vstack([src_pts, boundary_pts])
    dst_all = np.vstack([dst_pts, boundary_pts])

    tri = Delaunay(dst_all)
    y, x = np.mgrid[0:height, 0:width]
    dst_coords = np.column_stack([x.ravel(), y.ravel()])
    simplex_indices = tri.find_simplex(dst_coords)

    n_tri = len(tri.simplices)
    tri_conf = np.ones(n_tri, dtype=np.float32)

    for i, s in enumerate(tri.simplices):
        t_pts = dst_all[s]
        s_pts = src_all[s]

        a_t = 0.5 * abs((t_pts[1, 0] - t_pts[0, 0]) * (t_pts[2, 1] - t_pts[0, 1]) -
                        (t_pts[2, 0] - t_pts[0, 0]) * (t_pts[1, 1] - t_pts[0, 1]))
        a_s = 0.5 * abs((s_pts[1, 0] - s_pts[0, 0]) * (s_pts[2, 1] - s_pts[0, 1]) -
                        (s_pts[2, 0] - s_pts[0, 0]) * (s_pts[1, 1] - s_pts[0, 1]))

        area_ratio = max(a_t / (a_s + 1e-4), a_s / (a_t + 1e-4))
        v_t = t_pts[1:] - t_pts[0]
        v_s = s_pts[1:] - s_pts[0]
        try:
            m = np.linalg.lstsq(v_s, v_t, rcond=None)[0]
            svals = np.linalg.svd(m, compute_uv=False)
            cond = svals[0] / max(svals[1], 1e-4)
        except Exception:
            cond = 2.0

        distortion = float(np.sqrt(area_ratio * cond))
        conf = float(np.exp(-0.25 * max(0.0, distortion - 1.0)))
        tri_conf[i] = max(0.10, min(1.0, conf))

    conf_map = np.ones(dst_coords.shape[0], dtype=np.float32)
    valid = simplex_indices >= 0
    conf_map[valid] = tri_conf[simplex_indices[valid]]
    conf_2d = conf_map.reshape((height, width))
    conf_2d = cv2.GaussianBlur(conf_2d, (5, 5), 1.0)
    return conf_2d


def preprocess_single_face(preprocessor: RealFacePreprocessor, image_path: str):
    """
    Runs landmark detection once on raw CelebA image and returns aligned image and structures.
    """
    img = Image.open(image_path).convert('RGB')
    det = preprocessor.detect_landmarks(img)
    aligned_img, aligned_5pts, aligned_dense = align_face_similarity(
        image=img,
        detected_landmarks=det.key_landmarks_5pts,
        target_size=128,
        extra_landmarks=det.dense_landmarks
    )
    mask = generate_facial_mask(aligned_dense[MEDIAPIPE_FACE_OVAL_INDICES], 128, preprocessor.feather_radius)
    l_map = generate_landmark_map(aligned_dense, 128)
    tensor = normalize_image_tensor(aligned_img)
    return aligned_img, tensor, mask, l_map, aligned_dense, aligned_5pts


def warp_and_prepare_source(
    src_img_pil: Image.Image,
    s_dense: np.ndarray,
    t_dense: np.ndarray,
    img_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes 6D piecewise affine warp and confidence map.
    """
    src_np = np.array(src_img_pil)
    map_x, map_y = compute_piecewise_affine_map(s_dense, t_dense, img_size, img_size)
    warped_np = cv2.remap(src_np, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    conf_np = compute_triangle_confidence(s_dense, t_dense, img_size, img_size)

    warped_norm = (warped_np.astype(np.float32) / 127.5) - 1.0
    warped_tensor = torch.from_numpy(np.transpose(warped_norm, (2, 0, 1))).unsqueeze(0)
    conf_tensor = torch.from_numpy(conf_np).unsqueeze(0).unsqueeze(0).float()
    return warped_tensor, conf_tensor
