#!/usr/bin/env python3
"""
Phase 6D: Source->Target Geometric Correspondence Diagnostic
Tests whether MediaPipe 478-landmark representations establish a stable,
meaningful source->target facial correspondence for genuine CelebA cross-identity pairs.

Compares:
1. Global 5-point Umeyama similarity transform
2. Local piecewise-affine Delaunay landmark warping
Evaluates:
- Landmark reprojection errors (overall and per anatomical region)
- Source identity preservation via ArcFace embeddings (C, G, L, G-C, L-C)
- Target geometry match (re-running MediaPipe on warped images)
- Stratification across similar, moderate, and large pose differences
- Visual inspection grids
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import cv2
from scipy.spatial import Delaunay
import torch
import torch.nn.functional as F
from PIL import Image

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.inference.face_correspondence import compute_piecewise_affine_map
from ml.data.dataset import get_celeba_splits
from ml.training.face_preprocessing import (
    RealFacePreprocessor,
    denormalize_image_tensor,
    estimate_umeyama_similarity_transform,
    MEDIAPIPE_FACE_OVAL_INDICES
)
from ml.models.face_swap_model import ArcFaceIdentityExtractor

# Semantic landmark subsets
SUBSETS = {
    "contour_jaw": MEDIAPIPE_FACE_OVAL_INDICES,  # 36 points
    "left_eyebrow": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],  # 10 points
    "right_eyebrow": [336, 296, 334, 293, 300, 285, 295, 282, 283, 276],  # 10 points
    "left_eye": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],  # 16 points
    "right_eye": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],  # 16 points
    "nose": [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 327, 278, 48, 115, 220, 45, 275],  # 18 points
    "outer_lips": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185],  # 20 points
    "inner_lips": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191],  # 20 points
    "cheeks": [116, 117, 118, 123, 50, 205, 345, 346, 347, 352, 280, 425],  # 12 points
    "central_face": [1, 2, 4, 5, 6, 19, 94, 168, 195, 197, 0, 13, 14, 17],  # 14 points
}




def piecewise_affine_warp(src_np: np.ndarray, src_pts: np.ndarray, tgt_pts: np.ndarray, img_size: int = 128) -> np.ndarray:
    """Warps source image into target landmark geometry using Delaunay piecewise affine mapping."""
    map_x, map_y = compute_piecewise_affine_map(src_pts, tgt_pts, img_size, img_size)
    warped = cv2.remap(src_np, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return warped


def compute_pose_metrics(landmarks_5pts: np.ndarray) -> Dict[str, float]:
    """
    Estimates yaw asymmetry, pitch ratio, and interocular distance from 5 canonical points.
    5 points: [left_eye, right_eye, nose_tip, left_mouth, right_mouth]
    """
    l_eye = landmarks_5pts[0]
    r_eye = landmarks_5pts[1]
    nose = landmarks_5pts[2]
    l_mouth = landmarks_5pts[3]
    r_mouth = landmarks_5pts[4]

    iod = float(np.linalg.norm(r_eye - l_eye))
    iod = max(iod, 1e-4)

    # Yaw asymmetry ratio: difference in distance from nose to each eye normalized by IOD
    d_l = np.linalg.norm(nose - l_eye)
    d_r = np.linalg.norm(nose - r_eye)
    yaw_ratio = float((d_r - d_l) / iod)

    # Pitch ratio: vertical position of nose relative to eye-mouth span
    eye_y = (l_eye[1] + r_eye[1]) / 2.0
    mouth_y = (l_mouth[1] + r_mouth[1]) / 2.0
    span_y = max(abs(mouth_y - eye_y), 1e-4)
    pitch_ratio = float((nose[1] - eye_y) / span_y)

    return {
        "iod": iod,
        "yaw_ratio": yaw_ratio,
        "pitch_ratio": pitch_ratio
    }


def to_arcface_tensor(img_pil: Image.Image, device: torch.device) -> torch.Tensor:
    """Converts PIL RGB Image to normalized tensor [-1, 1] for ArcFace."""
    arr = np.array(img_pil.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).unsqueeze(0).to(device)


def evaluate_phase6d(
    dataset_root: str = "ml/data/celeba",
    arcface_checkpoint: str = "ml/models/weights/ms1mv2_iresnet50.pth",
    output_dir: str = "ml/checkpoints/stage2/phase6d_diagnostic",
    num_pairs: int = 100,
    seed: int = 42
) -> Dict[str, Any]:
    print(f"=== Starting Phase 6D Geometric Correspondence Diagnostic ===")
    print(f"Dataset root: {dataset_root}")
    print(f"ArcFace checkpoint: {arcface_checkpoint}")
    print(f"Target pairs count: {num_pairs}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_path = Path(output_dir)
    samples_path = out_path / "samples"
    samples_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Preprocessor and ArcFace
    pre = RealFacePreprocessor(image_size=128)
    arcface = ArcFaceIdentityExtractor(checkpoint_path=arcface_checkpoint).to(device)
    arcface.eval()

    # 2. Load held-out validation dataset
    _, val_ds = get_celeba_splits(
        dataset_root=dataset_root,
        image_size=128,
        same_identity_probability=0.0,
        seed=seed
    )
    print(f"Validation dataset size: {len(val_ds)}")

    # 3. Collect genuine cross-identity pairs with valid landmarks on both images
    valid_pairs = []
    print("Collecting validation pairs with verified MediaPipe 478-landmarks...")
    idx = 0
    while len(valid_pairs) < num_pairs and idx < len(val_ds):
        try:
            item = val_ds[idx]
            src_img = denormalize_image_tensor(item["source"].numpy())
            tgt_img = denormalize_image_tensor(item["target"].numpy())

            det_s = pre.detect_landmarks(src_img)
            det_t = pre.detect_landmarks(tgt_img)

            if det_s.dense_landmarks is not None and det_t.dense_landmarks is not None:
                if len(det_s.dense_landmarks) == 478 and len(det_t.dense_landmarks) == 478:
                    valid_pairs.append({
                        "index": idx,
                        "src_img": src_img,
                        "tgt_img": tgt_img,
                        "det_s": det_s,
                        "det_t": det_t,
                        "source_id": item["source_identity"],
                        "target_id": item["target_identity"]
                    })
        except Exception:
            pass
        idx += 1

    print(f"Successfully collected {len(valid_pairs)} verified cross-identity pairs.")

    # 4. Process each pair: Global transform, Piecewise warp, ArcFace, Re-detection
    pair_results = []
    t_start = time.time()

    for p_idx, pair in enumerate(valid_pairs):
        src_img = pair["src_img"]
        tgt_img = pair["tgt_img"]
        det_s = pair["det_s"]
        det_t = pair["det_t"]

        S = det_s.dense_landmarks  # [478, 2]
        T = det_t.dense_landmarks  # [478, 2]
        S_5 = det_s.key_landmarks_5pts  # [5, 2]
        T_5 = det_t.key_landmarks_5pts  # [5, 2]

        pose_s = compute_pose_metrics(S_5)
        pose_t = compute_pose_metrics(T_5)

        # Pose disparity metrics
        iod_t = pose_t["iod"]
        delta_yaw = abs(pose_s["yaw_ratio"] - pose_t["yaw_ratio"])
        delta_pitch = abs(pose_s["pitch_ratio"] - pose_t["pitch_ratio"])
        pose_disparity = np.sqrt(delta_yaw**2 + delta_pitch**2)
        dist_5pts = np.mean(np.linalg.norm(S_5 - T_5, axis=1))
        norm_dist_5pts = dist_5pts / iod_t

        # A. Global 5-point Umeyama similarity transform
        M_global = estimate_umeyama_similarity_transform(S_5, T_5)
        m_3x3 = np.vstack([M_global, [0, 0, 1]])
        inv_m = np.linalg.inv(m_3x3)[0:2, :]
        pil_matrix = (inv_m[0, 0], inv_m[0, 1], inv_m[0, 2], inv_m[1, 0], inv_m[1, 1], inv_m[1, 2])
        global_warped_img = src_img.transform((128, 128), Image.AFFINE, pil_matrix, resample=Image.BILINEAR)

        # Reproject source landmarks under global similarity transform
        ones_s = np.ones((478, 1), dtype=np.float32)
        homo_s = np.hstack([S, ones_s])
        S_global = np.dot(homo_s, M_global.T)

        # Global landmark errors
        global_errors = np.linalg.norm(S_global - T, axis=1)  # [478]
        e_global_mean = float(np.mean(global_errors))
        e_global_median = float(np.median(global_errors))
        e_global_p95 = float(np.percentile(global_errors, 95))
        e_global_norm = float(e_global_mean / iod_t)

        # Subregion global errors
        sub_global_errors = {}
        for sub_name, indices in SUBSETS.items():
            sub_global_errors[sub_name] = float(np.mean(global_errors[indices]))

        # B. Local Piecewise Affine Warp
        src_np = np.array(src_img)
        local_warped_np = piecewise_affine_warp(src_np, S, T, img_size=128)
        local_warped_img = Image.fromarray(local_warped_np)

        # C. Re-detect landmarks on warped images
        # 1. On local warped image
        redetect_success = True
        try:
            det_local = pre.detect_landmarks(local_warped_img)
            local_detected_lms = det_local.dense_landmarks
            local_redetect_errors = np.linalg.norm(local_detected_lms - T, axis=1)
            e_local_mean = float(np.mean(local_redetect_errors))
            e_local_median = float(np.median(local_redetect_errors))
            e_local_p95 = float(np.percentile(local_redetect_errors, 95))
            e_local_norm = float(e_local_mean / iod_t)

            sub_local_errors = {}
            for sub_name, indices in SUBSETS.items():
                sub_local_errors[sub_name] = float(np.mean(local_redetect_errors[indices]))
        except Exception:
            redetect_success = False
            e_local_mean = float("nan")
            e_local_median = float("nan")
            e_local_p95 = float("nan")
            e_local_norm = float("nan")
            sub_local_errors = {k: float("nan") for k in SUBSETS}

        # 2. On global warped image
        try:
            det_glob = pre.detect_landmarks(global_warped_img)
            glob_detected_lms = det_glob.dense_landmarks
            glob_redetect_errors = np.linalg.norm(glob_detected_lms - T, axis=1)
            e_glob_redetect_mean = float(np.mean(glob_redetect_errors))
        except Exception:
            e_glob_redetect_mean = float("nan")

        # D. ArcFace Identity Preservation
        with torch.no_grad():
            t_s = to_arcface_tensor(src_img, device)
            t_t = to_arcface_tensor(tgt_img, device)
            t_g = to_arcface_tensor(global_warped_img, device)
            t_l = to_arcface_tensor(local_warped_img, device)

            z_s = arcface(t_s)
            z_t = arcface(t_t)
            z_g = arcface(t_g)
            z_l = arcface(t_l)

            c = float(F.cosine_similarity(z_s, z_t).item())
            g = float(F.cosine_similarity(z_s, z_g).item())
            l = float(F.cosine_similarity(z_s, z_l).item())
            g_tgt = float(F.cosine_similarity(z_t, z_g).item())
            l_tgt = float(F.cosine_similarity(z_t, z_l).item())

        pair_data = {
            "pair_idx": p_idx,
            "source_id": pair["source_id"],
            "target_id": pair["target_id"],
            "iod_target": iod_t,
            "delta_yaw": delta_yaw,
            "delta_pitch": delta_pitch,
            "pose_disparity": pose_disparity,
            "dist_5pts": dist_5pts,
            "norm_dist_5pts": norm_dist_5pts,
            "C": c,
            "G": g,
            "L": l,
            "G_minus_C": g - c,
            "L_minus_C": l - c,
            "G_target": g_tgt,
            "L_target": l_tgt,
            "global_landmark_err": {
                "mean": e_global_mean,
                "median": e_global_median,
                "p95": e_global_p95,
                "norm": e_global_norm,
                "subregions": sub_global_errors
            },
            "local_landmark_err": {
                "redetect_success": redetect_success,
                "mean": e_local_mean,
                "median": e_local_median,
                "p95": e_local_p95,
                "norm": e_local_norm,
                "subregions": sub_local_errors,
                "glob_redetect_mean": e_glob_redetect_mean
            },
            "src_img": src_img,
            "tgt_img": tgt_img,
            "global_warped_img": global_warped_img,
            "local_warped_img": local_warped_img
        }
        pair_results.append(pair_data)

        if (p_idx + 1) % 20 == 0 or (p_idx + 1) == len(valid_pairs):
            print(f"Processed {p_idx + 1}/{len(valid_pairs)} pairs... Current C={c:.4f}, G={g:.4f}, L={l:.4f}")

    total_time = time.time() - t_start
    print(f"Completed evaluation of {len(pair_results)} pairs in {total_time:.2f}s ({total_time/len(pair_results):.3f}s/pair)")

    # 5. Pose Stratification: Terciles based on pose_disparity
    disparities = [p["pose_disparity"] for p in pair_results]
    t1, t2 = np.percentile(disparities, [33.33, 66.67])
    print(f"Pose disparity tertiles: T1={t1:.4f}, T2={t2:.4f}")

    for p in pair_results:
        disp = p["pose_disparity"]
        if disp <= t1:
            p["pose_category"] = "similar_pose"
        elif disp <= t2:
            p["pose_category"] = "moderate_pose"
        else:
            p["pose_category"] = "large_pose"

    # 6. Aggregate Statistics Calculation
    def aggregate_metrics(pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(pairs)
        if n == 0:
            return {}

        c_vals = [p["C"] for p in pairs]
        g_vals = [p["G"] for p in pairs]
        l_vals = [p["L"] for p in pairs]
        g_minus_c = [p["G_minus_C"] for p in pairs]
        l_minus_c = [p["L_minus_C"] for p in pairs]
        g_tgt_vals = [p["G_target"] for p in pairs]
        l_tgt_vals = [p["L_target"] for p in pairs]

        glob_means = [p["global_landmark_err"]["mean"] for p in pairs]
        glob_medians = [p["global_landmark_err"]["median"] for p in pairs]
        glob_p95s = [p["global_landmark_err"]["p95"] for p in pairs]
        glob_norms = [p["global_landmark_err"]["norm"] for p in pairs]

        valid_local = [p for p in pairs if p["local_landmark_err"]["redetect_success"]]
        local_means = [p["local_landmark_err"]["mean"] for p in valid_local]
        local_medians = [p["local_landmark_err"]["median"] for p in valid_local]
        local_p95s = [p["local_landmark_err"]["p95"] for p in valid_local]
        local_norms = [p["local_landmark_err"]["norm"] for p in valid_local]

        # Region errors
        glob_regions = {}
        local_regions = {}
        for sub_name in SUBSETS:
            glob_regions[sub_name] = float(np.mean([p["global_landmark_err"]["subregions"][sub_name] for p in pairs]))
            if len(valid_local) > 0:
                local_regions[sub_name] = float(np.mean([p["local_landmark_err"]["subregions"][sub_name] for p in valid_local]))
            else:
                local_regions[sub_name] = float("nan")

        return {
            "count": n,
            "redetect_success_rate": len(valid_local) / n,
            "identity": {
                "C_mean": float(np.mean(c_vals)),
                "C_std": float(np.std(c_vals)),
                "G_mean": float(np.mean(g_vals)),
                "G_std": float(np.std(g_vals)),
                "L_mean": float(np.mean(l_vals)),
                "L_std": float(np.std(l_vals)),
                "G_minus_C_mean": float(np.mean(g_minus_c)),
                "L_minus_C_mean": float(np.mean(l_minus_c)),
                "G_target_mean": float(np.mean(g_tgt_vals)),
                "L_target_mean": float(np.mean(l_tgt_vals)),
            },
            "global_landmarks": {
                "mean_px": float(np.mean(glob_means)),
                "median_px": float(np.mean(glob_medians)),
                "p95_px": float(np.mean(glob_p95s)),
                "norm_pct": float(np.mean(glob_norms) * 100.0),
                "regions": glob_regions
            },
            "local_landmarks": {
                "mean_px": float(np.mean(local_means)) if local_means else float("nan"),
                "median_px": float(np.mean(local_medians)) if local_medians else float("nan"),
                "p95_px": float(np.mean(local_p95s)) if local_p95s else float("nan"),
                "norm_pct": float(np.mean(local_norms) * 100.0) if local_norms else float("nan"),
                "regions": local_regions
            }
        }

    overall_stats = aggregate_metrics(pair_results)
    similar_pairs = [p for p in pair_results if p["pose_category"] == "similar_pose"]
    moderate_pairs = [p for p in pair_results if p["pose_category"] == "moderate_pose"]
    large_pairs = [p for p in pair_results if p["pose_category"] == "large_pose"]

    similar_stats = aggregate_metrics(similar_pairs)
    moderate_stats = aggregate_metrics(moderate_pairs)
    large_stats = aggregate_metrics(large_pairs)

    print("\n=== SUMMARY OF RESULTS ===")
    print(f"Overall Pairs: {len(pair_results)}")
    print(f"  C (Source vs Target cosine): {overall_stats['identity']['C_mean']:.4f}")
    print(f"  G (Global warp vs Source):   {overall_stats['identity']['G_mean']:.4f} (G-C: {overall_stats['identity']['G_minus_C_mean']:+.4f})")
    print(f"  L (Local warp vs Source):    {overall_stats['identity']['L_mean']:.4f} (L-C: {overall_stats['identity']['L_minus_C_mean']:+.4f})")
    print(f"  G_tgt (Global warp vs Target): {overall_stats['identity']['G_target_mean']:.4f}")
    print(f"  L_tgt (Local warp vs Target):  {overall_stats['identity']['L_target_mean']:.4f}")
    print(f"  Global landmark error: {overall_stats['global_landmarks']['mean_px']:.2f} px ({overall_stats['global_landmarks']['norm_pct']:.2f}% IOD)")
    print(f"  Local landmark error:  {overall_stats['local_landmarks']['mean_px']:.2f} px ({overall_stats['local_landmarks']['norm_pct']:.2f}% IOD)")
    print(f"  Local redetect rate:   {overall_stats['redetect_success_rate']*100:.1f}%")

    print("\n--- Pose Stratification ---")
    print(f"Similar Pose (N={similar_stats['count']}):   C={similar_stats['identity']['C_mean']:.4f}, L={similar_stats['identity']['L_mean']:.4f} (L-C={similar_stats['identity']['L_minus_C_mean']:+.4f}), LocalErr={similar_stats['local_landmarks']['mean_px']:.2f}px")
    print(f"Moderate Pose (N={moderate_stats['count']}):  C={moderate_stats['identity']['C_mean']:.4f}, L={moderate_stats['identity']['L_mean']:.4f} (L-C={moderate_stats['identity']['L_minus_C_mean']:+.4f}), LocalErr={moderate_stats['local_landmarks']['mean_px']:.2f}px")
    print(f"Large Pose (N={large_stats['count']}):     C={large_stats['identity']['C_mean']:.4f}, L={large_stats['identity']['L_mean']:.4f} (L-C={large_stats['identity']['L_minus_C_mean']:+.4f}), LocalErr={large_stats['local_landmarks']['mean_px']:.2f}px")

    # 7. Generate Visual Diagnostic Grids
    # For each pose category, select representative samples (good, typical, edge/failure)
    print("\nGenerating visual diagnostic sample grids...")
    def make_sample_grid(pair_list: List[Dict[str, Any]], title_prefix: str, max_samples: int = 4) -> str:
        rows = []
        for i in range(min(len(pair_list), max_samples)):
            p = pair_list[i]
            s = np.array(p["src_img"])
            t = np.array(p["tgt_img"])
            g = np.array(p["global_warped_img"])
            l = np.array(p["local_warped_img"])
            # Stack horizontally: [SOURCE, TARGET, GLOBAL_WARP, LOCAL_WARP]
            row = np.hstack([s, t, g, l])
            rows.append(row)

        grid = np.vstack(rows)
        filename = f"{title_prefix}_grid.png"
        filepath = samples_path / filename
        cv2.imwrite(str(filepath), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        return str(filepath)

    similar_grid_path = make_sample_grid(similar_pairs, "similar_pose", 4)
    moderate_grid_path = make_sample_grid(moderate_pairs, "moderate_pose", 4)
    large_grid_path = make_sample_grid(large_pairs, "large_pose", 4)

    # Combined master grid with 6 diverse pairs
    sample_indices = [
        similar_pairs[0]["pair_idx"],
        similar_pairs[1]["pair_idx"] if len(similar_pairs) > 1 else 0,
        moderate_pairs[0]["pair_idx"],
        moderate_pairs[1]["pair_idx"] if len(moderate_pairs) > 1 else 0,
        large_pairs[0]["pair_idx"],
        large_pairs[-1]["pair_idx"] if len(large_pairs) > 1 else 0
    ]
    master_pairs = [pair_results[idx] for idx in sample_indices]
    master_grid_path = make_sample_grid(master_pairs, "master_diagnostic", 6)
    print(f"Saved master grid to: {master_grid_path}")

    # 8. Save Metrics JSON (strip out PIL images for JSON serialization)
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    clean_pair_records = []
    for p in pair_results:
        rec = {k: v for k, v in p.items() if k not in ("src_img", "tgt_img", "global_warped_img", "local_warped_img")}
        clean_pair_records.append(rec)

    metrics_output = {
        "num_pairs_evaluated": len(pair_results),
        "pose_disparity_tertiles": {"t1": float(t1), "t2": float(t2)},
        "overall": overall_stats,
        "similar_pose": similar_stats,
        "moderate_pose": moderate_stats,
        "large_pose": large_stats,
        "pair_records": clean_pair_records
    }

    metrics_file = out_path / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics_output, f, indent=2, cls=NumpyEncoder)
    print(f"Saved metrics to: {metrics_file}")

    return metrics_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 6D Source->Target Geometric Correspondence Diagnostic")
    parser.add_argument("--dataset-root", default="ml/data/celeba", help="Path to CelebA dataset root")
    parser.add_argument("--arcface-checkpoint", default="ml/models/weights/ms1mv2_iresnet50.pth", help="Path to ArcFace weights")
    parser.add_argument("--output-dir", default="ml/checkpoints/stage2/phase6d_diagnostic", help="Output directory")
    parser.add_argument("--num-pairs", type=int, default=100, help="Number of genuine cross-ID pairs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    evaluate_phase6d(
        dataset_root=args.dataset_root,
        arcface_checkpoint=args.arcface_checkpoint,
        output_dir=args.output_dir,
        num_pairs=args.num_pairs,
        seed=args.seed
    )
