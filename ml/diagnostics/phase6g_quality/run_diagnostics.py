"""
Phase 6G Visual Quality & Identity Transfer Diagnostic Runner
Performs controlled empirical diagnostic on 100 genuine held-out CelebA cross-ID pairs:
  - Diagnostic 1: Identity Metric Decomposition (A, B, C, D, A-C, A-B, B-C, distributions, percent A>B, A>C, A-C>0)
  - Diagnostic 2: Visual Quality Metrics (Laplacian variance/sharpness, high-freq energy, PSNR, SSIM, face/bg L1, edge error, mask coverage/leakage)
  - Diagnostic 3: 20-Pair Representative Comparison Grids (strong, average, weak, large pose, small pose)
  - Diagnostic 4: Face-Region vs Background Analysis (I_swap vs target inside/outside face; composite vs target inside/outside)
  - Diagnostic 6: Identity vs Sharpness Correlation (Pearson & Spearman)
  - Diagnostic 7: Source/Target Ablations (Normal, z_id=0, aligned_source=0, disable_skips=True, both=0)
"""

import os
import sys
import time
import math
import json
import random
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple

repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats

from ml.data.dataset import CelebAPairedDataset
from ml.models.phase6g_model import MultiScaleCorrespondenceFaceSwapModel
from ml.training.face_preprocessing import (
    RealFacePreprocessor,
    denormalize_image_tensor
)
from ml.inference.face_correspondence import (
    preprocess_single_face,
    warp_and_prepare_source
)
from ml.training.train_phase6f_reconstruction import (
    create_gaussian_window,
    compute_ssim_torch,
    compute_psnr_torch
)

EXPECTED_ARCFACE_SHA = "2B75B93C48B01C78A4263F7295AB2DBF84F85190C51BE45B92C4C0B0AEDCEAA3"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def compute_distribution_stats(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }


def compute_sharpness_and_hf(img_u8: np.ndarray) -> Tuple[float, float]:
    """Computes Laplacian variance (sharpness) and Sobel high-frequency energy."""
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    hf_energy = float(np.mean(sobelx**2 + sobely**2))
    return lap_var, hf_energy


def compute_edge_mae(img1_norm: torch.Tensor, img2_norm: torch.Tensor) -> float:
    """Mean absolute error between spatial gradients."""
    dx1 = img1_norm[:, :, :, 1:] - img1_norm[:, :, :, :-1]
    dx2 = img2_norm[:, :, :, 1:] - img2_norm[:, :, :, :-1]
    dy1 = img1_norm[:, :, 1:, :] - img1_norm[:, :, :-1, :]
    dy2 = img2_norm[:, :, 1:, :] - img2_norm[:, :, :-1, :]
    return float(0.5 * (torch.abs(dx1 - dx2).mean() + torch.abs(dy1 - dy2).mean()).item())


def to_u8(t: torch.Tensor) -> np.ndarray:
    return ((t.squeeze(0).cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)


def main():
    print("=" * 70)
    print("PHASE 6G — VISUAL QUALITY & IDENTITY TRANSFER DIAGNOSTIC SUITE")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Verify ArcFace Checkpoint Integrity
    arcface_p = Path("ml/models/weights/ms1mv2_iresnet50.pth")
    assert arcface_p.exists(), f"ArcFace checkpoint missing: {arcface_p}"
    with open(arcface_p, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest().upper()
    assert sha256 == EXPECTED_ARCFACE_SHA, f"ArcFace SHA-256 mismatch! Got {sha256}"
    print(f"ArcFace SHA-256 verified: {sha256}")

    # 2. Load Phase 6G Model
    ckpt_path = Path("ml/checkpoints/stage2/phase6g/best_model.pt")
    assert ckpt_path.exists(), f"Phase 6G checkpoint missing: {ckpt_path}"
    print(f"Loading Phase 6G checkpoint: {ckpt_path}")

    model = MultiScaleCorrespondenceFaceSwapModel(
        base_channels=64,
        bottleneck_channels=512,
        embedding_dim=512,
        blur_kernel_size=9,
        blur_sigma=3.0,
        arcface_checkpoint_path=str(arcface_p)
    ).to(device)

    state = torch.load(str(ckpt_path), map_location=device)
    model_weights = state["model_state_dict"] if "model_state_dict" in state else state
    model.load_state_dict(model_weights)
    model.eval()
    model.arcface.eval()
    for p in model.parameters():
        p.requires_grad = False
    print("Phase 6G model loaded and verified.")

    preprocessor = RealFacePreprocessor(image_size=128)
    ssim_window = create_gaussian_window(11, 1.5, 3).to(device)

    # 3. Sample 100 genuine held-out cross-ID pairs from CelebA val split
    val_dataset = CelebAPairedDataset(dataset_root="ml/data/celeba", split="val", seed=42)
    identities = list(val_dataset.split_identity_to_images.keys())
    print(f"Validation dataset identities: {len(identities)}")

    num_eval_pairs = 100
    rng = random.Random(42)
    eval_pairs = []
    attempts = 0
    t0_cache = time.time()
    print(f"\nSampling and preprocessing {num_eval_pairs} genuine cross-ID pairs...")

    while len(eval_pairs) < num_eval_pairs and attempts < num_eval_pairs * 15:
        attempts += 1
        id_s, id_t = rng.sample(identities, 2)
        p_s = rng.choice(val_dataset.split_identity_to_images[id_s])
        p_t = rng.choice(val_dataset.split_identity_to_images[id_t])

        try:
            s_img, s_tensor, s_mask, _, s_dense, s_5pts = preprocess_single_face(preprocessor, str(p_s))
            t_img, t_tensor, t_mask, t_lmap, t_dense, t_5pts = preprocess_single_face(preprocessor, str(p_t))

            aln_src_tensor, conf_tensor = warp_and_prepare_source(s_img, s_dense, t_dense, 128)

            m_tensor = torch.from_numpy(t_mask).float()
            if m_tensor.ndim == 2:
                m_tensor = m_tensor.unsqueeze(0)
            elif m_tensor.ndim == 3 and m_tensor.shape[0] != 1:
                m_tensor = m_tensor[:1]

            # Estimate pose difference from normalized 5-point landmarks
            pose_diff = float(np.linalg.norm(s_5pts - t_5pts, axis=1).mean())

            eval_pairs.append({
                "pair_id": len(eval_pairs) + 1,
                "src_path": str(p_s),
                "tgt_path": str(p_t),
                "src_name": Path(p_s).name,
                "tgt_name": Path(p_t).name,
                "id_s": id_s,
                "id_t": id_t,
                "s_img": s_img,
                "t_img": t_img,
                "s_tensor": torch.from_numpy(s_tensor).unsqueeze(0),
                "t_tensor": torch.from_numpy(t_tensor).unsqueeze(0),
                "t_mask": m_tensor.unsqueeze(0),
                "t_lmap": torch.from_numpy(t_lmap).unsqueeze(0),
                "aln_src": aln_src_tensor,
                "conf_map": conf_tensor,
                "s_5pts": s_5pts,
                "t_5pts": t_5pts,
                "pose_diff": pose_diff
            })

            if len(eval_pairs) % 20 == 0 or len(eval_pairs) == num_eval_pairs:
                print(f"  Preprocessed {len(eval_pairs)}/{num_eval_pairs} pairs ({time.time() - t0_cache:.1f}s)...")
        except Exception as e:
            continue

    assert len(eval_pairs) == num_eval_pairs, f"Expected {num_eval_pairs} pairs, got {len(eval_pairs)}"
    print(f"All {num_eval_pairs} pairs preprocessed in {time.time() - t0_cache:.2f}s.\n")

    # =========================================================================
    # EXECUTE EVALUATION ON ALL 100 PAIRS
    # =========================================================================
    results_list = []
    print("Executing Phase 6G inference & diagnostic measurements on 100 pairs...")
    t0_eval = time.time()

    with torch.no_grad():
        for idx, p in enumerate(eval_pairs):
            i_src = p["s_tensor"].to(device)
            i_tgt = p["t_tensor"].to(device)
            l_tgt = p["t_lmap"].to(device)
            aln_src = p["aln_src"].to(device)
            conf_m = p["conf_map"].to(device)
            m_gt = p["t_mask"].to(device) # [1, 1, 128, 128]

            # Model inference
            t_infer0 = time.time()
            outputs = model(
                i_target=i_tgt,
                l_target=l_tgt,
                i_source=i_src,
                aligned_source=aln_src,
                confidence_map=conf_m,
                disable_skips=False
            )
            infer_ms = (time.time() - t_infer0) * 1000.0

            i_swap = outputs["i_swap"]
            m_pred = outputs["m_pred"]
            i_comp = outputs["i_composite"]

            # Identity embeddings
            z_src = model.arcface(i_src)
            z_tgt = model.arcface(i_tgt)
            z_comp = model.arcface(i_comp)
            z_swap = model.arcface(i_swap)
            z_aln = model.arcface(aln_src)

            # ArcFace metrics
            cos_A = float(F.cosine_similarity(z_src, z_comp).item()) # source -> composite
            cos_B = float(F.cosine_similarity(z_tgt, z_comp).item()) # target -> composite
            cos_C = float(F.cosine_similarity(z_src, z_tgt).item())  # source -> target
            cos_D = float(F.cosine_similarity(z_tgt, z_src).item())  # symmetric to C
            cos_D_swap = float(F.cosine_similarity(z_src, z_swap).item()) # source -> raw swap
            cos_B_swap = float(F.cosine_similarity(z_tgt, z_swap).item()) # target -> raw swap
            cos_L_aln = float(F.cosine_similarity(z_src, z_aln).item())   # source -> 6D aligned

            id_gain = cos_A - cos_C
            src_tgt_adv = cos_A - cos_B
            tgt_leakage = cos_B - cos_C
            src_transfer_ratio = (cos_A - cos_C) / max(1.0 - cos_C, 1e-6)

            # Convert to uint8 images
            src_u8 = to_u8(i_src)
            tgt_u8 = to_u8(i_tgt)
            aln_u8 = to_u8(aln_src)
            swap_u8 = to_u8(i_swap)
            comp_u8 = to_u8(i_comp)
            m_pred_np = m_pred.squeeze().cpu().numpy()
            m_gt_np = m_gt.squeeze().cpu().numpy()

            # Sharpness and High-Frequency Energy
            sharp_tgt, hf_tgt = compute_sharpness_and_hf(tgt_u8)
            sharp_aln, hf_aln = compute_sharpness_and_hf(aln_u8)
            sharp_swap, hf_swap = compute_sharpness_and_hf(swap_u8)
            sharp_comp, hf_comp = compute_sharpness_and_hf(comp_u8)

            # Face vs Background L1 difference (Diagnostic 4)
            # Mask binary threshold at 0.5
            face_mask_3d = (m_gt_np > 0.5)[:, :, None]
            bg_mask_3d = ~face_mask_3d

            diff_swap_tgt = np.abs(swap_u8.astype(np.float32) - tgt_u8.astype(np.float32)) / 255.0
            diff_comp_tgt = np.abs(comp_u8.astype(np.float32) - tgt_u8.astype(np.float32)) / 255.0

            swap_diff_face = float(diff_swap_tgt[face_mask_3d.repeat(3, axis=2)].mean()) if face_mask_3d.any() else 0.0
            swap_diff_bg = float(diff_swap_tgt[bg_mask_3d.repeat(3, axis=2)].mean()) if bg_mask_3d.any() else 0.0
            comp_diff_face = float(diff_comp_tgt[face_mask_3d.repeat(3, axis=2)].mean()) if face_mask_3d.any() else 0.0
            comp_diff_bg = float(diff_comp_tgt[bg_mask_3d.repeat(3, axis=2)].mean()) if bg_mask_3d.any() else 0.0

            # Edge error
            edge_err_comp_tgt = compute_edge_mae(i_comp, i_tgt)
            edge_err_swap_aln = compute_edge_mae(i_swap, aln_src)

            # PSNR & SSIM
            psnr_comp_tgt = compute_psnr_torch(i_comp, i_tgt)
            ssim_comp_tgt = float(compute_ssim_torch(i_comp, i_tgt, ssim_window).item())
            psnr_swap_aln = compute_psnr_torch(i_swap, aln_src)
            ssim_swap_aln = float(compute_ssim_torch(i_swap, aln_src, ssim_window).item())

            # Mask coverage & leakage
            mask_mean = float(m_pred_np.mean())
            mask_pct_50 = float((m_pred_np > 0.5).mean() * 100.0)
            # Leakage: mask mass outside GT face mask
            mask_sum = float(m_pred_np.sum()) + 1e-6
            mask_leakage_pct = float(((m_pred_np * (1.0 - m_gt_np)).sum() / mask_sum) * 100.0)

            # Re-detection and landmark error on composite
            comp_pil = Image.fromarray(comp_u8)
            redetect_ok = False
            lm_err = None
            try:
                det = preprocessor.detect_landmarks(comp_pil)
                if det is not None and det.key_landmarks_5pts is not None:
                    redetect_ok = True
                    lm_err = float(np.linalg.norm(det.key_landmarks_5pts - p["t_5pts"], axis=1).mean())
            except Exception:
                pass

            item = {
                "pair_id": p["pair_id"],
                "src_name": p["src_name"],
                "tgt_name": p["tgt_name"],
                "id_s": p["id_s"],
                "id_t": p["id_t"],
                "pose_diff": p["pose_diff"],
                "infer_ms": infer_ms,
                # ArcFace
                "cos_A": cos_A,
                "cos_B": cos_B,
                "cos_C": cos_C,
                "cos_D": cos_D,
                "cos_D_swap": cos_D_swap,
                "cos_B_swap": cos_B_swap,
                "cos_L_aln": cos_L_aln,
                "id_gain": id_gain,
                "src_tgt_adv": src_tgt_adv,
                "tgt_leakage": tgt_leakage,
                "src_transfer_ratio": src_transfer_ratio,
                # Visual Quality
                "sharp_tgt": sharp_tgt,
                "sharp_aln": sharp_aln,
                "sharp_swap": sharp_swap,
                "sharp_comp": sharp_comp,
                "hf_tgt": hf_tgt,
                "hf_aln": hf_aln,
                "hf_swap": hf_swap,
                "hf_comp": hf_comp,
                # Face vs BG difference
                "swap_diff_face": swap_diff_face,
                "swap_diff_bg": swap_diff_bg,
                "comp_diff_face": comp_diff_face,
                "comp_diff_bg": comp_diff_bg,
                # Structural
                "edge_err_comp_tgt": edge_err_comp_tgt,
                "edge_err_swap_aln": edge_err_swap_aln,
                "psnr_comp_tgt": psnr_comp_tgt,
                "ssim_comp_tgt": ssim_comp_tgt,
                "psnr_swap_aln": psnr_swap_aln,
                "ssim_swap_aln": ssim_swap_aln,
                # Mask
                "mask_mean": mask_mean,
                "mask_pct_50": mask_pct_50,
                "mask_leakage_pct": mask_leakage_pct,
                # Re-detection
                "redetect_ok": redetect_ok,
                "lm_err": lm_err,
                # Raw images for grid generation
                "src_u8": src_u8,
                "tgt_u8": tgt_u8,
                "aln_u8": aln_u8,
                "swap_u8": swap_u8,
                "comp_u8": comp_u8,
                "mask_u8": (m_pred_np * 255.0).clip(0, 255).astype(np.uint8)
            }
            results_list.append(item)

            if (idx + 1) % 25 == 0 or (idx + 1) == num_eval_pairs:
                print(f"  Processed {idx + 1}/{num_eval_pairs} pairs ({time.time() - t0_eval:.1f}s)...")

    print(f"Evaluation complete in {time.time() - t0_eval:.2f}s.\n")

    # =========================================================================
    # DIAGNOSTIC 1: IDENTITY METRIC DECOMPOSITION
    # =========================================================================
    A_vals = [r["cos_A"] for r in results_list]
    B_vals = [r["cos_B"] for r in results_list]
    C_vals = [r["cos_C"] for r in results_list]
    D_vals = [r["cos_D"] for r in results_list]
    D_swap_vals = [r["cos_D_swap"] for r in results_list]
    B_swap_vals = [r["cos_B_swap"] for r in results_list]
    L_aln_vals = [r["cos_L_aln"] for r in results_list]

    AC_vals = [r["id_gain"] for r in results_list]
    AB_vals = [r["src_tgt_adv"] for r in results_list]
    BC_vals = [r["tgt_leakage"] for r in results_list]
    ratio_vals = [r["src_transfer_ratio"] for r in results_list]

    pct_A_gt_B = float(np.mean([1.0 if a > b else 0.0 for a, b in zip(A_vals, B_vals)]) * 100.0)
    pct_A_gt_C = float(np.mean([1.0 if a > c else 0.0 for a, c in zip(A_vals, C_vals)]) * 100.0)
    pct_AC_gt_0 = float(np.mean([1.0 if ac > 0 else 0.0 for ac in AC_vals]) * 100.0)
    pct_Dswap_gt_Bswap = float(np.mean([1.0 if d > b else 0.0 for d, b in zip(D_swap_vals, B_swap_vals)]) * 100.0)

    diag1_stats = {
        "A_cos_src_comp": compute_distribution_stats(A_vals),
        "B_cos_tgt_comp": compute_distribution_stats(B_vals),
        "C_cos_src_tgt": compute_distribution_stats(C_vals),
        "D_cos_tgt_src": compute_distribution_stats(D_vals),
        "D_swap_cos_src_swap": compute_distribution_stats(D_swap_vals),
        "B_swap_cos_tgt_swap": compute_distribution_stats(B_swap_vals),
        "L_aln_cos_src_aln": compute_distribution_stats(L_aln_vals),
        "A_minus_C_gain": compute_distribution_stats(AC_vals),
        "A_minus_B_adv": compute_distribution_stats(AB_vals),
        "B_minus_C_leak": compute_distribution_stats(BC_vals),
        "transfer_ratio": compute_distribution_stats(ratio_vals),
        "pct_A_gt_B": pct_A_gt_B,
        "pct_A_gt_C": pct_A_gt_C,
        "pct_AC_gt_0": pct_AC_gt_0,
        "pct_Dswap_gt_Bswap": pct_Dswap_gt_Bswap
    }

    print("--- DIAGNOSTIC 1 SUMMARY ---")
    print(f"  A (src -> comp):   mean={diag1_stats['A_cos_src_comp']['mean']:.4f}, median={diag1_stats['A_cos_src_comp']['median']:.4f}")
    print(f"  B (tgt -> comp):   mean={diag1_stats['B_cos_tgt_comp']['mean']:.4f}, median={diag1_stats['B_cos_tgt_comp']['median']:.4f}")
    print(f"  C (src -> tgt):    mean={diag1_stats['C_cos_src_tgt']['mean']:.4f}, median={diag1_stats['C_cos_src_tgt']['median']:.4f}")
    print(f"  A - C (Gain):      mean={diag1_stats['A_minus_C_gain']['mean']:+.4f}, median={diag1_stats['A_minus_C_gain']['median']:+.4f}")
    print(f"  A - B (Advantage): mean={diag1_stats['A_minus_B_adv']['mean']:+.4f}, median={diag1_stats['A_minus_B_adv']['median']:+.4f}")
    print(f"  B - C (Leakage):   mean={diag1_stats['B_minus_C_leak']['mean']:+.4f}, median={diag1_stats['B_minus_C_leak']['median']:+.4f}")
    print(f"  6D Aligned L:      mean={diag1_stats['L_aln_cos_src_aln']['mean']:.4f}, median={diag1_stats['L_aln_cos_src_aln']['median']:.4f}")
    print(f"  Percentage A > B:  {pct_A_gt_B:.1f}%")
    print(f"  Percentage A > C:  {pct_A_gt_C:.1f}%")
    print(f"  Percentage A-C > 0:{pct_AC_gt_0:.1f}%\n")

    # =========================================================================
    # DIAGNOSTIC 2: VISUAL QUALITY METRICS
    # =========================================================================
    diag2_stats = {
        "sharpness": {
            "target": compute_distribution_stats([r["sharp_tgt"] for r in results_list]),
            "aligned_source": compute_distribution_stats([r["sharp_aln"] for r in results_list]),
            "raw_swap": compute_distribution_stats([r["sharp_swap"] for r in results_list]),
            "composite": compute_distribution_stats([r["sharp_comp"] for r in results_list]),
        },
        "high_freq_energy": {
            "target": compute_distribution_stats([r["hf_tgt"] for r in results_list]),
            "aligned_source": compute_distribution_stats([r["hf_aln"] for r in results_list]),
            "raw_swap": compute_distribution_stats([r["hf_swap"] for r in results_list]),
            "composite": compute_distribution_stats([r["hf_comp"] for r in results_list]),
        },
        "edge_error": {
            "composite_vs_tgt": compute_distribution_stats([r["edge_err_comp_tgt"] for r in results_list]),
            "swap_vs_aln": compute_distribution_stats([r["edge_err_swap_aln"] for r in results_list]),
        },
        "structural_fidelity": {
            "psnr_comp_tgt": compute_distribution_stats([r["psnr_comp_tgt"] for r in results_list]),
            "ssim_comp_tgt": compute_distribution_stats([r["ssim_comp_tgt"] for r in results_list]),
            "psnr_swap_aln": compute_distribution_stats([r["psnr_swap_aln"] for r in results_list]),
            "ssim_swap_aln": compute_distribution_stats([r["ssim_swap_aln"] for r in results_list]),
        },
        "mask": {
            "coverage_mean": compute_distribution_stats([r["mask_mean"] for r in results_list]),
            "coverage_pct_50": compute_distribution_stats([r["mask_pct_50"] for r in results_list]),
            "leakage_pct": compute_distribution_stats([r["mask_leakage_pct"] for r in results_list]),
        },
        "face_redetection": {
            "detection_rate": float(np.mean([1.0 if r["redetect_ok"] else 0.0 for r in results_list]) * 100.0),
            "landmark_error_px": compute_distribution_stats([r["lm_err"] for r in results_list if r["lm_err"] is not None])
        }
    }

    print("--- DIAGNOSTIC 2 SUMMARY ---")
    print(f"  Target Sharpness (Laplacian Var):   {diag2_stats['sharpness']['target']['mean']:.1f}")
    print(f"  6D Aligned Sharpness:               {diag2_stats['sharpness']['aligned_source']['mean']:.1f}")
    print(f"  Raw Swap Sharpness:                 {diag2_stats['sharpness']['raw_swap']['mean']:.1f}")
    print(f"  Composite Sharpness:                {diag2_stats['sharpness']['composite']['mean']:.1f}")
    print(f"  Sharpness Loss (Swap vs Aligned):   {((diag2_stats['sharpness']['raw_swap']['mean'] - diag2_stats['sharpness']['aligned_source']['mean']) / diag2_stats['sharpness']['aligned_source']['mean']) * 100:.1f}%")
    print(f"  Mask Coverage (>0.5):               {diag2_stats['mask']['coverage_pct_50']['mean']:.1f}%")
    print(f"  Mask Leakage outside face:          {diag2_stats['mask']['leakage_pct']['mean']:.2f}%")
    print(f"  Face Redetection Rate:              {diag2_stats['face_redetection']['detection_rate']:.1f}%")
    print(f"  Mean Landmark Error:                {diag2_stats['face_redetection']['landmark_error_px']['mean']:.2f} px\n")

    # =========================================================================
    # DIAGNOSTIC 4: FACE-REGION VS BACKGROUND
    # =========================================================================
    diag4_stats = {
        "swap_diff_face": compute_distribution_stats([r["swap_diff_face"] for r in results_list]),
        "swap_diff_bg": compute_distribution_stats([r["swap_diff_bg"] for r in results_list]),
        "comp_diff_face": compute_distribution_stats([r["comp_diff_face"] for r in results_list]),
        "comp_diff_bg": compute_distribution_stats([r["comp_diff_bg"] for r in results_list]),
    }

    face_delta = diag4_stats["swap_diff_face"]["mean"]
    bg_delta = diag4_stats["swap_diff_bg"]["mean"]
    print("--- DIAGNOSTIC 4 SUMMARY ---")
    print(f"  I_swap vs Target inside face:       {face_delta:.4f} (L1 normalized [0,1])")
    print(f"  I_swap vs Target outside face:      {bg_delta:.4f}")
    print(f"  Composite vs Target inside face:    {diag4_stats['comp_diff_face']['mean']:.4f}")
    print(f"  Composite vs Target outside face:   {diag4_stats['comp_diff_bg']['mean']:.4f}")

    # Classification logic based on empirical evidence
    # A: barely changing target (< 0.05 inside face)
    # B: changing target significantly (> 0.15) but A < B (target dominant)
    # C: transferring source identity (A > B) but losing detail
    # D: successfully transferring identity and retaining detail
    if face_delta < 0.05:
        behavior_class = "A. barely changing the target"
    elif pct_A_gt_B < 30.0:
        if diag2_stats['sharpness']['raw_swap']['mean'] < 0.65 * diag2_stats['sharpness']['aligned_source']['mean']:
            behavior_class = "B & C: changing target face significantly but remaining target-dominant (A < B) while severely losing 6D aligned high-frequency detail"
        else:
            behavior_class = "B. changing the target but not toward source identity"
    elif pct_A_gt_B >= 50.0 and diag2_stats['sharpness']['raw_swap']['mean'] < 0.65 * diag2_stats['sharpness']['aligned_source']['mean']:
        behavior_class = "C. transferring source identity but losing detail (visually blurred)"
    else:
        behavior_class = "D. successfully transferring identity but visually blurred"
    print(f"  Empirical Classification:           {behavior_class}\n")

    # =========================================================================
    # DIAGNOSTIC 6: IDENTITY VS SHARPNESS CORRELATIONS
    # =========================================================================
    sharpness_comp_list = [r["sharp_comp"] for r in results_list]
    sharpness_swap_list = [r["sharp_swap"] for r in results_list]

    pearson_AB_sharp_comp, p_AB_comp = scipy.stats.pearsonr(AB_vals, sharpness_comp_list)
    spearman_AB_sharp_comp, sp_AB_comp = scipy.stats.spearmanr(AB_vals, sharpness_comp_list)

    pearson_AC_sharp_comp, p_AC_comp = scipy.stats.pearsonr(AC_vals, sharpness_comp_list)
    spearman_AC_sharp_comp, sp_AC_comp = scipy.stats.spearmanr(AC_vals, sharpness_comp_list)

    pearson_BC_sharp_comp, p_BC_comp = scipy.stats.pearsonr(BC_vals, sharpness_comp_list)
    spearman_BC_sharp_comp, sp_BC_comp = scipy.stats.spearmanr(BC_vals, sharpness_comp_list)

    # Also correlate with raw swap sharpness
    pearson_AB_sharp_swap, _ = scipy.stats.pearsonr(AB_vals, sharpness_swap_list)
    pearson_AC_sharp_swap, _ = scipy.stats.pearsonr(AC_vals, sharpness_swap_list)

    diag6_stats = {
        "A_minus_B_vs_sharpness_comp": {
            "pearson_r": float(pearson_AB_sharp_comp),
            "pearson_p": float(p_AB_comp),
            "spearman_r": float(spearman_AB_sharp_comp),
            "spearman_p": float(sp_AB_comp),
        },
        "A_minus_C_vs_sharpness_comp": {
            "pearson_r": float(pearson_AC_sharp_comp),
            "pearson_p": float(p_AC_comp),
            "spearman_r": float(spearman_AC_sharp_comp),
            "spearman_p": float(sp_AC_comp),
        },
        "B_minus_C_vs_sharpness_comp": {
            "pearson_r": float(pearson_BC_sharp_comp),
            "pearson_p": float(p_BC_comp),
            "spearman_r": float(spearman_BC_sharp_comp),
            "spearman_p": float(sp_BC_comp),
        },
        "A_minus_B_vs_sharpness_swap": {
            "pearson_r": float(pearson_AB_sharp_swap)
        },
        "A_minus_C_vs_sharpness_swap": {
            "pearson_r": float(pearson_AC_sharp_swap)
        }
    }

    print("--- DIAGNOSTIC 6 SUMMARY ---")
    print(f"  Corr (A - B) vs Composite Sharpness: Pearson r={pearson_AB_sharp_comp:+.4f} (p={p_AB_comp:.3e}), Spearman r={spearman_AB_sharp_comp:+.4f}")
    print(f"  Corr (A - C) vs Composite Sharpness: Pearson r={pearson_AC_sharp_comp:+.4f} (p={p_AC_comp:.3e}), Spearman r={spearman_AC_sharp_comp:+.4f}")
    print(f"  Corr (B - C) vs Composite Sharpness: Pearson r={pearson_BC_sharp_comp:+.4f} (p={p_BC_comp:.3e}), Spearman r={spearman_BC_sharp_comp:+.4f}")
    print(f"  Corr (A - B) vs Raw Swap Sharpness:  Pearson r={pearson_AB_sharp_swap:+.4f}")
    print(f"  Corr (A - C) vs Raw Swap Sharpness:  Pearson r={pearson_AC_sharp_swap:+.4f}\n")

    # =========================================================================
    # DIAGNOSTIC 3: 20 REPRESENTATIVE PAIR COMPARISON GRIDS
    # =========================================================================
    # Sort results to pick categories:
    # 4 strong transfer (highest A - B)
    # 4 average transfer (median A - B)
    # 4 weak transfer (lowest A - B)
    # 4 large pose difference (highest pose_diff)
    # 4 small pose difference (lowest pose_diff)
    sorted_by_adv = sorted(results_list, key=lambda x: x["src_tgt_adv"], reverse=True)
    sorted_by_pose = sorted(results_list, key=lambda x: x["pose_diff"], reverse=True)

    strong_pairs = sorted_by_adv[:4]
    weak_pairs = sorted_by_adv[-4:]
    mid_idx = len(sorted_by_adv) // 2
    avg_pairs = sorted_by_adv[mid_idx - 2:mid_idx + 2]

    large_pose_candidates = [p for p in sorted_by_pose if p["pair_id"] not in {x["pair_id"] for x in strong_pairs + weak_pairs + avg_pairs}]
    large_pose_pairs = large_pose_candidates[:4]

    small_pose_candidates = [p for p in reversed(sorted_by_pose) if p["pair_id"] not in {x["pair_id"] for x in strong_pairs + weak_pairs + avg_pairs + large_pose_pairs}]
    small_pose_pairs = small_pose_candidates[:4]

    selected_20 = []
    for p in strong_pairs:
        p_c = p.copy()
        p_c["category"] = "Strong Transfer (Highest A-B)"
        selected_20.append(p_c)
    for p in avg_pairs:
        p_c = p.copy()
        p_c["category"] = "Average Transfer (Median A-B)"
        selected_20.append(p_c)
    for p in weak_pairs:
        p_c = p.copy()
        p_c["category"] = "Weak Transfer (Target-Dominant / Lowest A-B)"
        selected_20.append(p_c)
    for p in large_pose_pairs:
        p_c = p.copy()
        p_c["category"] = "Large Pose Difference"
        selected_20.append(p_c)
    for p in small_pose_pairs:
        p_c = p.copy()
        p_c["category"] = "Small Pose Difference"
        selected_20.append(p_c)

    print(f"Generating 20 representative comparison grids in ml/diagnostics/phase6g_quality/grids/...")
    grid_dir = Path("ml/diagnostics/phase6g_quality/grids")
    grid_dir.mkdir(parents=True, exist_ok=True)

    grid_metadata = []
    for g_idx, item in enumerate(selected_20):
        src_u8 = item["src_u8"]
        tgt_u8 = item["tgt_u8"]
        aln_u8 = item["aln_u8"]
        swap_u8 = item["swap_u8"]
        comp_u8 = item["comp_u8"]
        m_u8 = np.repeat(item["mask_u8"][:, :, None], 3, axis=2)

        # Build 6-column composite image: [SOURCE | TARGET | 6D ALIGNED | I_SWAP | COMPOSITE | MASK]
        strip = np.concatenate([src_u8, tgt_u8, aln_u8, swap_u8, comp_u8, m_u8], axis=1)

        # Add top header (36px) and bottom stats footer (44px)
        w_total = 128 * 6
        h_header = 36
        h_footer = 48

        header = np.full((h_header, w_total, 3), 24, dtype=np.uint8)
        footer = np.full((h_footer, w_total, 3), 18, dtype=np.uint8)

        full_img_np = np.concatenate([header, strip, footer], axis=0)
        pil_img = Image.fromarray(full_img_np)
        draw = ImageDraw.Draw(pil_img)

        # Draw column labels
        col_names = ["1. SOURCE", "2. TARGET", "3. 6D ALIGNED", "4. I_SWAP", "5. COMPOSITE", "6. MASK"]
        for c_i, name in enumerate(col_names):
            draw.text((c_i * 128 + 12, 10), name, fill=(240, 240, 240))

        # Draw metrics in footer
        cat_str = f"Pair #{item['pair_id']} [{item['category']}]  {item['src_name']} -> {item['tgt_name']}"
        metrics_str1 = f"ArcFace A(Src->Comp): {item['cos_A']:.4f}  |  B(Tgt->Comp): {item['cos_B']:.4f}  |  C(Src->Tgt): {item['cos_C']:.4f}  |  D_swap: {item['cos_D_swap']:.4f}"
        metrics_str2 = f"Gain (A-C): {item['id_gain']:+.4f}  |  Advantage (A-B): {item['src_tgt_adv']:+.4f}  |  Mask: {item['mask_pct_50']:.1f}%  |  LM Err: {item['lm_err'] if item['lm_err'] is not None else -1:.2f}px"

        draw.text((12, h_header + 128 + 4), cat_str, fill=(90, 180, 255))
        draw.text((12, h_header + 128 + 18), metrics_str1, fill=(220, 220, 220))
        draw.text((12, h_header + 128 + 32), metrics_str2, fill=(160, 230, 160) if item['src_tgt_adv'] > 0 else (255, 170, 170))

        filename = f"grid_{g_idx+1:02d}_pair{item['pair_id']}_{item['src_name'].replace('.jpg','')}_{item['tgt_name'].replace('.jpg','')}.png"
        out_grid_path = grid_dir / filename
        pil_img.save(out_grid_path)

        grid_metadata.append({
            "index": g_idx + 1,
            "filename": filename,
            "pair_id": item["pair_id"],
            "category": item["category"],
            "src_name": item["src_name"],
            "tgt_name": item["tgt_name"],
            "cos_A": item["cos_A"],
            "cos_B": item["cos_B"],
            "cos_C": item["cos_C"],
            "cos_D_swap": item["cos_D_swap"],
            "id_gain": item["id_gain"],
            "src_tgt_adv": item["src_tgt_adv"],
            "sharp_comp": item["sharp_comp"],
            "sharp_aln": item["sharp_aln"],
            "mask_pct_50": item["mask_pct_50"],
            "lm_err": item["lm_err"]
        })

    print(f"Saved 20 comparison grids to {grid_dir}.\n")

    # =========================================================================
    # DIAGNOSTIC 7: SOURCE/TARGET INFERENCE ABLATIONS (20 PAIRS SUBSET)
    # =========================================================================
    print("Running Diagnostic 7: Source/Target Ablations on 20-pair subset...")
    ablation_subset = eval_pairs[:20]

    # Conditions to test:
    # 1. Normal Phase 6G
    # 2. Source identity conditioning disabled (z_id = 0)
    # 3. 6D aligned source disabled (aln_src = 0)
    # 4. Source spatial skips disabled (disable_skips = True)
    # 5. Full ablation: z_id = 0 AND disable_skips = True
    ablation_results = {}

    conditions = [
        ("normal_phase6g", {"zero_zid": False, "zero_aln": False, "disable_skips": False}),
        ("source_id_disabled_zid0", {"zero_zid": True, "zero_aln": False, "disable_skips": False}),
        ("aligned_source_disabled", {"zero_zid": False, "zero_aln": True, "disable_skips": False}),
        ("spatial_skips_disabled", {"zero_zid": False, "zero_aln": False, "disable_skips": True}),
        ("both_id_and_skips_disabled", {"zero_zid": True, "zero_aln": False, "disable_skips": True}),
    ]

    with torch.no_grad():
        for cond_name, flags in conditions:
            c_A, c_B, c_C, c_D_swap = [], [], [], []
            c_sharp, c_redetect = [], []

            for p in ablation_subset:
                i_src = p["s_tensor"].to(device)
                i_tgt = p["t_tensor"].to(device)
                l_tgt = p["t_lmap"].to(device)
                aln_src = torch.zeros_like(p["aln_src"].to(device)) if flags["zero_aln"] else p["aln_src"].to(device)
                conf_m = p["conf_map"].to(device)

                # Custom forward pass with ablation flags
                if flags["zero_zid"]:
                    # Temporarily hook or override z_id
                    z_id = torch.zeros((1, 512), device=device)
                    x_tgt = model.prepare_target_input(i_tgt, l_tgt)
                    f_tgt = model.target_encoder(x_tgt)

                    source_skips = None
                    if not flags["disable_skips"] and not flags["zero_aln"]:
                        source_skips = model.source_encoder(aln_src)

                    i_swap, m_pred = model.generator(
                        f_tgt=f_tgt,
                        z_id=z_id,
                        source_skips=source_skips,
                        confidence_map=conf_m,
                        disable_skips=flags["disable_skips"]
                    )
                    i_comp = model.compositor(i_swap, m_pred, i_tgt)
                else:
                    outputs = model(
                        i_target=i_tgt,
                        l_target=l_tgt,
                        i_source=i_src,
                        aligned_source=aln_src,
                        confidence_map=conf_m,
                        disable_skips=flags["disable_skips"]
                    )
                    i_swap = outputs["i_swap"]
                    i_comp = outputs["i_composite"]

                z_src = model.arcface(i_src)
                z_tgt = model.arcface(i_tgt)
                z_comp = model.arcface(i_comp)
                z_swap = model.arcface(i_swap)

                cos_a = float(F.cosine_similarity(z_src, z_comp).item())
                cos_b = float(F.cosine_similarity(z_tgt, z_comp).item())
                cos_c = float(F.cosine_similarity(z_src, z_tgt).item())
                cos_d_sw = float(F.cosine_similarity(z_src, z_swap).item())

                comp_u8 = to_u8(i_comp)
                sharp, _ = compute_sharpness_and_hf(comp_u8)

                # Re-detection
                comp_pil = Image.fromarray(comp_u8)
                det_ok = False
                try:
                    det = preprocessor.detect_landmarks(comp_pil)
                    if det is not None and det.key_landmarks_5pts is not None:
                        det_ok = True
                except Exception:
                    pass

                c_A.append(cos_a)
                c_B.append(cos_b)
                c_C.append(cos_c)
                c_D_swap.append(cos_d_sw)
                c_sharp.append(sharp)
                c_redetect.append(1.0 if det_ok else 0.0)

            ac_list = [a - c for a, c in zip(c_A, c_C)]
            ab_list = [a - b for a, b in zip(c_A, c_B)]

            ablation_results[cond_name] = {
                "cos_A_src_comp": float(np.mean(c_A)),
                "cos_B_tgt_comp": float(np.mean(c_B)),
                "cos_C_src_tgt": float(np.mean(c_C)),
                "cos_D_src_swap": float(np.mean(c_D_swap)),
                "id_gain_A_minus_C": float(np.mean(ac_list)),
                "src_tgt_adv_A_minus_B": float(np.mean(ab_list)),
                "pct_A_gt_B": float(np.mean([1.0 if a > b else 0.0 for a, b in zip(c_A, c_B)]) * 100.0),
                "sharpness": float(np.mean(c_sharp)),
                "face_detection_rate": float(np.mean(c_redetect) * 100.0)
            }

            print(f"  [{cond_name}]")
            print(f"    A: {ablation_results[cond_name]['cos_A_src_comp']:.4f} | B: {ablation_results[cond_name]['cos_B_tgt_comp']:.4f} | Gain (A-C): {ablation_results[cond_name]['id_gain_A_minus_C']:+.4f} | Adv (A-B): {ablation_results[cond_name]['src_tgt_adv_A_minus_B']:+.4f} | Sharp: {ablation_results[cond_name]['sharpness']:.1f}")

    # =========================================================================
    # EXPORT COMPLETE RESULTS JSON
    # =========================================================================
    # Strip heavy image numpy arrays before json serialization
    serializable_results = []
    for r in results_list:
        r_copy = {k: v for k, v in r.items() if not k.endswith("_u8")}
        serializable_results.append(r_copy)

    full_diagnostic_payload = {
        "metadata": {
            "num_pairs": num_eval_pairs,
            "seed": 42,
            "split": "val",
            "model_checkpoint": str(ckpt_path),
            "arcface_sha256": sha256,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(device)
        },
        "diagnostic_1_identity": diag1_stats,
        "diagnostic_2_visual_quality": diag2_stats,
        "diagnostic_4_face_vs_background": {
            "stats": diag4_stats,
            "classification": behavior_class
        },
        "diagnostic_6_correlations": diag6_stats,
        "diagnostic_7_ablations": ablation_results,
        "representative_20_grids": grid_metadata,
        "all_pairs": serializable_results
    }

    out_json_path = Path("ml/diagnostics/phase6g_quality/diagnostic_results.json")
    with open(out_json_path, "w") as f:
        json.dump(full_diagnostic_payload, f, indent=2, cls=NumpyEncoder)
    print(f"\nAll diagnostic results successfully saved to {out_json_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
