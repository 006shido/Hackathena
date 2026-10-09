"""
Phase 6G.1 Comprehensive Evaluation, Baseline Comparison, Ablation & Grid Generation
Evaluates:
  1. Baseline Phase 6G vs Experimental Phase 6G.1 on 100 genuine held-out CelebA pairs
  2. Part 9 Ablation: Phase 6G.1 with 128x128 skip ENABLED vs DISABLED
  3. Part 10 Visual Grids: 20 representative 8-column comparison grids:
     [ SOURCE | TARGET | 6D ALIGNED | BASELINE SWAP | EXP 6G.1 SWAP | BASELINE COMP | EXP 6G.1 COMP | EXP MASK ]
"""

import os
import sys
import time
import json
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple

repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import torch
import torch.nn.functional as F

from ml.models.phase6g_model import MultiScaleCorrespondenceFaceSwapModel
from ml.experiments.phase6g_1_fullres_skip.model_phase6g1 import MultiScaleCorrespondenceFaceSwapModel6G1
from ml.data.dataset import CelebAPairedDataset
from ml.training.face_preprocessing import RealFacePreprocessor
from ml.inference.face_correspondence import (
    preprocess_single_face,
    warp_and_prepare_source
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


def to_u8(t: torch.Tensor) -> np.ndarray:
    return ((t.squeeze(0).cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)


def compute_sharpness_and_hf(img_u8: np.ndarray) -> Tuple[float, float, float]:
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = float(np.mean(np.sqrt(sobelx**2 + sobely**2)))
    hf_energy = float(np.mean(sobelx**2 + sobely**2))
    return lap_var, grad_mag, hf_energy


def main():
    print("=" * 70)
    print("PHASE 6G.1 EVALUATION & BASELINE COMPARISON SUITE")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    arcface_p = Path("ml/models/weights/ms1mv2_iresnet50.pth")
    assert arcface_p.exists()
    with open(arcface_p, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest().upper()
    assert sha256 == EXPECTED_ARCFACE_SHA, f"ArcFace SHA mismatch!"

    # 1. Load Baseline Phase 6G
    base_ckpt = Path("ml/checkpoints/stage2/phase6g/best_model.pt")
    assert base_ckpt.exists()
    print(f"Loading Baseline Phase 6G from: {base_ckpt}")
    base_model = MultiScaleCorrespondenceFaceSwapModel(
        base_channels=64,
        bottleneck_channels=512,
        embedding_dim=512,
        blur_kernel_size=9,
        blur_sigma=3.0,
        arcface_checkpoint_path=str(arcface_p)
    ).to(device)
    base_state = torch.load(str(base_ckpt), map_location=device)
    base_weights = base_state["model_state_dict"] if "model_state_dict" in base_state else base_state
    base_model.load_state_dict(base_weights)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    # 2. Load Experimental Phase 6G.1
    exp_ckpt = Path("ml/experiments/phase6g_1_fullres_skip/checkpoints/best_model.pt")
    if not exp_ckpt.exists():
        exp_ckpt = Path("ml/experiments/phase6g_1_fullres_skip/checkpoints/latest_model.pt")
    assert exp_ckpt.exists(), f"Experimental checkpoint missing: {exp_ckpt}"
    print(f"Loading Experimental Phase 6G.1 from: {exp_ckpt}")

    exp_model = MultiScaleCorrespondenceFaceSwapModel6G1(
        base_channels=64,
        bottleneck_channels=512,
        embedding_dim=512,
        fullres_channels=32,
        blur_kernel_size=9,
        blur_sigma=3.0,
        arcface_checkpoint_path=str(arcface_p)
    ).to(device)
    exp_state = torch.load(str(exp_ckpt), map_location=device)
    exp_weights = exp_state["model_state_dict"] if "model_state_dict" in exp_state else exp_state
    exp_model.load_state_dict(exp_weights)
    exp_model.eval()
    for p in exp_model.parameters():
        p.requires_grad = False

    preprocessor = RealFacePreprocessor(image_size=128)

    # 3. Sample 100 genuine held-out validation pairs
    val_dataset = CelebAPairedDataset(dataset_root="ml/data/celeba", split="val", seed=42)
    identities = list(val_dataset.split_identity_to_images.keys())
    print(f"Sampling 100 genuine cross-ID pairs (seed=42)...")

    num_eval = 100
    rng = random.Random(42)
    eval_pairs = []
    attempts = 0
    t0_cache = time.time()

    while len(eval_pairs) < num_eval and attempts < num_eval * 15:
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
        except Exception:
            continue

    assert len(eval_pairs) == num_eval
    print(f"Preprocessed {num_eval} pairs in {time.time() - t0_cache:.1f}s.")

    # 4. Evaluation Loop
    print("\nRunning inference for Baseline, Experimental, and Ablation on all 100 pairs...")
    pair_records = []
    t0_eval = time.time()

    with torch.no_grad():
        for idx, p in enumerate(eval_pairs):
            i_src = p["s_tensor"].to(device)
            i_tgt = p["t_tensor"].to(device)
            l_tgt = p["t_lmap"].to(device)
            aln_src = p["aln_src"].to(device)
            conf_m = p["conf_map"].to(device)
            m_gt = p["t_mask"].to(device)
            m_gt_np = m_gt.squeeze().cpu().numpy()

            z_src = exp_model.arcface(i_src)
            z_tgt = exp_model.arcface(i_tgt)
            cos_C = float(F.cosine_similarity(z_src, z_tgt).item())

            # (A) Baseline Phase 6G Inference
            base_out = base_model(i_tgt, l_tgt, i_src, aln_src, conf_m, disable_skips=False)
            base_swap = base_out["i_swap"]
            base_comp = base_out["i_composite"]
            base_mask = base_out["m_pred"]

            z_base_comp = exp_model.arcface(base_comp)
            z_base_swap = exp_model.arcface(base_swap)
            base_A = float(F.cosine_similarity(z_src, z_base_comp).item())
            base_B = float(F.cosine_similarity(z_tgt, z_base_comp).item())
            base_D_swap = float(F.cosine_similarity(z_src, z_base_swap).item())

            # (B) Experimental Phase 6G.1 (128x128 Skip ENABLED)
            exp_out = exp_model(i_tgt, l_tgt, i_src, aln_src, conf_m, disable_skips=False, disable_fullres_skip=False)
            exp_swap = exp_out["i_swap"]
            exp_comp = exp_out["i_composite"]
            exp_mask = exp_out["m_pred"]

            z_exp_comp = exp_model.arcface(exp_comp)
            z_exp_swap = exp_model.arcface(exp_swap)
            exp_A = float(F.cosine_similarity(z_src, z_exp_comp).item())
            exp_B = float(F.cosine_similarity(z_tgt, z_exp_comp).item())
            exp_D_swap = float(F.cosine_similarity(z_src, z_exp_swap).item())

            # (C) Ablation Phase 6G.1 (128x128 Skip DISABLED)
            abl_out = exp_model(i_tgt, l_tgt, i_src, aln_src, conf_m, disable_skips=False, disable_fullres_skip=True)
            abl_swap = abl_out["i_swap"]
            abl_comp = abl_out["i_composite"]

            z_abl_comp = exp_model.arcface(abl_comp)
            abl_A = float(F.cosine_similarity(z_src, z_abl_comp).item())
            abl_B = float(F.cosine_similarity(z_tgt, z_abl_comp).item())

            # Convert images to uint8
            src_u8 = to_u8(i_src)
            tgt_u8 = to_u8(i_tgt)
            aln_u8 = to_u8(aln_src)
            base_swap_u8 = to_u8(base_swap)
            exp_swap_u8 = to_u8(exp_swap)
            abl_swap_u8 = to_u8(abl_swap)
            base_comp_u8 = to_u8(base_comp)
            exp_comp_u8 = to_u8(exp_comp)
            abl_comp_u8 = to_u8(abl_comp)
            exp_m_np = exp_mask.squeeze().cpu().numpy()

            # Sharpness and High-Frequency Energy
            sh_tgt, gm_tgt, hf_tgt = compute_sharpness_and_hf(tgt_u8)
            sh_aln, gm_aln, hf_aln = compute_sharpness_and_hf(aln_u8)
            sh_b_swap, gm_b_swap, hf_b_swap = compute_sharpness_and_hf(base_swap_u8)
            sh_e_swap, gm_e_swap, hf_e_swap = compute_sharpness_and_hf(exp_swap_u8)
            sh_abl_swap, _, _ = compute_sharpness_and_hf(abl_swap_u8)
            sh_b_comp, gm_b_comp, hf_b_comp = compute_sharpness_and_hf(base_comp_u8)
            sh_e_comp, gm_e_comp, hf_e_comp = compute_sharpness_and_hf(exp_comp_u8)
            sh_abl_comp, _, _ = compute_sharpness_and_hf(abl_comp_u8)

            # Face vs Background sharpness & diff
            face_mask_bool = (m_gt_np > 0.5)
            bg_mask_bool = ~face_mask_bool

            diff_e_tgt = np.abs(exp_comp_u8.astype(np.float32) - tgt_u8.astype(np.float32)) / 255.0
            face_diff_l1 = float(diff_e_tgt[face_mask_bool[:, :, None].repeat(3, axis=2)].mean())
            bg_diff_l1 = float(diff_e_tgt[bg_mask_bool[:, :, None].repeat(3, axis=2)].mean())

            # Mask coverage & leakage
            mask_cov_50 = float((exp_m_np > 0.5).mean() * 100.0)
            mask_sum = float(exp_m_np.sum()) + 1e-6
            mask_leakage_pct = float(((exp_m_np * (1.0 - m_gt_np)).sum() / mask_sum) * 100.0)

            # Re-detection on experimental composite
            exp_pil = Image.fromarray(exp_comp_u8)
            det_ok = False
            lm_err = None
            try:
                det = preprocessor.detect_landmarks(exp_pil)
                if det is not None and det.key_landmarks_5pts is not None:
                    det_ok = True
                    lm_err = float(np.linalg.norm(det.key_landmarks_5pts - p["t_5pts"], axis=1).mean())
            except Exception:
                pass

            record = {
                "pair_id": p["pair_id"],
                "src_name": p["src_name"],
                "tgt_name": p["tgt_name"],
                "pose_diff": p["pose_diff"],
                "cos_C": cos_C,
                # Baseline 6G
                "base_A": base_A,
                "base_B": base_B,
                "base_D_swap": base_D_swap,
                "base_gain": base_A - cos_C,
                "base_adv": base_A - base_B,
                "base_sh_swap": sh_b_swap,
                "base_sh_comp": sh_b_comp,
                # Experimental 6G.1
                "exp_A": exp_A,
                "exp_B": exp_B,
                "exp_D_swap": exp_D_swap,
                "exp_gain": exp_A - cos_C,
                "exp_adv": exp_A - exp_B,
                "exp_sh_swap": sh_e_swap,
                "exp_sh_comp": sh_e_comp,
                "exp_gm_comp": gm_e_comp,
                "exp_hf_comp": hf_e_comp,
                # Ablation (128x128 skip disabled)
                "abl_A": abl_A,
                "abl_B": abl_B,
                "abl_adv": abl_A - abl_B,
                "abl_sh_swap": sh_abl_swap,
                "abl_sh_comp": sh_abl_comp,
                # Differences
                "adv_delta": (exp_A - exp_B) - (base_A - base_B),
                "sh_swap_delta": sh_e_swap - sh_b_swap,
                "sh_comp_delta": sh_e_comp - sh_b_comp,
                "skip128_adv_effect": (exp_A - exp_B) - (abl_A - abl_B),
                "skip128_sh_swap_effect": sh_e_swap - sh_abl_swap,
                # Mask & detection
                "mask_cov_50": mask_cov_50,
                "mask_leakage_pct": mask_leakage_pct,
                "face_diff_l1": face_diff_l1,
                "bg_diff_l1": bg_diff_l1,
                "det_ok": det_ok,
                "lm_err": lm_err,
                # Images for grids
                "src_u8": src_u8,
                "tgt_u8": tgt_u8,
                "aln_u8": aln_u8,
                "base_swap_u8": base_swap_u8,
                "exp_swap_u8": exp_swap_u8,
                "base_comp_u8": base_comp_u8,
                "exp_comp_u8": exp_comp_u8,
                "exp_mask_u8": (exp_m_np * 255.0).clip(0, 255).astype(np.uint8)
            }
            pair_records.append(record)

            if (idx + 1) % 25 == 0 or (idx + 1) == num_eval:
                print(f"  Processed {idx + 1}/{num_eval} pairs ({time.time() - t0_eval:.1f}s)...")

    # 5. Aggregate Quantitative Metrics
    base_A_vals = [r["base_A"] for r in pair_records]
    base_B_vals = [r["base_B"] for r in pair_records]
    base_C_vals = [r["cos_C"] for r in pair_records]
    base_adv_vals = [r["base_adv"] for r in pair_records]
    base_gain_vals = [r["base_gain"] for r in pair_records]
    base_sh_swap_vals = [r["base_sh_swap"] for r in pair_records]
    base_sh_comp_vals = [r["base_sh_comp"] for r in pair_records]

    exp_A_vals = [r["exp_A"] for r in pair_records]
    exp_B_vals = [r["exp_B"] for r in pair_records]
    exp_adv_vals = [r["exp_adv"] for r in pair_records]
    exp_gain_vals = [r["exp_gain"] for r in pair_records]
    exp_sh_swap_vals = [r["exp_sh_swap"] for r in pair_records]
    exp_sh_comp_vals = [r["exp_sh_comp"] for r in pair_records]

    abl_A_vals = [r["abl_A"] for r in pair_records]
    abl_B_vals = [r["abl_B"] for r in pair_records]
    abl_adv_vals = [r["abl_adv"] for r in pair_records]
    abl_sh_swap_vals = [r["abl_sh_swap"] for r in pair_records]
    abl_sh_comp_vals = [r["abl_sh_comp"] for r in pair_records]

    print("\n" + "=" * 70)
    print("QUANTITATIVE COMPARISON: BASELINE PHASE 6G vs EXPERIMENTAL PHASE 6G.1")
    print("=" * 70)
    print(f"  Baseline Phase 6G:")
    print(f"    Mean A (Src->Comp):   {np.mean(base_A_vals):.4f}")
    print(f"    Mean B (Tgt->Comp):   {np.mean(base_B_vals):.4f}")
    print(f"    Mean Advantage (A-B): {np.mean(base_adv_vals):+.4f}")
    print(f"    Mean Gain (A-C):      {np.mean(base_gain_vals):+.4f}")
    print(f"    P(A > B):             {np.mean([1.0 if a > b else 0.0 for a, b in zip(base_A_vals, base_B_vals)]) * 100:.1f}%")
    print(f"    Raw Swap Sharpness:   {np.mean(base_sh_swap_vals):.1f}")
    print(f"    Composite Sharpness:  {np.mean(base_sh_comp_vals):.1f}")

    print(f"\n  Experimental Phase 6G.1 (128x128 Skip ENABLED):")
    print(f"    Mean A (Src->Comp):   {np.mean(exp_A_vals):.4f} (Delta: {np.mean(exp_A_vals) - np.mean(base_A_vals):+.4f})")
    print(f"    Mean B (Tgt->Comp):   {np.mean(exp_B_vals):.4f} (Delta: {np.mean(exp_B_vals) - np.mean(base_B_vals):+.4f})")
    print(f"    Mean Advantage (A-B): {np.mean(exp_adv_vals):+.4f} (Delta: {np.mean(exp_adv_vals) - np.mean(base_adv_vals):+.4f})")
    print(f"    Mean Gain (A-C):      {np.mean(exp_gain_vals):+.4f} (Delta: {np.mean(exp_gain_vals) - np.mean(base_gain_vals):+.4f})")
    print(f"    P(A > B):             {np.mean([1.0 if a > b else 0.0 for a, b in zip(exp_A_vals, exp_B_vals)]) * 100:.1f}%")
    print(f"    Raw Swap Sharpness:   {np.mean(exp_sh_swap_vals):.1f} (Delta: {np.mean(exp_sh_swap_vals) - np.mean(base_sh_swap_vals):+.1f}, +{((np.mean(exp_sh_swap_vals) - np.mean(base_sh_swap_vals))/np.mean(base_sh_swap_vals))*100:.1f}%)")
    print(f"    Composite Sharpness:  {np.mean(exp_sh_comp_vals):.1f} (Delta: {np.mean(exp_sh_comp_vals) - np.mean(base_sh_comp_vals):+.1f}, +{((np.mean(exp_sh_comp_vals) - np.mean(base_sh_comp_vals))/np.mean(base_sh_comp_vals))*100:.1f}%)")

    print(f"\n  Part 9 Ablation: Phase 6G.1 (128x128 Skip DISABLED):")
    print(f"    Ablation A:           {np.mean(abl_A_vals):.4f}")
    print(f"    Ablation B:           {np.mean(abl_B_vals):.4f}")
    print(f"    Ablation Adv (A-B):   {np.mean(abl_adv_vals):+.4f}")
    print(f"    Ablation Swap Sharp:  {np.mean(abl_sh_swap_vals):.1f}")
    print(f"    128 Skip Adv Delta:   {np.mean(exp_adv_vals) - np.mean(abl_adv_vals):+.4f}")
    print(f"    128 Skip Sharp Delta: {np.mean(exp_sh_swap_vals) - np.mean(abl_sh_swap_vals):+.1f}")

    # 6. Part 10: Generate 20 Representative Comparison Grids (8 columns each)
    # [ 1. Source | 2. Target | 3. 6D Aligned | 4. Base Swap | 5. Exp Swap | 6. Base Comp | 7. Exp Comp | 8. Exp Mask ]
    sorted_by_adv = sorted(pair_records, key=lambda x: x["exp_adv"], reverse=True)
    sorted_by_pose = sorted(pair_records, key=lambda x: x["pose_diff"], reverse=True)

    strong_pairs = sorted_by_adv[:4]
    weak_pairs = sorted_by_adv[-4:]
    mid_idx = len(sorted_by_adv) // 2
    avg_pairs = sorted_by_adv[mid_idx - 2:mid_idx + 2]

    taken_ids = {p["pair_id"] for p in strong_pairs + weak_pairs + avg_pairs}
    large_pose_candidates = [p for p in sorted_by_pose if p["pair_id"] not in taken_ids]
    large_pose_pairs = large_pose_candidates[:4]
    taken_ids.update({p["pair_id"] for p in large_pose_pairs})

    small_pose_candidates = [p for p in reversed(sorted_by_pose) if p["pair_id"] not in taken_ids]
    small_pose_pairs = small_pose_candidates[:4]

    selected_20 = []
    for p in strong_pairs:
        c = p.copy()
        c["category"] = "Strong Transfer (Highest A-B)"
        selected_20.append(c)
    for p in avg_pairs:
        c = p.copy()
        c["category"] = "Average Transfer (Median A-B)"
        selected_20.append(c)
    for p in weak_pairs:
        c = p.copy()
        c["category"] = "Weak Transfer (Target Dominant)"
        selected_20.append(c)
    for p in large_pose_pairs:
        c = p.copy()
        c["category"] = "Large Pose Difference"
        selected_20.append(c)
    for p in small_pose_pairs:
        c = p.copy()
        c["category"] = "Small Pose Difference"
        selected_20.append(c)

    grids_dir = Path("ml/experiments/phase6g_1_fullres_skip/grids")
    grids_dir.mkdir(parents=True, exist_ok=True)
    grid_meta_list = []

    print(f"\nGenerating 20 representative 8-column comparison grids in {grids_dir}...")

    for g_idx, item in enumerate(selected_20):
        src_u8 = item["src_u8"]
        tgt_u8 = item["tgt_u8"]
        aln_u8 = item["aln_u8"]
        base_swap_u8 = item["base_swap_u8"]
        exp_swap_u8 = item["exp_swap_u8"]
        base_comp_u8 = item["base_comp_u8"]
        exp_comp_u8 = item["exp_comp_u8"]
        mask_u8 = np.repeat(item["exp_mask_u8"][:, :, None], 3, axis=2)

        strip = np.concatenate([
            src_u8, tgt_u8, aln_u8,
            base_swap_u8, exp_swap_u8,
            base_comp_u8, exp_comp_u8,
            mask_u8
        ], axis=1)

        w_col = 128
        w_total = w_col * 8
        h_header = 36
        h_footer = 52

        header = np.full((h_header, w_total, 3), 25, dtype=np.uint8)
        footer = np.full((h_footer, w_total, 3), 18, dtype=np.uint8)

        full_img_np = np.concatenate([header, strip, footer], axis=0)
        pil_img = Image.fromarray(full_img_np)
        draw = ImageDraw.Draw(pil_img)

        col_labels = [
            "1. SOURCE", "2. TARGET", "3. 6D ALIGNED",
            "4. 6G BASE SWAP", "5. 6G.1 EXP SWAP",
            "6. 6G BASE COMP", "7. 6G.1 EXP COMP", "8. EXP MASK"
        ]
        for c_i, lbl in enumerate(col_labels):
            draw.text((c_i * w_col + 8, 10), lbl, fill=(240, 240, 240))

        title_str = f"Pair #{item['pair_id']} [{item['category']}]  {item['src_name']} -> {item['tgt_name']}  (Pose Diff: {item['pose_diff']:.2f}px)"
        line1 = f"Base 6G: A={item['base_A']:.4f} | B={item['base_B']:.4f} | Adv(A-B)={item['base_adv']:+.4f} | SwapSharp={item['base_sh_swap']:.1f} | CompSharp={item['base_sh_comp']:.1f}"
        line2 = f"Exp 6G.1: A={item['exp_A']:.4f} | B={item['exp_B']:.4f} | Adv(A-B)={item['exp_adv']:+.4f} | SwapSharp={item['exp_sh_swap']:.1f} | CompSharp={item['exp_sh_comp']:.1f} | DeltaAdv={item['adv_delta']:+.4f}"

        draw.text((10, h_header + 128 + 4), title_str, fill=(90, 180, 255))
        draw.text((10, h_header + 128 + 18), line1, fill=(200, 200, 200))
        draw.text((10, h_header + 128 + 32), line2, fill=(140, 240, 140) if item["exp_adv"] > item["base_adv"] else (255, 170, 170))

        filename = f"grid_{g_idx+1:02d}_pair{item['pair_id']}_{item['src_name'].replace('.jpg','')}_{item['tgt_name'].replace('.jpg','')}.png"
        pil_img.save(grids_dir / filename)

        grid_meta_list.append({
            "index": g_idx + 1,
            "filename": filename,
            "pair_id": item["pair_id"],
            "category": item["category"],
            "src_name": item["src_name"],
            "tgt_name": item["tgt_name"],
            "base_A": item["base_A"],
            "base_B": item["base_B"],
            "base_adv": item["base_adv"],
            "base_sh_swap": item["base_sh_swap"],
            "exp_A": item["exp_A"],
            "exp_B": item["exp_B"],
            "exp_adv": item["exp_adv"],
            "exp_sh_swap": item["exp_sh_swap"],
            "adv_delta": item["adv_delta"]
        })

    print(f"Saved 20 grids to {grids_dir}.")

    # 7. Save Full Evaluation JSON
    clean_pairs = []
    for r in pair_records:
        clean_pairs.append({k: v for k, v in r.items() if not k.endswith("_u8")})

    eval_summary = {
        "num_pairs": num_eval,
        "baseline_phase6g": {
            "A_mean": float(np.mean(base_A_vals)),
            "B_mean": float(np.mean(base_B_vals)),
            "C_mean": float(np.mean(base_C_vals)),
            "Adv_mean": float(np.mean(base_adv_vals)),
            "Gain_mean": float(np.mean(base_gain_vals)),
            "pct_A_gt_B": float(np.mean([1.0 if a > b else 0.0 for a, b in zip(base_A_vals, base_B_vals)]) * 100.0),
            "pct_A_gt_C": float(np.mean([1.0 if a > c else 0.0 for a, c in zip(base_A_vals, base_C_vals)]) * 100.0),
            "pct_gain_gt_0": float(np.mean([1.0 if (a - c) > 0 else 0.0 for a, c in zip(base_A_vals, base_C_vals)]) * 100.0),
            "sharpness_swap": compute_distribution_stats(base_sh_swap_vals),
            "sharpness_comp": compute_distribution_stats(base_sh_comp_vals)
        },
        "experimental_phase6g1": {
            "A_mean": float(np.mean(exp_A_vals)),
            "B_mean": float(np.mean(exp_B_vals)),
            "C_mean": float(np.mean(base_C_vals)),
            "Adv_mean": float(np.mean(exp_adv_vals)),
            "Gain_mean": float(np.mean(exp_gain_vals)),
            "pct_A_gt_B": float(np.mean([1.0 if a > b else 0.0 for a, b in zip(exp_A_vals, exp_B_vals)]) * 100.0),
            "pct_A_gt_C": float(np.mean([1.0 if a > c else 0.0 for a, c in zip(exp_A_vals, base_C_vals)]) * 100.0),
            "pct_gain_gt_0": float(np.mean([1.0 if (a - c) > 0 else 0.0 for a, c in zip(exp_A_vals, base_C_vals)]) * 100.0),
            "sharpness_swap": compute_distribution_stats(exp_sh_swap_vals),
            "sharpness_comp": compute_distribution_stats(exp_sh_comp_vals),
            "face_detection_rate": float(np.mean([1.0 if r["det_ok"] else 0.0 for r in pair_records]) * 100.0),
            "landmark_error_px": compute_distribution_stats([r["lm_err"] for r in pair_records if r["lm_err"] is not None]),
            "mask_coverage_50": compute_distribution_stats([r["mask_cov_50"] for r in pair_records]),
            "mask_leakage_pct": compute_distribution_stats([r["mask_leakage_pct"] for r in pair_records]),
            "face_diff_l1": compute_distribution_stats([r["face_diff_l1"] for r in pair_records]),
            "bg_diff_l1": compute_distribution_stats([r["bg_diff_l1"] for r in pair_records])
        },
        "ablation_phase6g1_skip_disabled": {
            "A_mean": float(np.mean(abl_A_vals)),
            "B_mean": float(np.mean(abl_B_vals)),
            "Adv_mean": float(np.mean(abl_adv_vals)),
            "sharpness_swap": compute_distribution_stats(abl_sh_swap_vals),
            "sharpness_comp": compute_distribution_stats(abl_sh_comp_vals),
            "skip128_adv_delta": float(np.mean(exp_adv_vals) - np.mean(abl_adv_vals)),
            "skip128_sharp_swap_delta": float(np.mean(exp_sh_swap_vals) - np.mean(abl_sh_swap_vals))
        },
        "representative_20_grids": grid_meta_list,
        "pairs": clean_pairs
    }

    out_eval_path = Path("ml/experiments/phase6g_1_fullres_skip/evaluation_results.json")
    with open(out_eval_path, "w") as f:
        json.dump(eval_summary, f, indent=2, cls=NumpyEncoder)
    print(f"Saved evaluation results to {out_eval_path}.\n")


if __name__ == "__main__":
    main()
