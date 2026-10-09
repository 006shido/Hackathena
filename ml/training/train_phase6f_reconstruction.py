"""
Phase 6F: Aligned-Source Neural Reconstruction Diagnostic
Investigates whether neural encoder/decoder architectures can reconstruct
the 6D piecewise-affine aligned source face image, comparing:
  - Condition A: Direct Bottleneck Autoencoder (no skips, bilinear)
  - Condition B: Multi-Scale Skip Autoencoder (U-Net style, full skips at 64, 32, 16)
  - Condition C: 6E-Style Bottleneck Autoencoder (4 ResBlocks at 8x8, nearest upsampling)
"""

import os
import sys
import time
import math
import hashlib
import random
import json
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

# Import existing verified models and utilities
from ml.models.phase6f_models import (
    ConditionA_BottleneckAutoencoder,
    ConditionB_MultiScaleSkipAutoencoder,
    ConditionC_6EStyleBottleneckAutoencoder,
)
from ml.models.face_swap_model import ArcFaceIdentityExtractor
from ml.data.dataset import CelebAPairedDataset
from ml.training.face_preprocessing import (
    RealFacePreprocessor,
    align_face_similarity,
    generate_facial_mask,
    MEDIAPIPE_FACE_OVAL_INDICES
)
from ml.training.phase6d_landmark_correspondence import (
    compute_piecewise_affine_map,
    compute_pose_metrics
)
from ml.inference.face_correspondence import (
    preprocess_single_face,
    warp_and_prepare_source
)

EXPECTED_ARCFACE_SHA = "2B75B93C48B01C78A4263F7295AB2DBF84F85190C51BE45B92C4C0B0AEDCEAA3"


# -------------------------------------------------------------
# Metric helper functions
# -------------------------------------------------------------
def create_gaussian_window(window_size: int = 11, sigma: float = 1.5, channels: int = 3) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * (sigma ** 2)))
    g = g / g.sum()
    w1d = g.unsqueeze(1)
    w2d = w1d.mm(w1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = w2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


def compute_ssim_torch(x: torch.Tensor, y: torch.Tensor, window: Optional[torch.Tensor] = None) -> torch.Tensor:
    # x, y in [-1, 1], normalize to [0, 1]
    x_norm = ((x + 1.0) / 2.0).clamp(0.0, 1.0)
    y_norm = ((y + 1.0) / 2.0).clamp(0.0, 1.0)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    b, c, h, w = x_norm.shape
    if window is None or window.device != x_norm.device:
        window = create_gaussian_window(11, 1.5, c).to(x_norm.device)
    mu_x = F.conv2d(x_norm, window, padding=5, groups=c)
    mu_y = F.conv2d(y_norm, window, padding=5, groups=c)
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y
    sigma_x_sq = F.conv2d(x_norm * x_norm, window, padding=5, groups=c) - mu_x_sq
    sigma_y_sq = F.conv2d(y_norm * y_norm, window, padding=5, groups=c) - mu_y_sq
    sigma_xy = F.conv2d(x_norm * y_norm, window, padding=5, groups=c) - mu_xy
    ssim_map = ((2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)) / ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
    return ssim_map.mean()


def compute_psnr_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    # x, y in [-1, 1], range is 2.0
    mse = F.mse_loss(x, y).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10(4.0 / mse)


def compute_gradient_stats(x: torch.Tensor) -> Tuple[float, float, float]:
    """
    Computes horizontal and vertical gradient magnitudes and their ratio.
    dx = diff along columns (x direction)
    dy = diff along rows (y direction)
    """
    dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean().item()
    dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean().item()
    ratio = dx / (dy + 1e-6)
    return dx, dy, ratio


def compute_edge_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean absolute error between spatial gradients of pred and target."""
    dx_p = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dx_t = target[:, :, :, 1:] - target[:, :, :, :-1]
    dy_p = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dy_t = target[:, :, 1:, :] - target[:, :, :-1, :]
    err_x = torch.abs(dx_p - dx_t).mean().item()
    err_y = torch.abs(dy_p - dy_t).mean().item()
    return 0.5 * (err_x + err_y)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# -------------------------------------------------------------
# Data caching helper
# -------------------------------------------------------------
def cache_aligned_sources(
    dataset: CelebAPairedDataset,
    preprocessor: RealFacePreprocessor,
    num_samples: int,
    seed: int,
    desc: str
) -> List[Dict[str, Any]]:
    print(f"Caching {num_samples} samples for {desc} (seed={seed})...", flush=True)
    rng = random.Random(seed)
    identities = list(dataset.split_identity_to_images.keys())
    cached = []
    attempts = 0
    t0 = time.time()

    while len(cached) < num_samples and attempts < num_samples * 10:
        attempts += 1
        id_s, id_t = rng.sample(identities, 2)
        p_s = rng.choice(dataset.split_identity_to_images[id_s])
        p_t = rng.choice(dataset.split_identity_to_images[id_t])

        try:
            s_img, s_tensor, s_mask, _, s_dense, s_5pts = preprocess_single_face(preprocessor, str(p_s))
            t_img, t_tensor, t_mask, _, t_dense, t_5pts = preprocess_single_face(preprocessor, str(p_t))

            aln_src_tensor, conf_tensor = warp_and_prepare_source(s_img, s_dense, t_dense, 128)

            m_tensor = torch.from_numpy(t_mask).float()
            if m_tensor.ndim == 2:
                m_tensor = m_tensor.unsqueeze(0)
            elif m_tensor.ndim == 3 and m_tensor.shape[0] != 1:
                m_tensor = m_tensor[:1]

            cached.append({
                "aligned_source": aln_src_tensor.squeeze(0), # [3, 128, 128]
                "target_mask": m_tensor,                     # [1, 128, 128]
                "target_dense": t_dense,                     # [478, 2]
                "target_5pts": t_5pts,                       # [5, 2]
                "src_img_pil": s_img,
                "tgt_img_pil": t_img
            })

            if len(cached) % 25 == 0 or len(cached) == num_samples:
                elapsed = time.time() - t0
                print(f"  Cached {len(cached)}/{num_samples} ({elapsed:.1f}s, {len(cached)/elapsed:.1f} samples/s)", flush=True)

        except Exception as e:
            continue

    if len(cached) < num_samples:
        print(f"Warning: Only collected {len(cached)}/{num_samples} after {attempts} attempts", flush=True)
    return cached


# -------------------------------------------------------------
# Validation evaluation function
# -------------------------------------------------------------
def evaluate_condition(
    model: nn.Module,
    val_samples: List[Dict[str, Any]],
    device: torch.device,
    batch_size: int = 16,
    skip_args: Optional[Dict[str, bool]] = None
) -> Dict[str, float]:
    model.eval()
    total_l1_list = []
    face_l1_list = []
    bg_l1_list = []
    psnr_list = []
    ssim_list = []
    edge_err_list = []
    dx_list = []
    dy_list = []
    ratio_list = []

    ssim_window = create_gaussian_window(11, 1.5, 3).to(device)

    num_samples = len(val_samples)
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch = val_samples[i:i + batch_size]
            b_aln = torch.stack([s["aligned_source"] for s in batch], dim=0).to(device)
            b_mask = torch.stack([s["target_mask"] for s in batch], dim=0).to(device)
            while b_mask.ndim > 4:
                b_mask = b_mask.squeeze(1)

            if skip_args is not None:
                preds = model(b_aln, **skip_args)
            else:
                preds = model(b_aln)

            # Total L1
            l1 = F.l1_loss(preds, b_aln, reduction='none').mean(dim=(1, 2, 3))
            total_l1_list.extend(l1.cpu().tolist())

            # Face L1
            diff = torch.abs(preds - b_aln)
            face_mask = b_mask.expand(-1, 3, -1, -1)
            f_l1 = (diff * face_mask).sum(dim=(1, 2, 3)) / (face_mask.sum(dim=(1, 2, 3)) + 1e-6)
            face_l1_list.extend(f_l1.cpu().tolist())

            # Background L1
            bg_mask = (1.0 - b_mask).expand(-1, 3, -1, -1)
            b_l1 = (diff * bg_mask).sum(dim=(1, 2, 3)) / (bg_mask.sum(dim=(1, 2, 3)) + 1e-6)
            bg_l1_list.extend(b_l1.cpu().tolist())

            # PSNR & SSIM per sample
            for j in range(preds.shape[0]):
                p_s = preds[j:j + 1]
                t_s = b_aln[j:j + 1]
                psnr_list.append(compute_psnr_torch(p_s, t_s))
                ssim_list.append(compute_ssim_torch(p_s, t_s, ssim_window).item())
                edge_err_list.append(compute_edge_error(p_s, t_s))
                dx, dy, ratio = compute_gradient_stats(p_s)
                dx_list.append(dx)
                dy_list.append(dy)
                ratio_list.append(ratio)

    return {
        "total_l1": float(np.mean(total_l1_list)),
        "face_l1": float(np.mean(face_l1_list)),
        "bg_l1": float(np.mean(bg_l1_list)),
        "psnr": float(np.mean(psnr_list)),
        "ssim": float(np.mean(ssim_list)),
        "edge_error": float(np.mean(edge_err_list)),
        "grad_dx": float(np.mean(dx_list)),
        "grad_dy": float(np.mean(dy_list)),
        "grad_ratio_x_y": float(np.mean(ratio_list))
    }


# -------------------------------------------------------------
# Step 300 ArcFace & Landmark evaluation
# -------------------------------------------------------------
def evaluate_retention_and_landmarks(
    model: nn.Module,
    val_samples: List[Dict[str, Any]],
    arcface_model: nn.Module,
    preprocessor: RealFacePreprocessor,
    device: torch.device,
    skip_args: Optional[Dict[str, bool]] = None
) -> Dict[str, Any]:
    model.eval()
    arcface_model.eval()

    cos_sims = []
    landmark_errors = []
    detected_count = 0
    total_count = len(val_samples)

    for s in val_samples:
        aln = s["aligned_source"].unsqueeze(0).to(device)
        gt_dense = s["target_dense"]  # [478, 2]
        gt_5pts = s["target_5pts"]    # [5, 2]

        with torch.no_grad():
            if skip_args is not None:
                recon = model(aln, **skip_args)
            else:
                recon = model(aln)

            # ArcFace cosine similarity R
            z_aln = arcface_model(aln)
            z_rec = arcface_model(recon)
            sim = (z_aln * z_rec).sum(dim=1).item()
            cos_sims.append(sim)

        # Convert recon tensor to PIL image for MediaPipe
        recon_np = ((recon.squeeze(0).cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        recon_pil = Image.fromarray(recon_np)

        try:
            det = preprocessor.detect_landmarks(recon_pil)
            if det is not None and det.dense_landmarks is not None:
                detected_count += 1
                # Compute Euclidean landmark error across 5 key landmarks
                err_5pts = np.linalg.norm(det.key_landmarks_5pts - gt_5pts, axis=1).mean()
                landmark_errors.append(float(err_5pts))
        except Exception:
            pass

    detection_rate = float(detected_count / total_count)
    mean_lm_err = float(np.mean(landmark_errors)) if landmark_errors else float("nan")

    return {
        "arcface_r_mean": float(np.mean(cos_sims)),
        "arcface_r_std": float(np.std(cos_sims)),
        "arcface_r_min": float(np.min(cos_sims)),
        "arcface_r_max": float(np.max(cos_sims)),
        "detection_rate": detection_rate,
        "landmark_error_mean": mean_lm_err,
        "landmark_error_std": float(np.std(landmark_errors)) if landmark_errors else float("nan"),
        "detected_faces": detected_count,
        "total_faces": total_count
    }


# -------------------------------------------------------------
# Main Diagnostic Routine
# -------------------------------------------------------------
def main():
    print("=" * 60, flush=True)
    print("PHASE 6F: ALIGNED-SOURCE NEURAL RECONSTRUCTION DIAGNOSTIC", flush=True)
    print("=" * 60, flush=True)

    # 1. Hardware setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB", flush=True)
        torch.cuda.reset_peak_memory_stats()

    # 2. Safety / ArcFace Integrity Verification
    arcface_path = Path("ml/models/weights/ms1mv2_iresnet50.pth")
    assert arcface_path.exists(), f"ArcFace weights not found at {arcface_path}"
    with open(arcface_path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest().upper()
    print(f"ArcFace SHA-256: {sha256}", flush=True)
    assert sha256 == EXPECTED_ARCFACE_SHA, f"ArcFace SHA mismatch! Expected {EXPECTED_ARCFACE_SHA}, got {sha256}"

    # Load frozen ArcFace
    arcface = ArcFaceIdentityExtractor(checkpoint_path=str(arcface_path)).to(device)
    arcface.eval()
    print("ArcFace loaded and frozen successfully.", flush=True)

    # 3. Setup output directories
    base_out = Path("ml/checkpoints/stage2/phase6f_reconstruction")
    dir_a = base_out / "condition_A"
    dir_b = base_out / "condition_B"
    dir_c = base_out / "condition_C"
    dir_samples = base_out / "samples"
    for d in [base_out, dir_a, dir_b, dir_c, dir_samples]:
        d.mkdir(parents=True, exist_ok=True)

    # 4. Preprocess / cache datasets
    preprocessor = RealFacePreprocessor()
    val_dataset = CelebAPairedDataset(dataset_root="ml/data/celeba", split="val", seed=42)
    train_dataset = CelebAPairedDataset(dataset_root="ml/data/celeba", split="train", seed=42)

    val_samples = cache_aligned_sources(val_dataset, preprocessor, num_samples=100, seed=42, desc="Validation")
    train_samples = cache_aligned_sources(train_dataset, preprocessor, num_samples=200, seed=123, desc="Training")

    assert len(val_samples) >= 90, f"Expected ~100 validation samples, got {len(val_samples)}"
    assert len(train_samples) >= 150, f"Expected ~200 training samples, got {len(train_samples)}"

    # Ground truth aligned source gradient stats
    gt_dx_list, gt_dy_list, gt_ratio_list = [], [], []
    for s in val_samples:
        dx, dy, ratio = compute_gradient_stats(s["aligned_source"].unsqueeze(0))
        gt_dx_list.append(dx)
        gt_dy_list.append(dy)
        gt_ratio_list.append(ratio)
    gt_grad_stats = {
        "gt_dx": float(np.mean(gt_dx_list)),
        "gt_dy": float(np.mean(gt_dy_list)),
        "gt_ratio_x_y": float(np.mean(gt_ratio_list))
    }
    print(f"Ground Truth Aligned Source Gradients: dx={gt_grad_stats['gt_dx']:.4f}, dy={gt_grad_stats['gt_dy']:.4f}, ratio(dx/dy)={gt_grad_stats['gt_ratio_x_y']:.4f}", flush=True)

    # 5. Define Conditions
    conditions = {
        "A": {
            "name": "Condition A (Direct Bottleneck Autoencoder)",
            "model_cls": ConditionA_BottleneckAutoencoder,
            "save_dir": dir_a,
            "skip_args": None
        },
        "B": {
            "name": "Condition B (Multi-Scale Skip Autoencoder)",
            "model_cls": ConditionB_MultiScaleSkipAutoencoder,
            "save_dir": dir_b,
            "skip_args": None
        },
        "C": {
            "name": "Condition C (6E-Style Bottleneck Autoencoder)",
            "model_cls": ConditionC_6EStyleBottleneckAutoencoder,
            "save_dir": dir_c,
            "skip_args": None
        }
    }

    batch_size = 8
    total_steps = 300
    val_interval = 50
    lr = 5e-4

    results_all: Dict[str, Any] = {
        "hardware": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
        },
        "ground_truth_stats": gt_grad_stats,
        "conditions": {}
    }

    models_trained: Dict[str, nn.Module] = {}

    # 6. Train each condition independently for exactly 300 steps
    for cond_key in ["A", "B", "C"]:
        cond = conditions[cond_key]
        print("\n" + "=" * 60, flush=True)
        print(f"STARTING TRAINING: {cond['name']}", flush=True)
        print("=" * 60, flush=True)

        torch.manual_seed(42)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(42)
            torch.cuda.reset_peak_memory_stats()

        model = cond["model_cls"]().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
        scaler = torch.amp.GradScaler(device.type)

        step_history = []
        t_start = time.time()

        # Step 0 initial validation evaluation
        val_0 = evaluate_condition(model, val_samples, device)
        val_0["step"] = 0
        step_history.append(val_0)
        print(f"[Step   0/300] Val L1: {val_0['total_l1']:.4f} | Face L1: {val_0['face_l1']:.4f} | PSNR: {val_0['psnr']:.2f}dB | SSIM: {val_0['ssim']:.4f}", flush=True)

        # Training loop
        rng_train = random.Random(42)
        model.train()
        train_indices = list(range(len(train_samples)))

        for step in range(1, total_steps + 1):
            batch_idxs = rng_train.sample(train_indices, batch_size)
            b_aln = torch.stack([train_samples[idx]["aligned_source"] for idx in batch_idxs], dim=0).to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device.type):
                pred = model(b_aln)
                loss = F.l1_loss(pred, b_aln)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if step % val_interval == 0:
                val_metrics = evaluate_condition(model, val_samples, device)
                val_metrics["step"] = step
                val_metrics["train_l1"] = float(loss.item())
                step_history.append(val_metrics)
                print(f"[Step {step:3d}/300] Train L1: {loss.item():.4f} | Val L1: {val_metrics['total_l1']:.4f} | Face L1: {val_metrics['face_l1']:.4f} | PSNR: {val_metrics['psnr']:.2f}dB | SSIM: {val_metrics['ssim']:.4f} | EdgeErr: {val_metrics['edge_error']:.4f}", flush=True)

        t_elapsed = time.time() - t_start
        steps_per_sec = total_steps / t_elapsed
        peak_vram_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2) if device.type == "cuda" else 0.0
        peak_vram_res = torch.cuda.max_memory_reserved() / (1024 ** 2) if device.type == "cuda" else 0.0

        print(f"Condition {cond_key} finished in {t_elapsed:.2f}s ({steps_per_sec:.2f} steps/s). Peak VRAM: {peak_vram_alloc:.1f}MB alloc / {peak_vram_res:.1f}MB res", flush=True)

        # Save checkpoint
        ckpt_path = cond["save_dir"] / "model_step300.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}", flush=True)

        # Step 300 detailed evaluation: ArcFace identity retention & MediaPipe landmarks
        print(f"Evaluating ArcFace retention & landmark preservation on 100 validation samples for Condition {cond_key}...", flush=True)
        retention_metrics = evaluate_retention_and_landmarks(model, val_samples, arcface, preprocessor, device)
        print(f"  ArcFace Retention R: {retention_metrics['arcface_r_mean']:.4f} ± {retention_metrics['arcface_r_std']:.4f} (range: [{retention_metrics['arcface_r_min']:.4f}, {retention_metrics['arcface_r_max']:.4f}])", flush=True)
        print(f"  MediaPipe Detection Rate: {retention_metrics['detection_rate'] * 100:.1f}% ({retention_metrics['detected_faces']}/{retention_metrics['total_faces']})", flush=True)
        print(f"  Landmark Error: {retention_metrics['landmark_error_mean']:.2f} px ± {retention_metrics['landmark_error_std']:.2f} px", flush=True)

        models_trained[cond_key] = model

        results_all["conditions"][cond_key] = {
            "name": cond["name"],
            "runtime_sec": float(t_elapsed),
            "steps_per_sec": float(steps_per_sec),
            "peak_vram_alloc_mb": float(peak_vram_alloc),
            "peak_vram_res_mb": float(peak_vram_res),
            "step_history": step_history,
            "final_metrics": step_history[-1],
            "retention": retention_metrics
        }

    # 7. Condition B Multi-Scale Skip Ablations (B1, B2, B3, B4)
    print("\n" + "=" * 60, flush=True)
    print("CONDITION B SKIP ABLATION STUDY (STEP 300)", flush=True)
    print("=" * 60, flush=True)

    ablations = {
        "B1_all_skips": {"use_skip64": True, "use_skip32": True, "use_skip16": True},
        "B2_no_skip64": {"use_skip64": False, "use_skip32": True, "use_skip16": True},
        "B3_no_skip32": {"use_skip64": True, "use_skip32": False, "use_skip16": True},
        "B4_no_skip16": {"use_skip64": True, "use_skip32": True, "use_skip16": False},
        "B5_no_skips_at_all": {"use_skip64": False, "use_skip32": False, "use_skip16": False}
    }

    ablation_results = {}
    model_b = models_trained["B"]
    for abl_key, abl_args in ablations.items():
        print(f"Evaluating {abl_key} (skips: {abl_args})...", flush=True)
        val_m = evaluate_condition(model_b, val_samples, device, skip_args=abl_args)
        ret_m = evaluate_retention_and_landmarks(model_b, val_samples, arcface, preprocessor, device, skip_args=abl_args)
        abl_combined = {**val_m, **ret_m}
        ablation_results[abl_key] = abl_combined
        print(f"  {abl_key}: Val L1={val_m['total_l1']:.4f} | Face L1={val_m['face_l1']:.4f} | PSNR={val_m['psnr']:.2f}dB | SSIM={val_m['ssim']:.4f} | ArcFace R={ret_m['arcface_r_mean']:.4f} | LmErr={ret_m['landmark_error_mean']:.2f}px", flush=True)

    results_all["ablation_study"] = ablation_results

    # 8. Generate Qualitative Visual Grids
    print("\n" + "=" * 60, flush=True)
    print("GENERATING QUALITATIVE VISUAL GRIDS AND CROPS", flush=True)
    print("=" * 60, flush=True)

    # Select 8 diverse validation samples
    sample_indices = [0, 10, 20, 30, 40, 50, 60, 70]
    grid_rows = []

    for s_idx in sample_indices:
        sample = val_samples[s_idx]
        aln = sample["aligned_source"].unsqueeze(0).to(device)

        with torch.no_grad():
            rec_a = models_trained["A"](aln)
            rec_b = models_trained["B"](aln)
            rec_c = models_trained["C"](aln)

        def to_img(t: torch.Tensor) -> np.ndarray:
            arr = ((t.squeeze(0).cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            return arr

        img_gt = to_img(aln)
        img_a = to_img(rec_a)
        img_b = to_img(rec_b)
        img_c = to_img(rec_c)

        # Absolute error map between Condition B and GT, scaled x4 for visibility
        err_b = np.abs(img_b.astype(np.float32) - img_gt.astype(np.float32)) * 4.0
        err_b = np.clip(err_b, 0, 255).astype(np.uint8)

        # Create row: [GT | Cond A | Cond B | Cond C | Error B (x4)]
        row = np.concatenate([img_gt, img_a, img_b, img_c, err_b], axis=1)
        grid_rows.append(row)

    master_grid = np.concatenate(grid_rows, axis=0)

    # Add header labels
    header_h = 32
    col_w = 128
    header = np.full((header_h, col_w * 5, 3), 30, dtype=np.uint8)
    labels = ["6D ALIGNED (GT)", "COND A (BOTTLENECK)", "COND B (MULTI-SCALE)", "COND C (6E BOTTLENECK)", "B ERR MAP (x4)"]
    for i, lbl in enumerate(labels):
        cv2.putText(header, lbl, (i * col_w + 4, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)

    final_master_grid = np.concatenate([header, master_grid], axis=0)
    grid_path = dir_samples / "master_reconstruction_grid.png"
    Image.fromarray(final_master_grid).save(grid_path)
    print(f"Saved master qualitative grid: {grid_path}", flush=True)

    # Generate Detailed Feature Crops (Eyes, Nose, Mouth, Jaw, Forehead, Hairline)
    # Using sample index 0
    s0 = val_samples[0]
    aln0 = s0["aligned_source"].unsqueeze(0).to(device)
    with torch.no_grad():
        r_a = models_trained["A"](aln0)
        r_b = models_trained["B"](aln0)
        r_c = models_trained["C"](aln0)

    gt_np = to_img(aln0)
    ra_np = to_img(r_a)
    rb_np = to_img(r_b)
    rc_np = to_img(r_c)

    crops_def = [
        ("EYES", (40, 68, 24, 104)),       # y1, y2, x1, x2
        ("NOSE", (55, 85, 44, 84)),
        ("MOUTH", (78, 108, 40, 88)),
        ("JAW", (96, 126, 44, 84)),
        ("FOREHEAD", (14, 44, 44, 84)),
        ("HAIRLINE", (4, 34, 34, 94))
    ]

    crop_rows = []
    target_crop_size = (160, 60)  # w, h
    for name, (y1, y2, x1, x2) in crops_def:
        c_gt = cv2.resize(gt_np[y1:y2, x1:x2], target_crop_size, interpolation=cv2.INTER_NEAREST)
        c_a  = cv2.resize(ra_np[y1:y2, x1:x2], target_crop_size, interpolation=cv2.INTER_NEAREST)
        c_b  = cv2.resize(rb_np[y1:y2, x1:x2], target_crop_size, interpolation=cv2.INTER_NEAREST)
        c_c  = cv2.resize(rc_np[y1:y2, x1:x2], target_crop_size, interpolation=cv2.INTER_NEAREST)

        # Label tag on left
        tag = np.full((60, 100, 3), 40, dtype=np.uint8)
        cv2.putText(tag, name, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        c_row = np.concatenate([tag, c_gt, c_a, c_b, c_c], axis=1)
        crop_rows.append(c_row)

    crops_body = np.concatenate(crop_rows, axis=0)
    c_header = np.full((32, 100 + 160 * 4, 3), 30, dtype=np.uint8)
    c_labels = ["REGION", "6D ALIGNED (GT)", "COND A", "COND B (SKIPS)", "COND C (6E)"]
    c_widths = [100, 160, 160, 160, 160]
    x_offset = 0
    for w, lbl in zip(c_widths, c_labels):
        cv2.putText(c_header, lbl, (x_offset + 10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        x_offset += w

    final_crops_grid = np.concatenate([c_header, crops_body], axis=0)
    crops_path = dir_samples / "crops_grid.png"
    Image.fromarray(final_crops_grid).save(crops_path)
    print(f"Saved feature crops grid: {crops_path}", flush=True)

    # 9. Frequency & Artifact Analysis
    print("\n" + "=" * 60, flush=True)
    print("FREQUENCY AND ARTIFACT ANALYSIS", flush=True)
    print("=" * 60, flush=True)

    for cond_key in ["A", "B", "C"]:
        m = results_all["conditions"][cond_key]["final_metrics"]
        print(f"Condition {cond_key}:", flush=True)
        print(f"  Horizontal Gradient dx: {m['grad_dx']:.4f} (GT: {gt_grad_stats['gt_dx']:.4f})", flush=True)
        print(f"  Vertical Gradient dy:   {m['grad_dy']:.4f} (GT: {gt_grad_stats['gt_dy']:.4f})", flush=True)
        print(f"  Gradient Ratio (dx/dy): {m['grad_ratio_x_y']:.4f} (GT: {gt_grad_stats['gt_ratio_x_y']:.4f})", flush=True)

    # Save metrics JSON
    metrics_path = base_out / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results_all, f, indent=2, cls=NumpyEncoder)
    print(f"\nAll metrics successfully saved to {metrics_path}", flush=True)

    # Final post-flight integrity checks
    print("\n" + "=" * 60, flush=True)
    print("FINAL INTEGRITY CHECKS", flush=True)
    print("=" * 60, flush=True)
    with open(arcface_path, "rb") as f:
        final_sha = hashlib.sha256(f.read()).hexdigest().upper()
    print(f"ArcFace SHA-256 post-check: {final_sha}", flush=True)
    assert final_sha == EXPECTED_ARCFACE_SHA, "ArcFace hash altered during run!"

    celeba_images = list(Path("ml/data/celeba/img_align_celeba").glob("*.jpg"))
    print(f"CelebA image count: {len(celeba_images)} (expected 202,599)", flush=True)
    assert len(celeba_images) == 202599, f"CelebA image count changed: {len(celeba_images)}"

    with open("ml/data/celeba/identity_CelebA.txt", "r") as f:
        id_lines = sum(1 for _ in f)
    print(f"identity_CelebA.txt lines: {id_lines} (expected 202,599)", flush=True)
    assert id_lines == 202599, f"identity_CelebA.txt lines altered: {id_lines}"

    print("Phase 6F Training and Evaluation successfully completed.", flush=True)


if __name__ == "__main__":
    main()
