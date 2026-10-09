#!/usr/bin/env python3
"""
Phase 6E: Correspondence-Aware Neural Prototype Training Script
Trains a 100-step controlled neural prototype incorporating the validated
Phase 6D local piecewise-affine source->target correspondence pathway.
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import cv2
from scipy.spatial import Delaunay
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.inference.face_correspondence import compute_triangle_confidence, preprocess_single_face, warp_and_prepare_source
from ml.data.dataset import get_celeba_splits
from ml.training.face_preprocessing import (
    RealFacePreprocessor,
    align_face_similarity,
    generate_facial_mask,
    generate_landmark_map,
    normalize_image_tensor,
    denormalize_image_tensor,
    MEDIAPIPE_FACE_OVAL_INDICES
)
from ml.training.phase6d_landmark_correspondence import (
    compute_piecewise_affine_map,
    compute_pose_metrics
)
from ml.models.phase6e_model import CorrespondenceAwareFaceSwapModel
from ml.models.discriminator import PatchGANDiscriminator, AdversarialLoss
from ml.models.losses import Stage2CompositeLoss








def prepare_validation_pairs(
    val_dataset,
    preprocessor: RealFacePreprocessor,
    num_pairs: int = 10
) -> List[Dict[str, Any]]:
    """Pre-caches fixed cross-ID validation pairs with verified landmarks, aligned source, and confidence map."""
    print(f"Pre-caching {num_pairs} fixed cross-ID validation pairs...", flush=True)
    identities = list(val_dataset.split_identity_to_images.keys())
    cached_val = []
    rng = random.Random(42)

    attempts = 0
    while len(cached_val) < num_pairs and attempts < 100:
        attempts += 1
        id_s, id_t = rng.sample(identities, 2)
        p_s = rng.choice(val_dataset.split_identity_to_images[id_s])
        p_t = rng.choice(val_dataset.split_identity_to_images[id_t])

        try:
            s_img, s_tensor, s_mask, _, s_dense, s_5pts = preprocess_single_face(preprocessor, str(p_s))
            t_img, t_tensor, t_mask, t_lmap, t_dense, t_5pts = preprocess_single_face(preprocessor, str(p_t))

            aln_src_tensor, conf_tensor = warp_and_prepare_source(s_img, s_dense, t_dense, 128)

            pose_s = compute_pose_metrics(s_5pts)
            pose_t = compute_pose_metrics(t_5pts)
            delta_yaw = abs(pose_s["yaw_ratio"] - pose_t["yaw_ratio"])
            delta_pitch = abs(pose_s["pitch_ratio"] - pose_t["pitch_ratio"])
            pose_disp = np.sqrt(delta_yaw**2 + delta_pitch**2)

            cached_val.append({
                "val_idx": len(cached_val),
                "source_id": id_s,
                "target_id": id_t,
                "source_img": s_img,
                "target_img": t_img,
                "s_dense": s_dense,
                "t_dense": t_dense,
                "source_tensor": torch.from_numpy(s_tensor).unsqueeze(0),
                "target_tensor": torch.from_numpy(t_tensor).unsqueeze(0),
                "target_mask": torch.from_numpy(t_mask).unsqueeze(0),
                "target_l_map": torch.from_numpy(t_lmap).unsqueeze(0),
                "aligned_source": aln_src_tensor,
                "confidence_map": conf_tensor,
                "pose_disparity": float(pose_disp)
            })
        except Exception:
            pass

    print(f"Successfully cached {len(cached_val)} validation pairs.", flush=True)
    return cached_val


def prepare_training_pool(
    train_dataset,
    preprocessor: RealFacePreprocessor,
    pool_size: int = 100
) -> List[Dict[str, Any]]:
    """Pre-caches a pool of verified cross-ID training pairs."""
    print(f"Pre-caching {pool_size} verified cross-ID training pairs (MediaPipe + 6D warp)...", flush=True)
    t0 = time.time()
    identities = list(train_dataset.split_identity_to_images.keys())
    pool = []
    rng = random.Random(1234)

    attempts = 0
    while len(pool) < pool_size and attempts < pool_size * 2:
        attempts += 1
        id_s, id_t = rng.sample(identities, 2)
        p_s = rng.choice(train_dataset.split_identity_to_images[id_s])
        p_t = rng.choice(train_dataset.split_identity_to_images[id_t])

        try:
            s_img, s_tensor, s_mask, _, s_dense, _ = preprocess_single_face(preprocessor, str(p_s))
            t_img, t_tensor, t_mask, t_lmap, t_dense, _ = preprocess_single_face(preprocessor, str(p_t))

            aln_src_tensor, conf_tensor = warp_and_prepare_source(s_img, s_dense, t_dense, 128)

            pool.append({
                "source": torch.from_numpy(s_tensor).unsqueeze(0),
                "target": torch.from_numpy(t_tensor).unsqueeze(0),
                "target_mask": torch.from_numpy(t_mask).unsqueeze(0),
                "target_landmark_map": torch.from_numpy(t_lmap).unsqueeze(0),
                "aligned_source": aln_src_tensor,
                "confidence_map": conf_tensor
            })
            if len(pool) % 25 == 0:
                print(f"  Cached {len(pool)}/{pool_size} pairs ({time.time() - t0:.1f}s)...", flush=True)
        except Exception:
            pass

    dt = time.time() - t0
    print(f"Cached {len(pool)} training pairs in {dt:.2f}s ({dt/len(pool):.3f}s/pair).", flush=True)
    return pool


def run_validation(
    model: CorrespondenceAwareFaceSwapModel,
    discriminator: nn.Module,
    val_pairs: List[Dict[str, Any]],
    criterion_g: Stage2CompositeLoss,
    criterion_adv: AdversarialLoss,
    preprocessor: RealFacePreprocessor,
    device: torch.device,
    save_samples_dir: Optional[Path] = None,
    step_num: int = 0
) -> Dict[str, Any]:
    """
    Evaluates fixed validation pairs with:
      - Aligned source pathway ENABLED (Mode A)
      - Aligned source pathway DISABLED (Mode B, ablation)
    """
    model.eval()
    discriminator.eval()

    # Mode A: Enabled metrics
    A_list, B_list, C_list, D_list = [], [], [], []
    mask_in_list, mask_out_list, mask_act_list, bg_err_list = [], [], [], []
    redetect_success_count = 0
    landmark_errors = []

    # Mode B: Disabled metrics (ablation)
    A_b_list, B_b_list, D_b_list = [], [], []

    sample_rows = []

    with torch.no_grad():
        for p in val_pairs:
            src = p["source_tensor"].to(device)
            tgt = p["target_tensor"].to(device)
            tgt_m = p["target_mask"].to(device)
            tgt_l = p["target_l_map"].to(device)
            aln_src = p["aligned_source"].to(device)
            conf_m = p["confidence_map"].to(device)

            z_src = model.arcface(src)
            z_tgt = model.arcface(tgt)

            # 1. Mode A: Aligned Source ENABLED
            out_a = model(
                i_target=tgt,
                l_target=tgt_l,
                i_source=src,
                aligned_source=aln_src,
                confidence_map=conf_m,
                disable_aligned_source=False
            )
            i_swap_a = out_a["i_swap"]
            m_pred_a = out_a["m_pred"]
            i_comp_a = out_a["i_composite"]

            z_comp_a = model.arcface(i_comp_a)
            z_swap_a = model.arcface(i_swap_a)

            cos_A = float(F.cosine_similarity(z_src, z_comp_a).item())
            cos_B = float(F.cosine_similarity(z_tgt, z_comp_a).item())
            cos_C = float(F.cosine_similarity(z_src, z_tgt).item())
            cos_D = float(F.cosine_similarity(z_src, z_swap_a).item())

            A_list.append(cos_A)
            B_list.append(cos_B)
            C_list.append(cos_C)
            D_list.append(cos_D)

            m_np = m_pred_a.squeeze().cpu().numpy()
            tm_np = tgt_m.squeeze().cpu().numpy()
            in_face = tm_np > 0.5
            out_face = tm_np < 0.1
            mask_in_list.append(float(m_np[in_face].mean()) if in_face.sum() > 0 else 0.5)
            mask_out_list.append(float(m_np[out_face].mean()) if out_face.sum() > 0 else 0.0)
            mask_act_list.append(float((m_np > 0.5).mean() * 100.0))

            bg_err = float(torch.abs((1.0 - tgt_m) * (i_comp_a - tgt)).mean().item())
            bg_err_list.append(bg_err)

            comp_pil = denormalize_image_tensor(i_comp_a.squeeze(0).cpu().numpy())
            try:
                det_comp = preprocessor.detect_landmarks(comp_pil)
                if det_comp.dense_landmarks is not None:
                    redetect_success_count += 1
                    err = float(np.mean(np.linalg.norm(det_comp.dense_landmarks - p["t_dense"], axis=1)))
                    landmark_errors.append(err)
            except Exception:
                pass

            # 2. Mode B: Aligned Source DISABLED (Ablation)
            out_b = model(
                i_target=tgt,
                l_target=tgt_l,
                i_source=src,
                aligned_source=aln_src,
                confidence_map=conf_m,
                disable_aligned_source=True
            )
            i_comp_b = out_b["i_composite"]
            i_swap_b = out_b["i_swap"]
            z_comp_b = model.arcface(i_comp_b)
            z_swap_b = model.arcface(i_swap_b)

            cos_A_b = float(F.cosine_similarity(z_src, z_comp_b).item())
            cos_B_b = float(F.cosine_similarity(z_tgt, z_comp_b).item())
            cos_D_b = float(F.cosine_similarity(z_src, z_swap_b).item())
            A_b_list.append(cos_A_b)
            B_b_list.append(cos_B_b)
            D_b_list.append(cos_D_b)

            if save_samples_dir is not None and len(sample_rows) < 6:
                s_np = np.array(p["source_img"])
                t_np = np.array(p["target_img"])
                aln_np = denormalize_image_tensor(p["aligned_source"].squeeze(0).numpy())
                aln_np = np.array(aln_np)
                comp_np = np.array(comp_pil)
                mask_vis = (m_np * 255.0).astype(np.uint8)
                mask_vis_rgb = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2RGB)
                row = np.hstack([s_np, t_np, aln_np, comp_np, mask_vis_rgb])
                sample_rows.append(row)

    if save_samples_dir is not None and sample_rows:
        grid = np.vstack(sample_rows)
        grid_path = save_samples_dir / f"step_{step_num:03d}_val_grid.png"
        cv2.imwrite(str(grid_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"   [Step {step_num:03d}] Saved qualitative grid to {grid_path.name}", flush=True)

    a_mean, a_std = float(np.mean(A_list)), float(np.std(A_list))
    b_mean, b_std = float(np.mean(B_list)), float(np.std(B_list))
    c_mean, c_std = float(np.mean(C_list)), float(np.std(C_list))
    d_mean, d_std = float(np.mean(D_list)), float(np.std(D_list))
    a_c_mean = a_mean - c_mean
    d_c_mean = d_mean - c_mean
    a_b_mean = a_mean - b_mean

    a_b_mode_mean = float(np.mean(A_b_list))
    a_c_mode_b = a_b_mode_mean - c_mean
    delta_a_c = a_c_mean - a_c_mode_b

    val_metrics = {
        "step": step_num,
        "A_mean": a_mean, "A_std": a_std,
        "B_mean": b_mean, "B_std": b_std,
        "C_mean": c_mean, "C_std": c_std,
        "D_mean": d_mean, "D_std": d_std,
        "A_minus_C": a_c_mean,
        "D_minus_C": d_c_mean,
        "A_minus_B": a_b_mean,
        "ablation_disabled_A": a_b_mode_mean,
        "ablation_disabled_A_minus_C": a_c_mode_b,
        "delta_A_minus_C_pathway_gain": delta_a_c,
        "mask_inside_face": float(np.mean(mask_in_list)),
        "mask_outside_face": float(np.mean(mask_out_list)),
        "active_mask_pct": float(np.mean(mask_act_list)),
        "bg_preservation_l1": float(np.mean(bg_err_list)),
        "redetect_success_rate": redetect_success_count / len(val_pairs),
        "mean_landmark_error_px": float(np.mean(landmark_errors)) if landmark_errors else float("nan")
    }

    print(f"   [Step {step_num:03d} Validation]", flush=True)
    print(f"     A (Source ID): {a_mean:.4f} +/- {a_std:.4f} | C: {c_mean:.4f} | A-C: {a_c_mean:+.4f}", flush=True)
    print(f"     B (Target ID): {b_mean:.4f} | D: {d_mean:.4f} | D-C: {d_c_mean:+.4f} | A-B: {a_b_mean:+.4f}", flush=True)
    print(f"     Ablation: Pathway Disabled A-C: {a_c_mode_b:+.4f} | Delta A-C (Gain): {delta_a_c:+.4f}", flush=True)
    print(f"     Mask: Inside={val_metrics['mask_inside_face']:.3f}, Outside={val_metrics['mask_outside_face']:.3f}, Active={val_metrics['active_mask_pct']:.1f}%", flush=True)
    print(f"     Landmark Re-Detect: {val_metrics['redetect_success_rate']*100:.1f}%, Error: {val_metrics['mean_landmark_error_px']:.2f} px", flush=True)

    return val_metrics


def train_phase6e(
    dataset_root: str = "ml/data/celeba",
    init_checkpoint: str = "ml/checkpoints/stage2/best_model.pt",
    arcface_checkpoint: str = "ml/models/weights/ms1mv2_iresnet50.pth",
    output_dir: str = "ml/checkpoints/stage2/phase6e_prototype",
    max_steps: int = 100,
    val_interval: int = 25,
    val_samples: int = 10,
    grad_accum_steps: int = 4,
    lr_g: float = 1e-4,
    lr_d: float = 1e-4,
    w_id: float = 10.0,
    w_id_swap: float = 8.0,
    w_struct: float = 5.0,
    w_bg: float = 5.0,
    w_mask: float = 5.0,
    w_adv: float = 0.5,
    seed: int = 42
):
    print("====================================================================", flush=True)
    print("      PHASE 6E: CORRESPONDENCE-AWARE NEURAL PROTOTYPE (100 STEPS)   ", flush=True)
    print("====================================================================", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device:               {device} ({gpu_name})", flush=True)
    if torch.cuda.is_available():
        print(f"Total VRAM:           {torch.cuda.get_device_properties(0).total_memory / (1024**2):.1f} MB", flush=True)
    print(f"Init Checkpoint:      {init_checkpoint}", flush=True)
    print(f"Output Directory:     {output_dir}", flush=True)
    print(f"Max Optimizer Steps:  {max_steps} (Validation every {val_interval} steps)", flush=True)
    print("--------------------------------------------------------------------", flush=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    samples_path = out_path / "samples"
    samples_path.mkdir(parents=True, exist_ok=True)

    # 1. Preprocessor & Datasets
    preprocessor = RealFacePreprocessor(image_size=128)
    train_ds, val_ds = get_celeba_splits(
        dataset_root=dataset_root,
        image_size=128,
        same_identity_probability=0.0,
        seed=seed
    )
    print(f"Train identities: {len(train_ds.split_identity_to_images)}, Val identities: {len(val_ds.split_identity_to_images)}", flush=True)

    # 2. Pre-cache fixed validation pairs
    val_pairs = prepare_validation_pairs(val_ds, preprocessor, num_pairs=val_samples)

    # 3. Pre-cache training pool (100 pairs)
    train_pool = prepare_training_pool(train_ds, preprocessor, pool_size=100)

    # 4. Initialize Models
    print(f"\nInitializing CorrespondenceAwareFaceSwapModel from {init_checkpoint}...", flush=True)
    model = CorrespondenceAwareFaceSwapModel(
        base_channels=64,
        bottleneck_channels=512,
        embedding_dim=512,
        blur_kernel_size=9,
        blur_sigma=3.0,
        arcface_checkpoint_path=arcface_checkpoint
    ).to(device)

    model.load_baseline_checkpoint(init_checkpoint)

    # Freeze ArcFace
    model.arcface.eval()
    for p in model.arcface.parameters():
        p.requires_grad = False

    discriminator = PatchGANDiscriminator(in_channels=3, base_channels=64).to(device)
    init_ckpt = torch.load(init_checkpoint, map_location=device, weights_only=False)
    if "discriminator_state_dict" in init_ckpt and init_ckpt["discriminator_state_dict"] is not None:
        discriminator.load_state_dict(init_ckpt["discriminator_state_dict"])
        print("Loaded baseline Discriminator weights.", flush=True)

    # 5. Losses & Optimizers
    criterion_g = Stage2CompositeLoss(
        arcface_extractor=model.arcface,
        w_id=w_id,
        w_id_swap=w_id_swap,
        w_struct=w_struct,
        w_bg=w_bg,
        w_mask=w_mask,
        w_adv=w_adv
    ).to(device)
    criterion_adv = AdversarialLoss().to(device)

    opt_g = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr_g, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

    scaler_g = torch.amp.GradScaler('cuda')
    scaler_d = torch.amp.GradScaler('cuda')

    # 6. Step 0 Baseline Validation
    print("\n>> Running Initial Baseline Evaluation (Step 0) on Fixed Validation Pairs...", flush=True)
    step_history = []
    val_0 = run_validation(
        model=model,
        discriminator=discriminator,
        val_pairs=val_pairs,
        criterion_g=criterion_g,
        criterion_adv=criterion_adv,
        preprocessor=preprocessor,
        device=device,
        save_samples_dir=samples_path,
        step_num=0
    )
    step_history.append(val_0)

    # Save Step 0 checkpoint
    torch.save({
        "step": 0,
        "model_state_dict": model.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "metrics": val_0
    }, out_path / "step_000.pt")
    print(f"Saved checkpoint: step_000.pt\n", flush=True)

    # 7. Training Loop (Exactly 100 optimizer steps)
    print(f">> Starting Controlled 100-Step Prototype Training...", flush=True)
    t_start = time.time()
    accum_idx = 0
    opt_step = 0

    opt_g.zero_grad()
    opt_d.zero_grad()

    while opt_step < max_steps:
        model.train()
        discriminator.train()
        model.arcface.eval()

        batch = train_pool[accum_idx % len(train_pool)]
        accum_idx += 1

        src = batch["source"].to(device)
        tgt = batch["target"].to(device)
        tgt_m = batch["target_mask"].to(device)
        tgt_l = batch["target_landmark_map"].to(device)
        aln_src = batch["aligned_source"].to(device)
        conf_m = batch["confidence_map"].to(device)

        # ------------------- Forward Generator -------------------
        with torch.amp.autocast('cuda'):
            outputs = model(
                i_target=tgt,
                l_target=tgt_l,
                i_source=src,
                aligned_source=aln_src,
                confidence_map=conf_m,
                disable_aligned_source=False
            )
            i_swap = outputs["i_swap"]
            m_pred = outputs["m_pred"]
            i_composite = outputs["i_composite"]

            d_fake_logits = discriminator(i_composite)
            loss_g_total, loss_dict = criterion_g(
                i_source=src,
                i_target=tgt,
                i_swap=i_swap,
                i_composite=i_composite,
                pred_mask=m_pred,
                target_mask=tgt_m,
                d_fake_logits=d_fake_logits
            )
            loss_g_accum = loss_g_total / grad_accum_steps

        scaler_g.scale(loss_g_accum).backward()

        # ------------------- Forward Discriminator -------------------
        with torch.amp.autocast('cuda'):
            d_real_logits = discriminator(tgt)
            d_fake_logits_detached = discriminator(i_composite.detach())
            loss_d, _, _ = criterion_adv.discriminator_loss(d_real_logits, d_fake_logits_detached)
            loss_d_accum = loss_d / grad_accum_steps

        scaler_d.scale(loss_d_accum).backward()

        # Step optimizer every grad_accum_steps
        if accum_idx % grad_accum_steps == 0:
            scaler_g.step(opt_g)
            scaler_g.update()
            opt_g.zero_grad()

            scaler_d.step(opt_d)
            scaler_d.update()
            opt_d.zero_grad()

            opt_step += 1

            if opt_step % 5 == 0 or opt_step == max_steps:
                elapsed = time.time() - t_start
                rate = opt_step / elapsed
                print(f"Step {opt_step:03d}/{max_steps} | G_Loss={loss_g_total.item():.3f} | D_Loss={loss_d.item():.3f} | Rate={rate:.2f} steps/s", flush=True)

            # Validation & Checkpointing
            if opt_step % val_interval == 0:
                print(f"\n>> Validation at Step {opt_step:03d}...", flush=True)
                val_res = run_validation(
                    model=model,
                    discriminator=discriminator,
                    val_pairs=val_pairs,
                    criterion_g=criterion_g,
                    criterion_adv=criterion_adv,
                    preprocessor=preprocessor,
                    device=device,
                    save_samples_dir=samples_path,
                    step_num=opt_step
                )
                step_history.append(val_res)

                ckpt_name = f"step_{opt_step:03d}.pt"
                torch.save({
                    "step": opt_step,
                    "model_state_dict": model.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "metrics": val_res
                }, out_path / ckpt_name)
                print(f"Saved checkpoint: {ckpt_name}\n", flush=True)

    total_training_time = time.time() - t_start
    print(f"\n====================================================================", flush=True)
    print(f"Completed 100 optimizer steps in {total_training_time:.2f}s ({100/total_training_time:.2f} steps/s).", flush=True)

    peak_alloc = torch.cuda.max_memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0.0
    peak_res = torch.cuda.max_memory_reserved(0) / (1024**2) if torch.cuda.is_available() else 0.0
    print(f"Peak VRAM Allocated: {peak_alloc:.1f} MB", flush=True)
    print(f"Peak VRAM Reserved:  {peak_res:.1f} MB", flush=True)
    print(f"====================================================================", flush=True)

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    metrics_summary = {
        "init_checkpoint": init_checkpoint,
        "max_steps": max_steps,
        "total_runtime_seconds": total_training_time,
        "steps_per_second": 100 / total_training_time,
        "peak_vram_allocated_mb": peak_alloc,
        "peak_vram_reserved_mb": peak_res,
        "step_history": step_history,
        "baseline_step0": step_history[0],
        "final_step100": step_history[-1]
    }

    metrics_file = out_path / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics_summary, f, indent=2, cls=NumpyEncoder)
    print(f"Saved metrics summary to: {metrics_file}", flush=True)

    return metrics_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 6E Correspondence-Aware Neural Prototype")
    parser.add_argument("--dataset-root", default="ml/data/celeba")
    parser.add_argument("--init-checkpoint", default="ml/checkpoints/stage2/best_model.pt")
    parser.add_argument("--arcface-checkpoint", default="ml/models/weights/ms1mv2_iresnet50.pth")
    parser.add_argument("--output-dir", default="ml/checkpoints/stage2/phase6e_prototype")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--val-interval", type=int, default=25)
    parser.add_argument("--val-samples", type=int, default=10)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr-g", type=float, default=1e-4)
    parser.add_argument("--lr-d", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_phase6e(
        dataset_root=args.dataset_root,
        init_checkpoint=args.init_checkpoint,
        arcface_checkpoint=args.arcface_checkpoint,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        val_interval=args.val_interval,
        val_samples=args.val_samples,
        grad_accum_steps=args.grad_accum_steps,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        seed=args.seed
    )
