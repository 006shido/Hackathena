"""
Phase 6G.1 Training Script: Full-Resolution (128x128) Source Skip Experiment
Trains a controlled 500-step experimental prototype starting from verified Phase 6G weights:
  - Adds 128x128 full-resolution source skip pathway
  - Evaluates on 100 genuine held-out CelebA cross-ID pairs
  - Saves checkpoints under ml/experiments/phase6g_1_fullres_skip/checkpoints/
  - Preserves all other architectural components, losses, and hyperparameters identical to Phase 6G.
"""

import os
import sys
import time
import math
import hashlib
import random
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.experiments.phase6g_1_fullres_skip.model_phase6g1 import MultiScaleCorrespondenceFaceSwapModel6G1
from ml.models.discriminator import PatchGANDiscriminator, AdversarialLoss
from ml.models.losses import Stage2CompositeLoss
from ml.data.dataset import CelebAPairedDataset
from ml.training.face_preprocessing import (
    RealFacePreprocessor,
    denormalize_image_tensor
)
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


def prepare_validation_pairs(
    val_dataset: CelebAPairedDataset,
    preprocessor: RealFacePreprocessor,
    num_pairs: int = 100,
    seed: int = 42
) -> List[Dict[str, Any]]:
    print(f"Pre-caching {num_pairs} fixed cross-ID validation pairs (seed={seed})...", flush=True)
    identities = list(val_dataset.split_identity_to_images.keys())
    cached_val = []
    rng = random.Random(seed)
    attempts = 0
    t0 = time.time()

    while len(cached_val) < num_pairs and attempts < num_pairs * 15:
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

            cached_val.append({
                "pair_id": len(cached_val) + 1,
                "source_tensor": torch.from_numpy(s_tensor).unsqueeze(0),
                "target_tensor": torch.from_numpy(t_tensor).unsqueeze(0),
                "target_mask": m_tensor.unsqueeze(0),
                "target_l_map": torch.from_numpy(t_lmap).unsqueeze(0),
                "aligned_source": aln_src_tensor,
                "confidence_map": conf_tensor,
                "target_5pts": t_5pts,
                "source_5pts": s_5pts,
                "target_dense": t_dense,
                "src_img_pil": s_img,
                "tgt_img_pil": t_img,
                "src_name": Path(p_s).name,
                "tgt_name": Path(p_t).name,
                "id_s": id_s,
                "id_t": id_t,
                "pose_diff": pose_diff
            })
            if len(cached_val) % 25 == 0 or len(cached_val) == num_pairs:
                print(f"  Cached {len(cached_val)}/{num_pairs} validation pairs ({time.time() - t0:.1f}s)...", flush=True)
        except Exception:
            continue

    print(f"Successfully cached {len(cached_val)} validation pairs in {time.time() - t0:.1f}s.", flush=True)
    return cached_val


def prepare_training_pool(
    train_dataset: CelebAPairedDataset,
    preprocessor: RealFacePreprocessor,
    pool_size: int = 100,
    seed: int = 123
) -> List[Dict[str, Any]]:
    print(f"Pre-caching {pool_size} verified cross-ID training pairs (seed={seed})...", flush=True)
    t0 = time.time()
    identities = list(train_dataset.split_identity_to_images.keys())
    pool = []
    rng = random.Random(seed)
    attempts = 0

    while len(pool) < pool_size and attempts < pool_size * 15:
        attempts += 1
        id_s, id_t = rng.sample(identities, 2)
        p_s = rng.choice(train_dataset.split_identity_to_images[id_s])
        p_t = rng.choice(train_dataset.split_identity_to_images[id_t])

        try:
            s_img, s_tensor, s_mask, _, s_dense, _ = preprocess_single_face(preprocessor, str(p_s))
            t_img, t_tensor, t_mask, t_lmap, t_dense, _ = preprocess_single_face(preprocessor, str(p_t))

            aln_src_tensor, conf_tensor = warp_and_prepare_source(s_img, s_dense, t_dense, 128)

            m_tensor = torch.from_numpy(t_mask).float()
            if m_tensor.ndim == 2:
                m_tensor = m_tensor.unsqueeze(0)
            elif m_tensor.ndim == 3 and m_tensor.shape[0] != 1:
                m_tensor = m_tensor[:1]

            pool.append({
                "source": torch.from_numpy(s_tensor).unsqueeze(0),
                "target": torch.from_numpy(t_tensor).unsqueeze(0),
                "target_mask": m_tensor.unsqueeze(0),
                "target_landmark_map": torch.from_numpy(t_lmap).unsqueeze(0),
                "aligned_source": aln_src_tensor,
                "confidence_map": conf_tensor
            })
            if len(pool) % 25 == 0 or len(pool) == pool_size:
                print(f"  Cached {len(pool)}/{pool_size} training pairs ({time.time() - t0:.1f}s)...", flush=True)
        except Exception:
            continue

    dt = time.time() - t0
    print(f"Cached {len(pool)} training pairs in {dt:.2f}s ({dt/len(pool):.3f}s/pair).", flush=True)
    return pool


def run_validation(
    model: MultiScaleCorrespondenceFaceSwapModel6G1,
    val_pairs: List[Dict[str, Any]],
    preprocessor: RealFacePreprocessor,
    device: torch.device
) -> Dict[str, Any]:
    model.eval()

    A_list, B_list, C_list, D_swap_list = [], [], [], []
    sharp_comp_list, sharp_swap_list = [], []
    mask_cov_list, redetect_list = [], []
    lm_err_list = []

    # Ablation metrics (full-res skip disabled)
    A_no128_list, B_no128_list, sharp_no128_list = [], [], []

    with torch.no_grad():
        for p in val_pairs:
            src = p["source_tensor"].to(device)
            tgt = p["target_tensor"].to(device)
            tgt_l = p["target_l_map"].to(device)
            aln_src = p["aligned_source"].to(device)
            conf_m = p["confidence_map"].to(device)
            gt_5pts = p["target_5pts"]

            z_src = model.arcface(src)
            z_tgt = model.arcface(tgt)

            # 1. Full Mode: 128x128 full-res skip ENABLED
            out = model(
                i_target=tgt,
                l_target=tgt_l,
                i_source=src,
                aligned_source=aln_src,
                confidence_map=conf_m,
                disable_skips=False,
                disable_fullres_skip=False
            )
            i_swap = out["i_swap"]
            m_pred = out["m_pred"]
            i_comp = out["i_composite"]

            z_comp = model.arcface(i_comp)
            z_swap = model.arcface(i_swap)

            cos_A = float(F.cosine_similarity(z_src, z_comp).item())
            cos_B = float(F.cosine_similarity(z_tgt, z_comp).item())
            cos_C = float(F.cosine_similarity(z_src, z_tgt).item())
            cos_D_swap = float(F.cosine_similarity(z_src, z_swap).item())

            A_list.append(cos_A)
            B_list.append(cos_B)
            C_list.append(cos_C)
            D_swap_list.append(cos_D_swap)

            # Sharpness
            comp_u8 = ((i_comp.squeeze(0).cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            swap_u8 = ((i_swap.squeeze(0).cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

            gray_comp = cv2.cvtColor(comp_u8, cv2.COLOR_RGB2GRAY)
            gray_swap = cv2.cvtColor(swap_u8, cv2.COLOR_RGB2GRAY)
            sharp_comp_list.append(float(cv2.Laplacian(gray_comp, cv2.CV_64F).var()))
            sharp_swap_list.append(float(cv2.Laplacian(gray_swap, cv2.CV_64F).var()))

            # Mask coverage
            m_np = m_pred.squeeze().cpu().numpy()
            mask_cov_list.append(float((m_np > 0.5).mean() * 100.0))

            # Landmark re-detection
            comp_pil = Image.fromarray(comp_u8)
            det_ok = False
            try:
                det = preprocessor.detect_landmarks(comp_pil)
                if det is not None and det.key_landmarks_5pts is not None:
                    det_ok = True
                    lm_err = float(np.linalg.norm(det.key_landmarks_5pts - gt_5pts, axis=1).mean())
                    lm_err_list.append(lm_err)
            except Exception:
                pass
            redetect_list.append(1.0 if det_ok else 0.0)

            # 2. Ablation: 128x128 full-res skip DISABLED
            out_no128 = model(
                i_target=tgt,
                l_target=tgt_l,
                i_source=src,
                aligned_source=aln_src,
                confidence_map=conf_m,
                disable_skips=False,
                disable_fullres_skip=True
            )
            z_comp_no128 = model.arcface(out_no128["i_composite"])
            A_no128_list.append(float(F.cosine_similarity(z_src, z_comp_no128).item()))
            B_no128_list.append(float(F.cosine_similarity(z_tgt, z_comp_no128).item()))

            no128_u8 = ((out_no128["i_composite"].squeeze(0).cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            gray_no128 = cv2.cvtColor(no128_u8, cv2.COLOR_RGB2GRAY)
            sharp_no128_list.append(float(cv2.Laplacian(gray_no128, cv2.CV_64F).var()))

    A_mean = float(np.mean(A_list))
    B_mean = float(np.mean(B_list))
    C_mean = float(np.mean(C_list))
    D_swap_mean = float(np.mean(D_swap_list))
    gain_mean = A_mean - C_mean
    adv_mean = A_mean - B_mean

    pct_A_gt_B = float(np.mean([1.0 if a > b else 0.0 for a, b in zip(A_list, B_list)]) * 100.0)
    pct_A_gt_C = float(np.mean([1.0 if a > c else 0.0 for a, c in zip(A_list, C_list)]) * 100.0)
    pct_gain_gt_0 = float(np.mean([1.0 if (a - c) > 0 else 0.0 for a, c in zip(A_list, C_list)]) * 100.0)

    A_no128_mean = float(np.mean(A_no128_list))
    B_no128_mean = float(np.mean(B_no128_list))

    return {
        "A": A_mean,
        "B": B_mean,
        "C": C_mean,
        "D_swap": D_swap_mean,
        "A_minus_C": gain_mean,
        "A_minus_B": adv_mean,
        "pct_A_gt_B": pct_A_gt_B,
        "pct_A_gt_C": pct_A_gt_C,
        "pct_gain_gt_0": pct_gain_gt_0,
        "sharp_comp": float(np.mean(sharp_comp_list)),
        "sharp_swap": float(np.mean(sharp_swap_list)),
        "mask_cov": float(np.mean(mask_cov_list)),
        "redetect_rate": float(np.mean(redetect_list) * 100.0),
        "mean_lm_err": float(np.mean(lm_err_list)) if lm_err_list else None,
        # Ablation
        "ablation_A_no128": A_no128_mean,
        "ablation_B_no128": B_no128_mean,
        "ablation_adv_no128": A_no128_mean - B_no128_mean,
        "ablation_sharp_no128": float(np.mean(sharp_no128_list)),
        "skip128_adv_delta": adv_mean - (A_no128_mean - B_no128_mean),
        "skip128_sharp_delta": float(np.mean(sharp_comp_list)) - float(np.mean(sharp_no128_list))
    }


def main():
    print("=" * 70, flush=True)
    print("PHASE 6G.1: FULL-RESOLUTION (128x128) SOURCE SKIP EXPERIMENT", flush=True)
    print("=" * 70, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # 1. Verify ArcFace Checkpoint Integrity
    arcface_p = Path("ml/models/weights/ms1mv2_iresnet50.pth")
    assert arcface_p.exists(), f"ArcFace checkpoint missing: {arcface_p}"
    with open(arcface_p, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest().upper()
    assert sha256 == EXPECTED_ARCFACE_SHA, f"ArcFace SHA mismatch! Expected {EXPECTED_ARCFACE_SHA}, got {sha256}"
    print(f"ArcFace SHA-256 confirmed: {sha256}", flush=True)

    # 2. Output and checkpoint directories
    exp_dir = Path("ml/experiments/phase6g_1_fullres_skip")
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 3. Datasets & Preprocessing
    preprocessor = RealFacePreprocessor(image_size=128)
    val_dataset = CelebAPairedDataset(dataset_root="ml/data/celeba", split="val", seed=42)
    train_dataset = CelebAPairedDataset(dataset_root="ml/data/celeba", split="train", seed=42)

    val_pairs = prepare_validation_pairs(val_dataset, preprocessor, num_pairs=100, seed=42)
    train_pool = prepare_training_pool(train_dataset, preprocessor, pool_size=100, seed=123)

    # 4. Instantiate Model & Initialize from verified Phase 6G best_model.pt
    print("\nInstantiating MultiScaleCorrespondenceFaceSwapModel6G1...", flush=True)
    model = MultiScaleCorrespondenceFaceSwapModel6G1(
        base_channels=64,
        bottleneck_channels=512,
        embedding_dim=512,
        fullres_channels=32,
        blur_kernel_size=9,
        blur_sigma=3.0,
        arcface_checkpoint_path=str(arcface_p)
    ).to(device)

    phase6g_ckpt = "ml/checkpoints/stage2/phase6g/best_model.pt"
    assert Path(phase6g_ckpt).exists(), f"Phase 6G baseline checkpoint missing: {phase6g_ckpt}"
    model.load_phase6g_baseline(phase6g_ckpt)

    # Freeze ArcFace
    model.arcface.eval()
    for p in model.arcface.parameters():
        p.requires_grad = False

    discriminator = PatchGANDiscriminator(in_channels=3, base_channels=64).to(device)
    baseline_ckpt = torch.load(phase6g_ckpt, map_location=device)
    if "discriminator_state_dict" in baseline_ckpt and baseline_ckpt["discriminator_state_dict"] is not None:
        discriminator.load_state_dict(baseline_ckpt["discriminator_state_dict"])
        print("Loaded baseline discriminator weights.", flush=True)

    # 5. Losses & Optimizers (Identical to Phase 6G)
    criterion_g = Stage2CompositeLoss(
        arcface_extractor=model.arcface,
        w_id=10.0,
        w_id_swap=8.0,
        w_struct=5.0,
        w_bg=5.0,
        w_mask=5.0,
        w_adv=0.5
    ).to(device)
    criterion_adv = AdversarialLoss().to(device)

    opt_g = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

    scaler_g = torch.amp.GradScaler('cuda')
    scaler_d = torch.amp.GradScaler('cuda')

    # 6. Initial Step 0 Evaluation
    print("\n>> Running Initial Validation (Step 0) on 100 held-out pairs...", flush=True)
    t0_val = time.time()
    val_0 = run_validation(model, val_pairs, preprocessor, device)
    val_0["step"] = 0
    val_history = [val_0]
    print(f"[Step   0/500] A={val_0['A']:.4f} | B={val_0['B']:.4f} | C={val_0['C']:.4f} | Adv(A-B)={val_0['A_minus_B']:+.4f} | P(A>B)={val_0['pct_A_gt_B']:.1f}% | SharpComp={val_0['sharp_comp']:.1f} | SharpSwap={val_0['sharp_swap']:.1f} ({time.time() - t0_val:.1f}s)", flush=True)

    # 7. Training Loop (500 steps)
    max_steps = 500
    val_interval = 50
    batch_size = 4
    rng_train = random.Random(42)

    best_adv = val_0["A_minus_B"]
    best_step = 0
    t_start = time.time()

    print(f"\nStarting 500-step training loop (batch_size={batch_size}, val_interval={val_interval})...", flush=True)

    for step in range(1, max_steps + 1):
        model.train()
        discriminator.train()

        batch_items = rng_train.sample(train_pool, batch_size)
        b_src = torch.cat([it["source"] for it in batch_items], dim=0).to(device)
        b_tgt = torch.cat([it["target"] for it in batch_items], dim=0).to(device)
        b_tgt_m = torch.cat([it["target_mask"] for it in batch_items], dim=0).to(device)
        b_tgt_l = torch.cat([it["target_landmark_map"] for it in batch_items], dim=0).to(device)
        b_aln = torch.cat([it["aligned_source"] for it in batch_items], dim=0).to(device)
        b_conf = torch.cat([it["confidence_map"] for it in batch_items], dim=0).to(device)

        # -----------------------------
        # (a) Generator Step
        # -----------------------------
        opt_g.zero_grad()
        with torch.amp.autocast('cuda'):
            out = model(
                i_target=b_tgt,
                l_target=b_tgt_l,
                i_source=b_src,
                aligned_source=b_aln,
                confidence_map=b_conf,
                disable_skips=False,
                disable_fullres_skip=False
            )
            i_swap = out["i_swap"]
            m_pred = out["m_pred"]
            i_comp = out["i_composite"]

            d_fake_pred = discriminator(i_comp)

            loss_g, loss_dict = criterion_g(
                i_source=b_src,
                i_target=b_tgt,
                i_swap=i_swap,
                i_composite=i_comp,
                pred_mask=m_pred,
                target_mask=b_tgt_m,
                d_fake_logits=d_fake_pred
            )

        scaler_g.scale(loss_g).backward()
        scaler_g.step(opt_g)
        scaler_g.update()

        # -----------------------------
        # (b) Discriminator Step
        # -----------------------------
        opt_d.zero_grad()
        with torch.amp.autocast('cuda'):
            d_real = discriminator(b_tgt)
            d_fake = discriminator(i_comp.detach())
            loss_d, _, _ = criterion_adv.discriminator_loss(d_real, d_fake)

        scaler_d.scale(loss_d).backward()
        scaler_d.step(opt_d)
        scaler_d.update()

        # -----------------------------
        # (c) Validation & Checkpoint
        # -----------------------------
        if step % val_interval == 0 or step == max_steps:
            t_v0 = time.time()
            v_res = run_validation(model, val_pairs, preprocessor, device)
            v_res["step"] = step
            val_history.append(v_res)
            v_time = time.time() - t_v0

            print(
                f"[Step {step:3d}/{max_steps}] "
                f"A={v_res['A']:.4f} | B={v_res['B']:.4f} | "
                f"Adv(A-B)={v_res['A_minus_B']:+.4f} | "
                f"P(A>B)={v_res['pct_A_gt_B']:.1f}% | "
                f"SharpComp={v_res['sharp_comp']:.1f} | SharpSwap={v_res['sharp_swap']:.1f} | "
                f"SkipDelta(Adv)={v_res['skip128_adv_delta']:+.4f} | "
                f"SkipDelta(Sharp)={v_res['skip128_sharp_delta']:+.1f} | "
                f"LossG={loss_g.item():.4f} ({v_time:.1f}s)",
                flush=True
            )

            # Save latest checkpoint
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "metrics": v_res
            }, ckpt_dir / "latest_model.pt")

            # Check if best model (highest A - B advantage, or higher gain with A > B)
            if v_res["A_minus_B"] > best_adv:
                best_adv = v_res["A_minus_B"]
                best_step = step
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "metrics": v_res
                }, ckpt_dir / "best_model.pt")
                print(f"  >>> New Best Model saved at step {step} with Adv(A-B)={best_adv:+.4f} <<<", flush=True)

    elapsed_total = time.time() - t_start
    print(f"\nTraining completed in {elapsed_total:.1f}s ({elapsed_total/max_steps:.2f}s/step).", flush=True)
    print(f"Best Model achieved at step {best_step} with Adv(A-B)={best_adv:+.4f}.", flush=True)

    # Save metrics JSON
    metrics_path = ckpt_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "best_step": best_step,
            "best_advantage": best_adv,
            "total_steps": max_steps,
            "training_time_seconds": elapsed_total,
            "history": val_history
        }, f, indent=2, cls=NumpyEncoder)
    print(f"Saved training history to {metrics_path}.", flush=True)


if __name__ == "__main__":
    main()
