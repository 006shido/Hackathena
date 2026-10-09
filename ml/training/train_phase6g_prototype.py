"""
Phase 6G: Correspondence-Aware Multi-Scale Face-Swap Training & Pilot Suite
Combines:
  - 6D local piecewise-affine correspondence
  - Multi-scale source encoder with lateral skips at 16x16, 32x32, 64x64
  - Target structure encoder
  - ArcFace 512D identity vector conditioning via AdaIN
  - PatchGAN adversarial realism
  - Anti-collapse mask supervision
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

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.models.phase6g_model import MultiScaleCorrespondenceFaceSwapModel
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
    num_pairs: int = 10,
    seed: int = 42
) -> List[Dict[str, Any]]:
    print(f"Pre-caching {num_pairs} fixed cross-ID validation pairs (seed={seed})...", flush=True)
    identities = list(val_dataset.split_identity_to_images.keys())
    cached_val = []
    rng = random.Random(seed)
    attempts = 0

    while len(cached_val) < num_pairs and attempts < num_pairs * 10:
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

            cached_val.append({
                "source_tensor": torch.from_numpy(s_tensor).unsqueeze(0),
                "target_tensor": torch.from_numpy(t_tensor).unsqueeze(0),
                "target_mask": m_tensor.unsqueeze(0),
                "target_l_map": torch.from_numpy(t_lmap).unsqueeze(0),
                "aligned_source": aln_src_tensor,
                "confidence_map": conf_tensor,
                "target_5pts": t_5pts,
                "target_dense": t_dense,
                "src_img_pil": s_img,
                "tgt_img_pil": t_img,
                "id_s": id_s,
                "id_t": id_t
            })
            if len(cached_val) % 5 == 0:
                print(f"  Cached {len(cached_val)}/{num_pairs} validation pairs...", flush=True)
        except Exception:
            continue

    print(f"Successfully cached {len(cached_val)} validation pairs.", flush=True)
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

    while len(pool) < pool_size and attempts < pool_size * 10:
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
            if len(pool) % 25 == 0:
                print(f"  Cached {len(pool)}/{pool_size} pairs ({time.time() - t0:.1f}s)...", flush=True)
        except Exception:
            continue

    dt = time.time() - t0
    print(f"Cached {len(pool)} training pairs in {dt:.2f}s ({dt/len(pool):.3f}s/pair).", flush=True)
    return pool


def run_validation(
    model: MultiScaleCorrespondenceFaceSwapModel,
    discriminator: nn.Module,
    val_pairs: List[Dict[str, Any]],
    criterion_g: Stage2CompositeLoss,
    preprocessor: RealFacePreprocessor,
    device: torch.device,
    save_samples_dir: Optional[Path] = None,
    step_num: int = 0
) -> Dict[str, Any]:
    model.eval()
    discriminator.eval()

    A_list, B_list, C_list, D_list = [], [], [], []
    mask_in_list, mask_out_list, mask_act_list, mask_mean_list = [], [], [], []
    bg_err_list, struct_err_list = [], []
    redetect_success_count = 0
    landmark_errors = []

    # Ablation metrics (skips disabled)
    A_noskip_list, D_noskip_list = [], []

    sample_rows = []

    with torch.no_grad():
        for i, p in enumerate(val_pairs):
            src = p["source_tensor"].to(device)
            tgt = p["target_tensor"].to(device)
            tgt_m = p["target_mask"].to(device)
            tgt_l = p["target_l_map"].to(device)
            aln_src = p["aligned_source"].to(device)
            conf_m = p["confidence_map"].to(device)
            gt_5pts = p["target_5pts"]

            z_src = model.arcface(src)
            z_tgt = model.arcface(tgt)

            # 1. Full Mode: Aligned source multi-scale skips ENABLED
            out = model(
                i_target=tgt,
                l_target=tgt_l,
                i_source=src,
                aligned_source=aln_src,
                confidence_map=conf_m,
                disable_skips=False
            )
            i_swap = out["i_swap"]
            m_pred = out["m_pred"]
            i_comp = out["i_composite"]

            z_comp = model.arcface(i_comp)
            z_swap = model.arcface(i_swap)

            cos_A = float(F.cosine_similarity(z_src, z_comp).item())
            cos_B = float(F.cosine_similarity(z_tgt, z_comp).item())
            cos_C = float(F.cosine_similarity(z_src, z_tgt).item())
            cos_D = float(F.cosine_similarity(z_src, z_swap).item())

            A_list.append(cos_A)
            B_list.append(cos_B)
            C_list.append(cos_C)
            D_list.append(cos_D)

            # Mask statistics
            m_np = m_pred.squeeze().cpu().numpy()
            tm_np = tgt_m.squeeze().cpu().numpy()
            in_face = tm_np > 0.5
            out_face = tm_np < 0.1
            mask_in_list.append(float(m_np[in_face].mean()) if in_face.sum() > 0 else 0.5)
            mask_out_list.append(float(m_np[out_face].mean()) if out_face.sum() > 0 else 0.0)
            mask_act_list.append(float((m_np > 0.5).mean() * 100.0))
            mask_mean_list.append(float(m_np.mean()))

            # Background & Structural loss
            bg_err = float(torch.abs((1.0 - tgt_m) * (i_comp - tgt)).mean().item())
            bg_err_list.append(bg_err)
            struct_err = float(torch.abs(tgt_m * (i_swap - tgt)).mean().item())
            struct_err_list.append(struct_err)

            # MediaPipe redetection on composite output
            comp_pil = denormalize_image_tensor(i_comp.squeeze(0).cpu().numpy())
            try:
                det_comp = preprocessor.detect_landmarks(comp_pil)
                if det_comp is not None and det_comp.dense_landmarks is not None:
                    redetect_success_count += 1
                    err_5pts = np.linalg.norm(det_comp.key_landmarks_5pts - gt_5pts, axis=1).mean()
                    landmark_errors.append(float(err_5pts))
            except Exception:
                pass

            # 2. Ablation: Multi-scale skips DISABLED
            out_noskip = model(
                i_target=tgt,
                l_target=tgt_l,
                i_source=src,
                aligned_source=aln_src,
                confidence_map=conf_m,
                disable_skips=True
            )
            z_comp_noskip = model.arcface(out_noskip["i_composite"])
            z_swap_noskip = model.arcface(out_noskip["i_swap"])
            A_noskip_list.append(float(F.cosine_similarity(z_src, z_comp_noskip).item()))
            D_noskip_list.append(float(F.cosine_similarity(z_src, z_swap_noskip).item()))

            # Collect visualization rows for first 5 samples
            if i < 5:
                def to_u8(t):
                    return ((t.squeeze(0).cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                src_u8 = to_u8(src)
                tgt_u8 = to_u8(tgt)
                aln_u8 = to_u8(aln_src)
                swap_u8 = to_u8(i_swap)
                mask_u8 = np.repeat((m_np * 255.0).clip(0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
                comp_u8 = to_u8(i_comp)
                row = np.concatenate([src_u8, tgt_u8, aln_u8, swap_u8, mask_u8, comp_u8], axis=1)
                sample_rows.append(row)

    # Save visual grid if directory provided
    if save_samples_dir is not None and sample_rows:
        grid_body = np.concatenate(sample_rows, axis=0)
        h, w_col = 32, 128
        header = np.full((h, w_col * 6, 3), 30, dtype=np.uint8)
        lbls = ["SOURCE", "TARGET", "6D ALIGNED", "SWAP PRED", "MASK PRED", "COMPOSITE"]
        for idx, lbl in enumerate(lbls):
            cv2.putText(header, lbl, (idx * w_col + 14, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        full_grid = np.concatenate([header, grid_body], axis=0)
        grid_path = save_samples_dir / f"step_{step_num:03d}_val_grid.png"
        Image.fromarray(full_grid).save(grid_path)

    A_mean = float(np.mean(A_list))
    B_mean = float(np.mean(B_list))
    C_mean = float(np.mean(C_list))
    D_mean = float(np.mean(D_list))
    A_minus_C = float(A_mean - C_mean)
    D_minus_C = float(D_mean - C_mean)

    A_noskip_mean = float(np.mean(A_noskip_list))
    skip_gain = float(A_mean - A_noskip_mean)

    lm_err_mean = float(np.mean(landmark_errors)) if landmark_errors else float("nan")

    return {
        "step": step_num,
        "A": A_mean,
        "B": B_mean,
        "C": C_mean,
        "D": D_mean,
        "A_minus_C": A_minus_C,
        "D_minus_C": D_minus_C,
        "A_noskip": A_noskip_mean,
        "skip_gain": skip_gain,
        "mask_in": float(np.mean(mask_in_list)),
        "mask_out": float(np.mean(mask_out_list)),
        "mask_act_pct": float(np.mean(mask_act_list)),
        "mask_mean": float(np.mean(mask_mean_list)),
        "bg_error": float(np.mean(bg_err_list)),
        "struct_error": float(np.mean(struct_err_list)),
        "redetect_rate": float(redetect_success_count / len(val_pairs)),
        "landmark_error": lm_err_mean
    }


def main():
    print("=" * 60, flush=True)
    print("PHASE 6G: CORRESPONDENCE-AWARE MULTI-SCALE FACE-SWAP PILOT", flush=True)
    print("=" * 60, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})", flush=True)
    torch.cuda.reset_peak_memory_stats()

    # 1. Verify ArcFace SHA-256
    arcface_path = Path("ml/models/weights/ms1mv2_iresnet50.pth")
    assert arcface_path.exists(), f"ArcFace weights missing: {arcface_path}"
    with open(arcface_path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest().upper()
    print(f"ArcFace SHA-256: {sha256}", flush=True)
    assert sha256 == EXPECTED_ARCFACE_SHA, f"ArcFace SHA mismatch! Expected {EXPECTED_ARCFACE_SHA}, got {sha256}"

    # 2. Setup directories
    out_dir = Path("ml/checkpoints/stage2/phase6g")
    samples_dir = out_dir / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    # 3. Datasets & Preprocessing
    preprocessor = RealFacePreprocessor(image_size=128)
    val_dataset = CelebAPairedDataset(dataset_root="ml/data/celeba", split="val", seed=42)
    train_dataset = CelebAPairedDataset(dataset_root="ml/data/celeba", split="train", seed=42)

    val_pairs = prepare_validation_pairs(val_dataset, preprocessor, num_pairs=10, seed=42)
    train_pool = prepare_training_pool(train_dataset, preprocessor, pool_size=100, seed=123)

    # 4. Instantiate Models
    print("Initializing MultiScaleCorrespondenceFaceSwapModel...", flush=True)
    model = MultiScaleCorrespondenceFaceSwapModel(
        base_channels=64,
        bottleneck_channels=512,
        embedding_dim=512,
        blur_kernel_size=9,
        blur_sigma=3.0,
        arcface_checkpoint_path=str(arcface_path)
    ).to(device)

    baseline_ckpt = "ml/checkpoints/stage2/best_model.pt"
    assert Path(baseline_ckpt).exists(), f"Baseline checkpoint missing: {baseline_ckpt}"
    model.load_baseline_checkpoint(baseline_ckpt)

    # Freeze ArcFace
    model.arcface.eval()
    for p in model.arcface.parameters():
        p.requires_grad = False

    discriminator = PatchGANDiscriminator(in_channels=3, base_channels=64).to(device)
    init_ckpt = torch.load(baseline_ckpt, map_location=device, weights_only=False)
    if "discriminator_state_dict" in init_ckpt and init_ckpt["discriminator_state_dict"] is not None:
        discriminator.load_state_dict(init_ckpt["discriminator_state_dict"])
        print("Loaded baseline discriminator weights.", flush=True)

    # 5. Losses & Optimizers
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

    # 6. Step 0 Initial Validation
    print("\n>> Running Initial Validation (Step 0)...", flush=True)
    step_history = []
    val_0 = run_validation(
        model=model,
        discriminator=discriminator,
        val_pairs=val_pairs,
        criterion_g=criterion_g,
        preprocessor=preprocessor,
        device=device,
        save_samples_dir=samples_dir,
        step_num=0
    )
    step_history.append(val_0)
    print(f"[Step   0/300] A={val_0['A']:.4f} | B={val_0['B']:.4f} | C={val_0['C']:.4f} | D={val_0['D']:.4f} | A-C={val_0['A_minus_C']:+.4f} | D-C={val_0['D_minus_C']:+.4f} | MaskIn={val_0['mask_in']:.3f} | SkipGain={val_0['skip_gain']:+.4f}", flush=True)

    # 7. Training Loop (Exactly 300 steps)
    max_steps = 300
    val_interval = 50
    batch_size = 4
    rng_train = random.Random(42)

    best_gain = val_0["A_minus_C"]
    best_step = 0
    t_start = time.time()

    print(f"\nStarting 300-step training loop (batch_size={batch_size}, val_interval={val_interval})...", flush=True)

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
                confidence_map=b_conf
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

        # Validation check
        if step % val_interval == 0:
            val_m = run_validation(
                model=model,
                discriminator=discriminator,
                val_pairs=val_pairs,
                criterion_g=criterion_g,
                preprocessor=preprocessor,
                device=device,
                save_samples_dir=samples_dir,
                step_num=step
            )
            val_m["loss_g"] = float(loss_g.item())
            val_m["loss_d"] = float(loss_d.item())
            step_history.append(val_m)

            print(
                f"[Step {step:3d}/300] LossG: {loss_g.item():.3f} | LossD: {loss_d.item():.3f} | "
                f"A: {val_m['A']:.4f} | C: {val_m['C']:.4f} | A-C: {val_m['A_minus_C']:+.4f} | "
                f"D-C: {val_m['D_minus_C']:+.4f} | SkipGain: {val_m['skip_gain']:+.4f} | "
                f"MaskIn: {val_m['mask_in']:.3f} | LmErr: {val_m['landmark_error']:.2f}px",
                flush=True
            )

            # Track best model by A-C gain provided mask is healthy
            if val_m["A_minus_C"] > best_gain and val_m["mask_in"] > 0.35:
                best_gain = val_m["A_minus_C"]
                best_step = step
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "metrics": val_m
                }, out_dir / "best_model.pt")

    t_elapsed = time.time() - t_start
    steps_per_sec = max_steps / t_elapsed
    peak_vram_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_vram_res = torch.cuda.max_memory_reserved() / (1024 ** 2)

    print(f"\n300 steps finished in {t_elapsed:.1f}s ({steps_per_sec:.2f} steps/s). Peak VRAM: {peak_vram_alloc:.1f}MB", flush=True)

    # Save latest model
    torch.save({
        "step": max_steps,
        "model_state_dict": model.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "metrics": step_history[-1]
    }, out_dir / "latest_model.pt")

    # If best_model.pt was never saved (e.g. step 0 was best), save latest as best
    if not (out_dir / "best_model.pt").exists():
        torch.save({
            "step": max_steps,
            "model_state_dict": model.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "metrics": step_history[-1]
        }, out_dir / "best_model.pt")

    # Save metrics JSON
    results = {
        "best_step": best_step,
        "best_A_minus_C": best_gain,
        "final_step": max_steps,
        "final_A_minus_C": step_history[-1]["A_minus_C"],
        "runtime_sec": t_elapsed,
        "steps_per_sec": steps_per_sec,
        "peak_vram_alloc_mb": peak_vram_alloc,
        "peak_vram_res_mb": peak_vram_res,
        "step_history": step_history
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"Metrics saved to {out_dir / 'metrics.json'}", flush=True)

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

    print("Phase 6G 300-step pilot successfully completed.", flush=True)


if __name__ == "__main__":
    main()
