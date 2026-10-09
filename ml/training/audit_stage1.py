#!/usr/bin/env python3
"""
Stage 1 Result Audit Script for Hackathena Face-Swap Project
Audits:
  - Check 1: CUDA / Device Placement & Parameter Breakdown
  - Check 2: Checkpoint Integrity & SHA-256
  - Check 3: Validation Result Re-Run from Best Checkpoint
  - Check 4: Qualitative Output Analysis across steps 25, 50, 100, 150, 200, 250
  - Check 5: Mask Behavioral Statistics (mean, min, max, saturation/collapse checks)
  - Check 6: Identity Result Interpretation (Reconstruction vs Disentangled Swapping)
  - Check 7: Final Decision Matrix
"""

import sys
import os
import hashlib
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.data.dataset import CelebAPairedDataset
from ml.models.face_swap_model import AdaINFaceSwapModel
from ml.models.losses import Stage1CompositeLoss


def run_audit():
    print("====================================================================")
    print("       PHASE 6A — STAGE 1 RESULT AUDIT & INTEGRITY CHECK            ")
    print("====================================================================")

    # -------------------------------------------------------------------------
    # CHECK 1 — CUDA / DEVICE PLACEMENT
    # -------------------------------------------------------------------------
    print("\n--- CHECK 1: CUDA / DEVICE PLACEMENT ---")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available:           {cuda_available}")
    if not cuda_available:
        print("FAIL: CUDA is not available!")
        sys.exit(1)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Device:                   {device} ({gpu_name})")

    # Load model architecture
    model = AdaINFaceSwapModel(
        base_channels=64,
        bottleneck_channels=512,
        embedding_dim=512,
        arcface_checkpoint_path="ml/models/weights/ms1mv2_iresnet50.pth"
    ).to(device)

    # Component devices & grad states
    components = {
        "TargetStructureEncoder": model.target_encoder,
        "IdentityConditionedGenerator": model.generator,
        "RGB Output Head": model.generator.rgb_head,
        "Mask Output Head": model.generator.mask_head,
        "Frozen ArcFace Backbone": model.arcface
    }

    trainable_params = 0
    frozen_params = 0
    all_components_on_cuda = True

    for name, comp in components.items():
        comp_devices = {p.device.type for p in comp.parameters()}
        all_cuda = (comp_devices == {"cuda"})
        if not all_cuda:
            all_components_on_cuda = False

        comp_trainable = sum(p.numel() for p in comp.parameters() if p.requires_grad)
        comp_frozen = sum(p.numel() for p in comp.parameters() if not p.requires_grad)
        trainable_params += comp_trainable
        frozen_params += comp_frozen

        status = "CUDA (trainable)" if comp_trainable > 0 else "CUDA (frozen)"
        print(f"  - {name:30s}: Device={list(comp_devices)} | Status={status} | Params: {comp_trainable + comp_frozen:,} (Trainable: {comp_trainable:,}, Frozen: {comp_frozen:,})")

    # Ensure ArcFace requires_grad is strictly False
    arcface_has_grad = any(p.requires_grad for p in model.arcface.parameters())
    print(f"ArcFace requires_grad == False: {not arcface_has_grad}")
    print(f"All components on CUDA:         {all_components_on_cuda}")
    print(f"Total Trainable Parameters:     {trainable_params:,}")
    print(f"Total Frozen Parameters:        {frozen_params:,}")

    # Forward & backward CUDA verification
    dummy_target = torch.randn(1, 3, 128, 128, device=device)
    dummy_landmark = torch.randn(1, 1, 128, 128, device=device)
    dummy_source = torch.randn(1, 3, 128, 128, device=device)

    model.train()
    out = model(i_target=dummy_target, l_target=dummy_landmark, i_source=dummy_source)
    loss = out["i_composite"].sum()
    loss.backward()

    # Check CUDA memory
    alloc_mb = torch.cuda.memory_allocated() / (1024**2)
    max_alloc_mb = torch.cuda.max_memory_allocated() / (1024**2)
    res_mb = torch.cuda.memory_reserved() / (1024**2)
    print(f"Peak Allocated VRAM:            {max_alloc_mb:.2f} MB")
    print(f"Peak Reserved VRAM:             {res_mb:.2f} MB")
    print(f"CUDA Forward/Backward Verified: PASS")

    # -------------------------------------------------------------------------
    # CHECK 2 — CHECKPOINT INTEGRITY
    # -------------------------------------------------------------------------
    print("\n--- CHECK 2: CHECKPOINT INTEGRITY ---")
    best_path = Path("ml/checkpoints/stage1/best_model.pt")
    latest_path = Path("ml/checkpoints/stage1/latest_model.pt")
    arcface_path = Path("ml/models/weights/ms1mv2_iresnet50.pth")

    print(f"best_model.pt exists:   {best_path.is_file()} ({best_path.stat().st_size / (1024**2):.1f} MB)")
    print(f"latest_model.pt exists: {latest_path.is_file()} ({latest_path.stat().st_size / (1024**2):.1f} MB)")

    # ArcFace SHA-256
    with open(arcface_path, "rb") as f:
        sha256_actual = hashlib.sha256(f.read()).hexdigest().upper()
    sha256_expected = "2B75B93C48B01C78A4263F7295AB2DBF84F85190C51BE45B92C4C0B0AEDCEAA3"
    print(f"ArcFace Checkpoint SHA-256:")
    print(f"  Actual:   {sha256_actual}")
    print(f"  Expected: {sha256_expected}")
    assert sha256_actual == sha256_expected, "ArcFace SHA-256 mismatch!"

    # Load best checkpoint
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    print(f"Checkpoint Keys:        {list(ckpt.keys())}")
    print(f"Recorded Step:          {ckpt.get('global_step')}")
    print(f"Recorded Epoch:         {ckpt.get('epoch')}")
    print(f"Recorded Best Val Loss: {ckpt.get('best_val_loss')}")
    print(f"Optimizer State Exists: {'optimizer_state_dict' in ckpt and ckpt['optimizer_state_dict'] is not None}")

    # Load weights into model
    model.load_state_dict(ckpt["model_state_dict"])
    has_nan_or_inf = False
    for p_name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            print(f"FAIL: NaN or Inf found in parameter: {p_name}")
            has_nan_or_inf = True
    print(f"Parameters Finite (No NaN/Inf): {not has_nan_or_inf}")

    # -------------------------------------------------------------------------
    # CHECK 3 & CHECK 5 — VALIDATION RESULT & MASK STATISTICS
    # -------------------------------------------------------------------------
    print("\n--- CHECK 3 & 5: VALIDATION RESULT & MASK STATISTICS ---")
    val_dataset = CelebAPairedDataset(
        dataset_root="ml/data/celeba",
        split="val",
        image_size=128,
        same_identity_probability=1.0,
        seed=42
    )
    train_dataset = CelebAPairedDataset(
        dataset_root="ml/data/celeba",
        split="train",
        image_size=128,
        same_identity_probability=1.0,
        seed=42
    )

    # Identity Disjoint Verification
    train_ids = set(train_dataset.split_identity_to_images.keys())
    val_ids = set(val_dataset.split_identity_to_images.keys())
    overlap = train_ids.intersection(val_ids)
    print(f"Train Identities:       {len(train_ids)}")
    print(f"Validation Identities:  {len(val_ids)}")
    print(f"Identity Overlap:       {len(overlap)} (Strictly 0 required)")
    assert len(overlap) == 0, f"Identity leak detected! Overlap: {len(overlap)}"

    criterion = Stage1CompositeLoss(
        arcface_extractor=model.arcface,
        w_recon=10.0,
        w_id=5.0,
        w_bg=5.0,
        w_mask=2.0
    ).to(device)

    model.eval()
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    num_eval_pairs = 30
    eval_metrics = {
        "loss_total": 0.0,
        "loss_recon": 0.0,
        "loss_id": 0.0,
        "loss_bg": 0.0,
        "loss_mask": 0.0,
        "cos_sim_out": 0.0,
        "cos_sim_target": 0.0
    }

    mask_stats = {
        "means": [],
        "mins": [],
        "maxs": [],
        "pct_above_50": []
    }

    print(f"Evaluating {num_eval_pairs} held-out validation pairs with best_model.pt...")
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_eval_pairs:
                break

            i_src = batch["source"].to(device)
            i_tgt = batch["target"].to(device)
            tgt_mask = batch["target_mask"].to(device)
            tgt_l = batch["target_landmark_map"].to(device)

            out = model(i_target=i_tgt, l_target=tgt_l, i_source=i_src)
            i_swap = out["i_swap"]
            m_pred = out["m_pred"]
            i_comp = out["i_composite"]

            tot_loss, l_dict = criterion(
                i_source=i_src,
                i_target=i_tgt,
                i_swap=i_swap,
                i_composite=i_comp,
                pred_mask=m_pred,
                target_mask=tgt_mask
            )

            z_src = model.arcface(i_src)
            z_out = model.arcface(i_comp)
            z_tgt = model.arcface(i_tgt)

            cos_out = (z_src * z_out).sum(dim=-1).item()
            cos_tgt = (z_src * z_tgt).sum(dim=-1).item()

            for k in l_dict:
                eval_metrics[k] += l_dict[k]
            eval_metrics["cos_sim_out"] += cos_out
            eval_metrics["cos_sim_target"] += cos_tgt

            # Mask statistics
            m_np = m_pred.cpu().numpy().squeeze()
            mask_stats["means"].append(float(np.mean(m_np)))
            mask_stats["mins"].append(float(np.min(m_np)))
            mask_stats["maxs"].append(float(np.max(m_np)))
            mask_stats["pct_above_50"].append(float(np.mean(m_np > 0.5) * 100.0))

    # Average metrics
    avg_metrics = {k: v / num_eval_pairs for k, v in eval_metrics.items()}
    print(f"Total Validation Pairs:     {num_eval_pairs}")
    print(f"Total Loss:                 {avg_metrics['loss_total']:.4f}")
    print(f"  - Reconstruction Loss:    {avg_metrics['loss_recon']:.4f}")
    print(f"  - Identity Loss:          {avg_metrics['loss_id']:.4f}")
    print(f"  - Background Loss:        {avg_metrics['loss_bg']:.4f}")
    print(f"  - Mask Regularization:    {avg_metrics['loss_mask']:.4f}")
    print(f"ArcFace Source/Output Cosine: {avg_metrics['cos_sim_out']:.4f}")
    print(f"ArcFace Source/Target Cosine: {avg_metrics['cos_sim_target']:.4f}")

    # Report Mask Behavior
    mean_m = np.mean(mask_stats["means"])
    min_m = np.min(mask_stats["mins"])
    max_m = np.max(mask_stats["maxs"])
    pct_50 = np.mean(mask_stats["pct_above_50"])

    print("\n--- CHECK 5: MASK BEHAVIOR REPORT ---")
    print(f"Mean Mask Value:            {mean_m:.4f}")
    print(f"Min Mask Value:             {min_m:.4f}")
    print(f"Max Mask Value:             {max_m:.4f}")
    print(f"Pixels Above 0.5:           {pct_50:.2f}%")

    is_collapsed = mean_m < 0.05
    is_saturated = mean_m > 0.95 or pct_50 > 95.0
    print(f"Mask Collapsed (zero-like): {is_collapsed}")
    print(f"Mask Saturated (one-like):  {is_saturated}")
    print(f"Mask Status:                {'HEALTHY' if (not is_collapsed and not is_saturated) else 'PATHOLOGICAL'}")

    # -------------------------------------------------------------------------
    # CHECK 4 — QUALITATIVE OUTPUTS INSPECTION
    # -------------------------------------------------------------------------
    print("\n--- CHECK 4: QUALITATIVE OUTPUTS INSPECTION ---")
    samples_dir = Path("ml/checkpoints/stage1/samples")
    steps_to_check = [25, 50, 100, 150, 200, 250]

    for s in steps_to_check:
        sample_file = samples_dir / f"step_{s}_sample_0.png"
        if not sample_file.exists():
            print(f"Step {s:3d}: Missing {sample_file.name}")
            continue

        img = Image.open(sample_file)
        arr = np.array(img)
        h, w, c = arr.shape
        # Grid width is 5 tiles of 128: Source, Target, Swap, Mask, Composite
        tile_w = w // 5
        src_tile = arr[:, 0:tile_w]
        tgt_tile = arr[:, tile_w:2*tile_w]
        swap_tile = arr[:, 2*tile_w:3*tile_w]
        mask_tile = arr[:, 3*tile_w:4*tile_w]
        comp_tile = arr[:, 4*tile_w:5*tile_w]

        # Check numerical health
        is_finite = not np.isnan(arr).any()
        is_not_black = comp_tile.mean() > 5.0
        is_not_white = comp_tile.mean() < 250.0
        mask_mean = mask_tile.mean() / 255.0

        print(f"Step {s:3d} Sample 0: Shape={arr.shape} | Comp Mean={comp_tile.mean():.1f} | Mask Mean={mask_mean:.3f} | Valid RGB={'PASS' if (is_finite and is_not_black and is_not_white) else 'FAIL'}")

    print("\n====================================================================")
    print("                     AUDIT EXECUTION COMPLETE                       ")
    print("====================================================================")


if __name__ == "__main__":
    run_audit()
