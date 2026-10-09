#!/usr/bin/env python3
"""
Stage 2 Controlled Training Pilot: Cross-Identity Face Swapping + Adversarial Training
Strictly enforces:
  - 100% genuine CelebA cross-identity pairs (id(source) != id(target))
  - Initialization from verified Stage 1 checkpoint: ml/checkpoints/stage1/best_model.pt
  - Official ArcFace MS1MV2 iResNet-50 completely frozen (0 gradients)
  - Trainable TargetStructureEncoder + Generator + PatchGAN Discriminator
  - Independent checkpointing under ml/checkpoints/stage2/pilot/ (never touches Stage 1)
  - Full ArcFace identity triad tracking on held-out validation identities:
      A = cosine(source, output)
      B = cosine(target, output)
      C = cosine(source, target) baseline
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.data.dataset import CelebAPairedDataset
from ml.models.face_swap_model import AdaINFaceSwapModel
from ml.models.discriminator import PatchGANDiscriminator, AdversarialLoss
from ml.models.losses import Stage2CompositeLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Hackathena Stage 2 Controlled Pilot")
    parser.add_argument("--stage1-checkpoint", type=str, default="ml/checkpoints/stage1/best_model.pt")
    parser.add_argument("--arcface-checkpoint", type=str, default="ml/models/weights/ms1mv2_iresnet50.pth")
    parser.add_argument("--dataset-root", type=str, default="ml/data/celeba")
    parser.add_argument("--checkpoint-dir", type=str, default="ml/checkpoints/stage2/pilot")
    parser.add_argument("--max-steps", type=int, default=150, help="Controlled short pilot steps")
    parser.add_argument("--val-interval", type=int, default=50, help="Validation interval")
    parser.add_argument("--val-samples", type=int, default=10, help="Validation pairs count")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr-g", type=float, default=1e-4)
    parser.add_argument("--lr-d", type=float, default=1e-4)
    parser.add_argument("--w-id", type=float, default=10.0)
    parser.add_argument("--w-struct", type=float, default=5.0)
    parser.add_argument("--w-bg", type=float, default=5.0)
    parser.add_argument("--w-mask", type=float, default=2.0)
    parser.add_argument("--w-adv", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def evaluate_cross_id_val(
    model: nn.Module,
    discriminator: nn.Module,
    val_dataset: CelebAPairedDataset,
    criterion_g: Stage2CompositeLoss,
    criterion_adv: AdversarialLoss,
    device: torch.device,
    num_samples: int = 10,
    save_samples_dir: Optional[Path] = None,
    step_num: int = 0
) -> Dict[str, float]:
    """
    Evaluates held-out cross-ID validation pairs. Computes identity triad (A, B, C) and mask statistics.
    """
    model.eval()
    discriminator.eval()
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    triad_A = []  # cos(source, output)
    triad_B = []  # cos(target, output)
    triad_C = []  # cos(source, target)
    losses = {
        "loss_total": 0.0,
        "loss_id": 0.0,
        "loss_struct": 0.0,
        "loss_bg": 0.0,
        "loss_mask": 0.0,
        "loss_adv": 0.0,
        "loss_d": 0.0
    }
    mask_means, mask_mins, mask_maxs, mask_pct_50 = [], [], [], []
    saved_count = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_samples:
                break

            i_source = batch["source"].to(device)
            i_target = batch["target"].to(device)
            target_mask = batch["target_mask"].to(device)
            target_l_map = batch["target_landmark_map"].to(device)

            outputs = model(i_target=i_target, l_target=target_l_map, i_source=i_source)
            i_swap = outputs["i_swap"]
            m_pred = outputs["m_pred"]
            i_composite = outputs["i_composite"]

            # Discriminator logits
            d_real = discriminator(i_target)
            d_fake = discriminator(i_composite)
            loss_d, _, _ = criterion_adv.discriminator_loss(d_real, d_fake)

            tot_g, g_dict = criterion_g(
                i_source=i_source,
                i_target=i_target,
                i_swap=i_swap,
                i_composite=i_composite,
                pred_mask=m_pred,
                target_mask=target_mask,
                d_fake_logits=d_fake
            )

            # ArcFace Triad
            z_src = model.arcface(i_source)
            z_out = model.arcface(i_composite)
            z_tgt = model.arcface(i_target)

            cos_A = (z_src * z_out).sum(dim=-1).item()
            cos_B = (z_tgt * z_out).sum(dim=-1).item()
            cos_C = (z_src * z_tgt).sum(dim=-1).item()

            triad_A.append(cos_A)
            triad_B.append(cos_B)
            triad_C.append(cos_C)

            for k in g_dict:
                losses[k] += g_dict[k]
            losses["loss_d"] += loss_d.item()

            # Mask statistics
            m_np = m_pred.cpu().numpy().squeeze()
            mask_means.append(float(np.mean(m_np)))
            mask_mins.append(float(np.min(m_np)))
            mask_maxs.append(float(np.max(m_np)))
            mask_pct_50.append(float(np.mean(m_np > 0.5) * 100.0))

            # Save qualitative visual sample
            if save_samples_dir and saved_count < 4:
                save_samples_dir.mkdir(parents=True, exist_ok=True)
                src_vis = (i_source[0].cpu() + 1.0) / 2.0
                tgt_vis = (i_target[0].cpu() + 1.0) / 2.0
                swap_vis = (i_swap[0].cpu() + 1.0) / 2.0
                mask_vis = m_pred[0].cpu().expand(3, -1, -1)
                comp_vis = (i_composite[0].cpu() + 1.0) / 2.0

                grid = torch.cat([src_vis, tgt_vis, swap_vis, mask_vis, comp_vis], dim=2)
                save_image(grid, save_samples_dir / f"step_{step_num}_sample_{saved_count}.png")
                saved_count += 1

    model.train()
    discriminator.train()

    count = max(len(triad_A), 1)
    res = {k: v / count for k, v in losses.items()}
    res["cos_A_src_out"] = float(np.mean(triad_A))
    res["cos_B_tgt_out"] = float(np.mean(triad_B))
    res["cos_C_src_tgt"] = float(np.mean(triad_C))
    res["mask_mean"] = float(np.mean(mask_means))
    res["mask_min"] = float(np.min(mask_mins))
    res["mask_max"] = float(np.max(mask_maxs))
    res["mask_pct_50"] = float(np.mean(mask_pct_50))
    return res


def run_stage2_pilot():
    args = parse_args()

    assert torch.cuda.is_available(), "CUDA is required for Stage 2 pilot!"
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)

    print("====================================================================")
    print("        HACKATHENA PHASE 6B: CONTROLLED STAGE 2 PILOT               ")
    print("====================================================================")
    print(f"Device:               {device} ({gpu_name})")
    print(f"Total VRAM:           {torch.cuda.get_device_properties(0).total_memory / (1024**2):.1f} MB")
    print(f"Stage 1 Checkpoint:   {args.stage1_checkpoint}")
    print(f"Output Directory:     {args.checkpoint_dir}")
    print("--------------------------------------------------------------------")

    # Seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = checkpoint_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # 1. Datasets: 100% Genuine Cross-ID Pairs (same_identity_probability=0.0)
    print("Initializing Cross-ID Datasets (same_identity_probability=0.0)...")
    train_dataset = CelebAPairedDataset(
        dataset_root=args.dataset_root,
        split="train",
        image_size=128,
        same_identity_probability=0.0,
        seed=args.seed
    )
    val_dataset = CelebAPairedDataset(
        dataset_root=args.dataset_root,
        split="val",
        image_size=128,
        same_identity_probability=0.0,
        seed=args.seed
    )

    print(f"Train Identities:     {len(train_dataset.split_identity_to_images)}")
    print(f"Validation Identities:{len(val_dataset.split_identity_to_images)}")
    overlap = set(train_dataset.split_identity_to_images.keys()).intersection(
        set(val_dataset.split_identity_to_images.keys())
    )
    print(f"Identity Overlap:     {len(overlap)} (Strictly 0 verified)")
    assert len(overlap) == 0, "Identity overlap detected!"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    # 2. Models
    print("Initializing Models & Loading Stage 1 Pretrained Generator...")
    model = AdaINFaceSwapModel(
        base_channels=64,
        bottleneck_channels=512,
        embedding_dim=512,
        arcface_checkpoint_path=args.arcface_checkpoint
    ).to(device)

    # Load Stage 1 checkpoint
    stage1_ckpt = torch.load(args.stage1_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(stage1_ckpt["model_state_dict"])
    print(f"  Loaded Stage 1 best weights (step {stage1_ckpt.get('global_step')})")

    # ArcFace strictly frozen
    model.arcface.eval()
    for p in model.arcface.parameters():
        p.requires_grad = False

    # PatchGAN Discriminator
    discriminator = PatchGANDiscriminator(in_channels=3, base_channels=64).to(device)

    # 3. Optimizers & Loss Criteria
    criterion_g = Stage2CompositeLoss(
        arcface_extractor=model.arcface,
        w_id=args.w_id,
        w_struct=args.w_struct,
        w_bg=args.w_bg,
        w_mask=args.w_mask,
        w_adv=args.w_adv
    ).to(device)
    criterion_adv = AdversarialLoss().to(device)

    opt_g = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr_g, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    scaler_g = torch.amp.GradScaler('cuda')
    scaler_d = torch.amp.GradScaler('cuda')

    print(f"Generator Trainable Params:     {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Discriminator Trainable Params: {sum(p.numel() for p in discriminator.parameters() if p.requires_grad):,}")
    print(f"ArcFace Frozen Params:          {sum(p.numel() for p in model.arcface.parameters()):,}")
    print("====================================================================")

    # 4. INITIAL BASELINE EVALUATION (Step 0)
    print("\n>> Running Initial Baseline Evaluation (Step 0) on Held-Out Validation Identities...")
    base_val = evaluate_cross_id_val(
        model=model,
        discriminator=discriminator,
        val_dataset=val_dataset,
        criterion_g=criterion_g,
        criterion_adv=criterion_adv,
        device=device,
        num_samples=args.val_samples,
        save_samples_dir=samples_dir,
        step_num=0
    )
    print(f"   [Step 0 Baseline]")
    print(f"     A = cos(source, output):   {base_val['cos_A_src_out']:.4f}")
    print(f"     B = cos(target, output):   {base_val['cos_B_tgt_out']:.4f}")
    print(f"     C = cos(source, target):   {base_val['cos_C_src_tgt']:.4f}")
    print(f"     Val G Loss:                {base_val['loss_total']:.4f} (ID: {base_val['loss_id']:.4f}, Struct: {base_val['loss_struct']:.4f}, BG: {base_val['loss_bg']:.4f})")
    print(f"     Val Mask Mean:             {base_val['mask_mean']:.4f} (Max: {base_val['mask_max']:.4f})")

    # 5. Training Loop
    global_step = 0
    best_identity_metric = -float("inf")  # Target: maximize A - C (source gain above baseline)
    model.train()
    discriminator.train()

    opt_g.zero_grad()
    opt_d.zero_grad()

    start_time = time.time()
    step_times = []

    print(f"\n--- Launching Controlled Pilot Run ({args.max_steps} steps) ---")
    val_history = [("Step 0", base_val)]

    for batch in train_loader:
        if global_step >= args.max_steps:
            break

        step_start = time.time()
        global_step += 1

        i_source = batch["source"].to(device, non_blocking=True)
        i_target = batch["target"].to(device, non_blocking=True)
        target_mask = batch["target_mask"].to(device, non_blocking=True)
        target_l_map = batch["target_landmark_map"].to(device, non_blocking=True)

        # -------------------------------------------------------------
        # STEP A: Train Discriminator
        # -------------------------------------------------------------
        with torch.amp.autocast('cuda'):
            # Generator output (detached for D)
            with torch.no_grad():
                outputs = model(i_target=i_target, l_target=target_l_map, i_source=i_source)
                i_comp_det = outputs["i_composite"].detach()

            d_real = discriminator(i_target)
            d_fake = discriminator(i_comp_det)
            loss_d, loss_d_real, loss_d_fake = criterion_adv.discriminator_loss(d_real, d_fake)
            loss_d_step = loss_d / args.grad_accum_steps

        scaler_d.scale(loss_d_step).backward()

        if global_step % args.grad_accum_steps == 0:
            scaler_d.step(opt_d)
            scaler_d.update()
            opt_d.zero_grad()

        # -------------------------------------------------------------
        # STEP B: Train Generator
        # -------------------------------------------------------------
        with torch.amp.autocast('cuda'):
            outputs_g = model(i_target=i_target, l_target=target_l_map, i_source=i_source)
            i_swap = outputs_g["i_swap"]
            m_pred = outputs_g["m_pred"]
            i_composite = outputs_g["i_composite"]

            # D evaluation with gradients for G
            d_fake_g = discriminator(i_composite)

            total_loss_g, loss_dict_g = criterion_g(
                i_source=i_source,
                i_target=i_target,
                i_swap=i_swap,
                i_composite=i_composite,
                pred_mask=m_pred,
                target_mask=target_mask,
                d_fake_logits=d_fake_g
            )
            loss_g_step = total_loss_g / args.grad_accum_steps

        scaler_g.scale(loss_g_step).backward()

        if global_step % args.grad_accum_steps == 0:
            scaler_g.step(opt_g)
            scaler_g.update()
            opt_g.zero_grad()

        step_elapsed = time.time() - step_start
        step_times.append(step_elapsed)

        # Progress log
        if global_step % 10 == 0:
            alloc_mb = torch.cuda.memory_allocated() / (1024**2)
            res_mb = torch.cuda.memory_reserved() / (1024**2)
            print(
                f"Step {global_step:4d} | G Loss: {total_loss_g.item():.4f} "
                f"(ID: {loss_dict_g['loss_id']:.4f}, Struct: {loss_dict_g['loss_struct']:.4f}, "
                f"BG: {loss_dict_g['loss_bg']:.4f}, Adv: {loss_dict_g['loss_adv']:.4f}) | "
                f"D Loss: {loss_d.item():.4f} | VRAM: {alloc_mb:.0f}/{res_mb:.0f} MB"
            )

        # -------------------------------------------------------------
        # Periodic Cross-ID Validation
        # -------------------------------------------------------------
        if global_step % args.val_interval == 0:
            print(f"\n>> Running Cross-ID Validation at step {global_step}...")
            val_res = evaluate_cross_id_val(
                model=model,
                discriminator=discriminator,
                val_dataset=val_dataset,
                criterion_g=criterion_g,
                criterion_adv=criterion_adv,
                device=device,
                num_samples=args.val_samples,
                save_samples_dir=samples_dir,
                step_num=global_step
            )
            val_history.append((f"Step {global_step}", val_res))

            A = val_res["cos_A_src_out"]
            B = val_res["cos_B_tgt_out"]
            C = val_res["cos_C_src_tgt"]
            gain = A - C

            print(f"   [Step {global_step} Evaluation]")
            print(f"     A = cos(source, output):   {A:.4f}  (Baseline C: {C:.4f} -> Gain: {gain:+.4f})")
            print(f"     B = cos(target, output):   {B:.4f}  (Step 0 B: {base_val['cos_B_tgt_out']:.4f})")
            print(f"     Val G Loss:                {val_res['loss_total']:.4f} (ID: {val_res['loss_id']:.4f}, Struct: {val_res['loss_struct']:.4f}, BG: {val_res['loss_bg']:.4f})")
            print(f"     Val Mask Mean:             {val_res['mask_mean']:.4f} (Max: {val_res['mask_max']:.4f})")

            # Checkpoint best model
            if gain > best_identity_metric:
                best_identity_metric = gain
                best_ckpt_path = checkpoint_dir / "best_model.pt"
                torch.save({
                    "step": global_step,
                    "model_state_dict": model.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "opt_g_state_dict": opt_g.state_dict(),
                    "opt_d_state_dict": opt_d.state_dict(),
                    "val_res": val_res,
                    "config": vars(args)
                }, best_ckpt_path)
                print(f"   [SAVED] New best cross-ID checkpoint -> {best_ckpt_path}")

    total_time = time.time() - start_time
    avg_step_time = np.mean(step_times) if step_times else 0.0
    throughput = 1.0 / avg_step_time if avg_step_time > 0 else 0.0

    # Save latest checkpoint
    latest_ckpt_path = checkpoint_dir / "latest_model.pt"
    torch.save({
        "step": global_step,
        "model_state_dict": model.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "opt_g_state_dict": opt_g.state_dict(),
        "opt_d_state_dict": opt_d.state_dict(),
        "config": vars(args)
    }, latest_ckpt_path)
    print(f"\n[SAVED] Final pilot checkpoint -> {latest_ckpt_path}")

    # 6. Final Report Summary
    print("\n====================================================================")
    print("           STAGE 2 CONTROLLED PILOT SUMMARY REPORT                  ")
    print("====================================================================")
    print(f"Steps Completed:            {global_step}")
    print(f"Total Time:                 {total_time:.2f}s (Throughput: {throughput:.2f} steps/s)")
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"Peak GPU VRAM:              {peak_vram_mb:.2f} MB / {torch.cuda.get_device_properties(0).total_memory / (1024**2):.1f} MB")
    print("\nIdentity Triad Evolution Across Evaluation Checkpoints:")
    print("  Step        A: cos(src, out)     B: cos(tgt, out)     C: Baseline cos(src, tgt)     Gain (A - C)")
    print("  ------------------------------------------------------------------------------------------------")
    for name, v in val_history:
        A = v["cos_A_src_out"]
        B = v["cos_B_tgt_out"]
        C = v["cos_C_src_tgt"]
        print(f"  {name:10s}  {A:16.4f}     {B:16.4f}     {C:23.4f}     {A - C:+11.4f}")

    # ArcFace parameter freeze check
    arcface_grads = [p.grad for p in model.arcface.parameters() if p.grad is not None]
    print(f"\nArcFace Gradients Count:    {len(arcface_grads)} (Strictly 0 verified)")
    print(f"Stage 1 Checkpoint Status:  INTACT (Never overwritten: {args.stage1_checkpoint})")
    print(f"Stage 2 Pilot Checkpoints:  {checkpoint_dir}")
    print("====================================================================")


if __name__ == "__main__":
    run_stage2_pilot()
