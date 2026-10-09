#!/usr/bin/env python3
"""
Stage 1 Training Pipeline: Same-Identity Face Reconstruction for Hackathena
Strictly enforces:
  - 100% genuine CelebA same-ID training pairs (Person A != Person B images, same identity)
  - Deterministic 85/15 identity-disjoint train/validation split (seed 42)
  - Frozen official MS1MV2 ArcFace iResNet-50 identity conditioning (0 gradients)
  - Masked L1 reconstruction loss normalized by active mask area
  - ArcFace cosine identity loss (L_id = 1 - cos(z_src, z_out))
  - Background preservation outside face-oval mask
  - Mask regularization (smoothness & coverage)
  - Automatic Mixed Precision (AMP) and gradient accumulation
  - Checkpointing under ml/checkpoints/stage1/
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.data.dataset import CelebAPairedDataset
from ml.models.face_swap_model import AdaINFaceSwapModel
from ml.models.losses import Stage1CompositeLoss
from ml.training.face_preprocessing import denormalize_image_tensor


def parse_args():
    parser = argparse.ArgumentParser(description="Hackathena Face-Swap Stage 1 Training")
    parser.add_argument("--dataset-root", type=str, default="ml/data/celeba", help="CelebA root directory")
    parser.add_argument("--identity-file", type=str, default="identity_CelebA.txt", help="Identity annotation file")
    parser.add_argument("--images-dir", type=str, default="img_align_celeba", help="Images directory")
    parser.add_argument("--checkpoint-dir", type=str, default="ml/checkpoints/stage1", help="Checkpoint save directory")
    parser.add_argument("--arcface-checkpoint", type=str, default="ml/models/weights/ms1mv2_iresnet50.pth", help="ArcFace weights")
    parser.add_argument("--image-size", type=int, default=128, help="Image resolution (default: 128)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per step (conservative for 8GB VRAM)")
    parser.add_argument("--grad-accum-steps", type=int, default=4, help="Gradient accumulation steps (effective batch size = batch_size * accum)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional max training steps (e.g. for smoke test)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Adam optimizer")
    parser.add_argument("--val-interval", type=int, default=250, help="Steps between validation runs")
    parser.add_argument("--val-samples", type=int, default=20, help="Number of validation pairs to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    return parser.parse_args()


def run_validation(
    model: nn.Module,
    val_dataset: CelebAPairedDataset,
    criterion: Stage1CompositeLoss,
    device: torch.device,
    num_samples: int = 20,
    save_samples_dir: Optional[Path] = None,
    step_num: int = 0
) -> Dict[str, float]:
    """
    Evaluates model on validation identities only. Computes identity cosine preservation and reconstruction loss.
    """
    model.eval()
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    total_metrics = {
        "loss_total": 0.0,
        "loss_recon": 0.0,
        "loss_bg": 0.0,
        "loss_id": 0.0,
        "loss_mask": 0.0,
        "cos_sim_out": 0.0,
        "cos_sim_target": 0.0
    }

    count = 0
    saved_examples = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if count >= num_samples:
                break

            i_source = batch["source"].to(device)
            i_target = batch["target"].to(device)
            target_mask = batch["target_mask"].to(device)
            target_l_map = batch["target_landmark_map"].to(device)

            outputs = model(
                i_target=i_target,
                l_target=target_l_map,
                i_source=i_source
            )
            i_swap = outputs["i_swap"]
            m_pred = outputs["m_pred"]
            i_composite = outputs["i_composite"]

            _, loss_dict = criterion(
                i_source=i_source,
                i_target=i_target,
                i_swap=i_swap,
                i_composite=i_composite,
                pred_mask=m_pred,
                target_mask=target_mask
            )

            # Cosine similarity metrics
            z_src = model.arcface(i_source)
            z_out = model.arcface(i_composite)
            z_tgt = model.arcface(i_target)

            cos_out = (z_src * z_out).sum(dim=-1).mean().item()
            cos_tgt = (z_src * z_tgt).sum(dim=-1).mean().item()

            for k in loss_dict:
                total_metrics[k] += loss_dict[k]
            total_metrics["cos_sim_out"] += cos_out
            total_metrics["cos_sim_target"] += cos_tgt
            count += 1

            # Save qualitative visual sample
            if save_samples_dir and saved_examples < 4:
                save_samples_dir.mkdir(parents=True, exist_ok=True)
                # Map [-1, 1] to [0, 1] for visual saving
                src_vis = (i_source[0].cpu() + 1.0) / 2.0
                tgt_vis = (i_target[0].cpu() + 1.0) / 2.0
                swap_vis = (i_swap[0].cpu() + 1.0) / 2.0
                mask_vis = m_pred[0].cpu().expand(3, -1, -1)
                comp_vis = (i_composite[0].cpu() + 1.0) / 2.0

                grid = torch.cat([src_vis, tgt_vis, swap_vis, mask_vis, comp_vis], dim=2)
                save_image(grid, save_samples_dir / f"step_{step_num}_sample_{saved_examples}.png")
                saved_examples += 1

    model.train()

    if count == 0:
        return {k: 0.0 for k in total_metrics}

    return {k: v / count for k, v in total_metrics.items()}


def train_stage1():
    args = parse_args()

    # 1. Hardware & CUDA verification
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("====================================================================")
    print("     HACKATHENA STAGE 1: SAME-ID RECONSTRUCTION TRAINING            ")
    print("====================================================================")
    print(f"Device:           {device}")
    if use_cuda:
        print(f"GPU Name:         {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM:       {torch.cuda.get_device_properties(0).total_memory / (1024**2):.1f} MB")
    else:
        print("WARNING: CUDA is not available. Running on CPU.")
    print("--------------------------------------------------------------------")

    # 2. Set random seeds
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = checkpoint_dir / "samples"

    # 3. Datasets
    print(f"Loading datasets from: {args.dataset_root}")
    train_dataset = CelebAPairedDataset(
        dataset_root=args.dataset_root,
        identity_file=args.identity_file,
        images_dir=args.images_dir,
        split="train",
        image_size=args.image_size,
        same_identity_probability=1.0,  # 100% same-ID for Stage 1
        seed=args.seed
    )
    val_dataset = CelebAPairedDataset(
        dataset_root=args.dataset_root,
        identity_file=args.identity_file,
        images_dir=args.images_dir,
        split="val",
        image_size=args.image_size,
        same_identity_probability=1.0,  # 100% same-ID for Stage 1 validation
        seed=args.seed
    )

    print(f"Train identities:      {len(train_dataset.split_identity_to_images)}")
    print(f"Train samples:         {len(train_dataset)}")
    print(f"Validation identities: {len(val_dataset.split_identity_to_images)}")
    print(f"Validation samples:    {len(val_dataset)}")
    print("--------------------------------------------------------------------")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=use_cuda
    )

    # 4. Initialize Model
    print("Initializing AdaINFaceSwapModel...")
    model = AdaINFaceSwapModel(
        base_channels=64,
        bottleneck_channels=512,
        embedding_dim=512,
        blur_kernel_size=9,
        blur_sigma=3.0,
        arcface_checkpoint_path=args.arcface_checkpoint
    ).to(device)

    # ArcFace is strictly frozen
    model.arcface.eval()
    for p in model.arcface.parameters():
        p.requires_grad = False

    # 5. Initialize Loss and Optimizer
    criterion = Stage1CompositeLoss(
        arcface_extractor=model.arcface,
        w_recon=10.0,
        w_id=5.0,
        w_bg=5.0,
        w_mask=2.0
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=args.lr, betas=(0.5, 0.999))

    use_amp = use_cuda and (not args.no_amp)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if use_cuda else torch.amp.GradScaler('cpu', enabled=False)
    print(f"AMP enabled:           {use_amp}")
    print(f"Batch size:            {args.batch_size} (Grad Accum: {args.grad_accum_steps}, Effective: {args.batch_size * args.grad_accum_steps})")
    print(f"Learning rate:         {args.lr}")
    print("====================================================================")

    # 6. Training Loop
    global_step = 0
    best_val_loss = float("inf")
    model.train()

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        epoch_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            global_step += 1

            i_source = batch["source"].to(device, non_blocking=True)
            i_target = batch["target"].to(device, non_blocking=True)
            target_mask = batch["target_mask"].to(device, non_blocking=True)
            target_l_map = batch["target_landmark_map"].to(device, non_blocking=True)

            with torch.amp.autocast('cuda' if use_cuda else 'cpu', enabled=use_amp):
                outputs = model(
                    i_target=i_target,
                    l_target=target_l_map,
                    i_source=i_source
                )
                i_swap = outputs["i_swap"]
                m_pred = outputs["m_pred"]
                i_composite = outputs["i_composite"]

                total_loss, loss_dict = criterion(
                    i_source=i_source,
                    i_target=i_target,
                    i_swap=i_swap,
                    i_composite=i_composite,
                    pred_mask=m_pred,
                    target_mask=target_mask
                )

                # Normalize loss for gradient accumulation
                loss_step = total_loss / args.grad_accum_steps

            # Backward pass with scaler
            scaler.scale(loss_step).backward()

            # Optimizer step on accumulation boundary
            if global_step % args.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += total_loss.item()

            # Progress logging
            if global_step % 10 == 0:
                vram_info = ""
                if use_cuda:
                    alloc_mb = torch.cuda.memory_allocated() / (1024**2)
                    res_mb = torch.cuda.memory_reserved() / (1024**2)
                    vram_info = f" | VRAM: {alloc_mb:.0f}/{res_mb:.0f} MB"
                print(
                    f"Step {global_step:5d} | Loss: {total_loss.item():.4f} "
                    f"(Recon: {loss_dict['loss_recon']:.4f}, ID: {loss_dict['loss_id']:.4f}, "
                    f"BG: {loss_dict['loss_bg']:.4f}, Mask: {loss_dict['loss_mask']:.4f}){vram_info}"
                )

            # Validation checkpoint
            if global_step % args.val_interval == 0:
                print(f"\n>> Running Validation at step {global_step}...")
                val_metrics = run_validation(
                    model=model,
                    val_dataset=val_dataset,
                    criterion=criterion,
                    device=device,
                    num_samples=args.val_samples,
                    save_samples_dir=samples_dir,
                    step_num=global_step
                )
                print(f"   Val Total Loss:  {val_metrics['loss_total']:.4f}")
                print(f"   Val Recon Loss:  {val_metrics['loss_recon']:.4f}")
                print(f"   Val ID Loss:     {val_metrics['loss_id']:.4f}")
                print(f"   Val BG Loss:     {val_metrics['loss_bg']:.4f}")
                print(f"   Val Cos Sim:     {val_metrics['cos_sim_out']:.4f} (Source-Target baseline: {val_metrics['cos_sim_target']:.4f})")

                # Save best checkpoint
                if val_metrics["loss_total"] < best_val_loss:
                    best_val_loss = val_metrics["loss_total"]
                    best_ckpt_path = checkpoint_dir / "best_model.pt"
                    torch.save({
                        "epoch": epoch,
                        "global_step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict() if use_amp else None,
                        "best_val_loss": best_val_loss,
                        "val_metrics": val_metrics,
                        "config": vars(args)
                    }, best_ckpt_path)
                    print(f"   [SAVED] New best validation checkpoint -> {best_ckpt_path}")

            # Check max steps limit
            if args.max_steps is not None and global_step >= args.max_steps:
                print(f"\n[STOP] Reached max_steps limit: {args.max_steps}")
                break

        if args.max_steps is not None and global_step >= args.max_steps:
            break

    # Save final latest checkpoint
    final_ckpt_path = checkpoint_dir / "latest_model.pt"
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if use_amp else None,
        "best_val_loss": best_val_loss,
        "config": vars(args)
    }, final_ckpt_path)
    print(f"\n[SAVED] Final checkpoint saved -> {final_ckpt_path}")
    print(f"Total training time: {time.time() - start_time:.2f}s")

    # Final Integrity & Performance Summary
    print("====================================================================")
    print("                STAGE 1 TRAINING SUMMARY REPORT                     ")
    print("====================================================================")
    print(f"Steps Completed:          {global_step}")
    print(f"Epochs Completed:         {epoch}")
    print(f"Final Train Loss:         {total_loss.item():.4f}")
    print(f"  - Recon Loss:           {loss_dict['loss_recon']:.4f}")
    print(f"  - ArcFace ID Loss:      {loss_dict['loss_id']:.4f}")
    print(f"  - Background Loss:      {loss_dict['loss_bg']:.4f}")
    print(f"  - Mask Reg Loss:        {loss_dict['loss_mask']:.4f}")
    print(f"Best Val Loss:            {best_val_loss:.4f}")
    if use_cuda:
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"Peak GPU VRAM:            {peak_vram_mb:.2f} MB / {torch.cuda.get_device_properties(0).total_memory / (1024**2):.1f} MB")

    # Frozen check verification
    arcface_grads = [p.grad for p in model.arcface.parameters() if p.grad is not None]
    print(f"ArcFace Gradients Count:  {len(arcface_grads)} (Strictly 0 required)")
    print(f"Best Checkpoint:          {checkpoint_dir / 'best_model.pt'}")
    print(f"Latest Checkpoint:        {final_ckpt_path}")
    print("====================================================================")


if __name__ == "__main__":
    train_stage1()
