#!/usr/bin/env python3
"""
Phase 6C: Source Identity Information Bottleneck Diagnostic.
Compares Condition A (Phase 6B.2 baseline) vs Condition B (Source Feature Conditioning Prototype)
over a controlled 100-step training test on genuine CelebA cross-identity pairs.
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
from ml.models.modules import IdentityConditionedGenerator
from ml.models.source_feature_encoder import SourceFeatureEncoder
from ml.models.discriminator import PatchGANDiscriminator, AdversarialLoss
from ml.models.losses import Stage2CompositeLoss


class SourceConditionedGenerator(IdentityConditionedGenerator):
    """
    Generator Decoder with Source Spatial Feature Injection.
    Injects source spatial feature maps alongside target structural features and 512-D ArcFace AdaIN.
    """
    def __init__(self, bottleneck_channels: int = 512, embedding_dim: int = 512):
        super().__init__(bottleneck_channels=bottleneck_channels, embedding_dim=embedding_dim)
        self.src_proj8 = nn.Conv2d(512, 512, kernel_size=1)
        self.src_proj16 = nn.Conv2d(256, 256, kernel_size=1)
        self.src_proj32 = nn.Conv2d(128, 128, kernel_size=1)
        self.src_proj64 = nn.Conv2d(64, 64, kernel_size=1)

        # Zero initialization for clean identity baseline at step 0
        for proj in [self.src_proj8, self.src_proj16, self.src_proj32, self.src_proj64]:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(
        self,
        f_tgt: torch.Tensor,
        z_id: torch.Tensor,
        src_feats: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.res1(f_tgt, z_id)
        x = self.res2(x, z_id)
        x = self.res3(x, z_id)
        x = self.res4(x, z_id)
        if src_feats is not None:
            x = x + self.src_proj8(src_feats["f8"])

        x = nn.functional.interpolate(x, scale_factor=2.0, mode='nearest')
        x = self.conv_up1(x)
        x = self.res_up1(x, z_id)
        if src_feats is not None:
            x = x + self.src_proj16(src_feats["f16"])

        x = nn.functional.interpolate(x, scale_factor=2.0, mode='nearest')
        x = self.conv_up2(x)
        x = self.res_up2(x, z_id)
        if src_feats is not None:
            x = x + self.src_proj32(src_feats["f32"])

        x = nn.functional.interpolate(x, scale_factor=2.0, mode='nearest')
        x = self.conv_up3(x)
        x = self.res_up3(x, z_id)
        if src_feats is not None:
            x = x + self.src_proj64(src_feats["f64"])

        x = nn.functional.interpolate(x, scale_factor=2.0, mode='nearest')
        x = self.conv_up4(x)
        x = self.res_up4(x, z_id)

        i_swap = self.rgb_head(x)
        m_pred = self.mask_head(x)
        return i_swap, m_pred


class SourceConditionedFaceSwapModel(AdaINFaceSwapModel):
    """Face swap model with source feature encoder and source spatial conditioning."""
    def __init__(
        self,
        base_channels: int = 64,
        bottleneck_channels: int = 512,
        embedding_dim: int = 512,
        blur_kernel_size: int = 9,
        blur_sigma: float = 3.0,
        arcface_checkpoint_path: Optional[str] = None
    ):
        super().__init__(
            base_channels=base_channels,
            bottleneck_channels=bottleneck_channels,
            embedding_dim=embedding_dim,
            blur_kernel_size=blur_kernel_size,
            blur_sigma=blur_sigma,
            arcface_checkpoint_path=arcface_checkpoint_path
        )
        self.source_encoder = SourceFeatureEncoder(in_channels=3, base_channels=base_channels)
        self.generator = SourceConditionedGenerator(
            bottleneck_channels=bottleneck_channels,
            embedding_dim=embedding_dim
        )

    def forward(
        self,
        i_target: torch.Tensor,
        l_target: torch.Tensor,
        z_id: Optional[torch.Tensor] = None,
        i_source: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        if z_id is None:
            if i_source is None:
                raise ValueError("Either z_id or i_source must be provided.")
            z_id = self.arcface(i_source)

        src_feats = None
        if i_source is not None:
            src_feats = self.source_encoder(i_source)

        x_tgt = self.prepare_target_input(i_target, l_target)
        f_tgt = self.target_encoder(x_tgt)
        i_swap, m_pred = self.generator(f_tgt, z_id, src_feats=src_feats)
        i_composite = self.compositor(i_swap, m_pred, i_target)

        return {
            "i_swap": i_swap,
            "m_pred": m_pred,
            "i_composite": i_composite,
            "f_tgt": f_tgt,
            "z_id": z_id,
            "src_feats": src_feats
        }


def evaluate_model(
    model: nn.Module,
    cached_batches: List[Dict[str, torch.Tensor]],
    device: torch.device = torch.device("cuda")
) -> Tuple[Dict[str, float], List[Dict[str, torch.Tensor]]]:
    model.eval()
    triad_A, triad_B, triad_C, triad_D = [], [], [], []
    mask_in, mask_out = [], []
    sample_records = []

    with torch.no_grad():
        for batch in cached_batches:
            src = batch["source"].to(device)
            tgt = batch["target"].to(device)
            tgt_mask = batch["target_mask"].to(device)
            l_map = batch["target_landmark_map"].to(device)

            out = model(i_target=tgt, l_target=l_map, i_source=src)
            i_swap = out["i_swap"]
            m_pred = out["m_pred"]
            i_comp = out["i_composite"]

            z_src = model.arcface(src)
            z_out = model.arcface(i_comp)
            z_tgt = model.arcface(tgt)
            z_swap = model.arcface(i_swap)

            cos_A = (z_src * z_out).sum(dim=-1).item()
            cos_B = (z_tgt * z_out).sum(dim=-1).item()
            cos_C = (z_src * z_tgt).sum(dim=-1).item()
            cos_D = (z_src * z_swap).sum(dim=-1).item()

            triad_A.append(cos_A)
            triad_B.append(cos_B)
            triad_C.append(cos_C)
            triad_D.append(cos_D)

            m_np = m_pred.cpu().numpy().squeeze()
            tgt_m_np = tgt_mask.cpu().numpy().squeeze()
            face_p = tgt_m_np > 0.5
            bg_p = tgt_m_np <= 0.5
            in_m = float(m_np[face_p].mean()) if face_p.any() else 0.0
            out_m = float(m_np[bg_p].mean()) if bg_p.any() else 0.0
            mask_in.append(in_m)
            mask_out.append(out_m)

            sample_records.append({
                "source": src.cpu(),
                "target": tgt.cpu(),
                "i_swap": i_swap.cpu(),
                "m_pred": m_pred.cpu(),
                "i_composite": i_comp.cpu()
            })

    model.train()
    metrics = {
        "A": float(np.mean(triad_A)),
        "B": float(np.mean(triad_B)),
        "C": float(np.mean(triad_C)),
        "D": float(np.mean(triad_D)),
        "gain_A": float(np.mean(triad_A)) - float(np.mean(triad_C)),
        "gain_D": float(np.mean(triad_D)) - float(np.mean(triad_C)),
        "mask_inside": float(np.mean(mask_in)),
        "mask_outside": float(np.mean(mask_out))
    }
    return metrics, sample_records


def run_phase6c_diagnostic():
    parser = argparse.ArgumentParser(description="Phase 6C Diagnostic")
    parser.add_argument("--init-checkpoint", default="ml/checkpoints/stage2/best_model.pt")
    parser.add_argument("--arcface-checkpoint", default="ml/models/weights/ms1mv2_iresnet50.pth")
    parser.add_argument("--dataset-root", default="ml/data/celeba")
    parser.add_argument("--output-dir", default="ml/checkpoints/stage2/phase6c_diagnostic")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--val-samples", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr-g", type=float, default=1e-4)
    parser.add_argument("--lr-d", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required for Phase 6C diagnostic!"
    device = torch.device("cuda")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.output_dir)
    samples_dir = out_dir / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("      PHASE 6C: SOURCE IDENTITY INFORMATION BOTTLENECK DIAGNOSTIC     ")
    print("=" * 76)

    # 1. Validation Dataset
    val_dataset = CelebAPairedDataset(
        dataset_root=args.dataset_root,
        split="val",
        image_size=128,
        same_identity_probability=0.0,
        seed=args.seed
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    cached_val_batches = []
    for i, batch in enumerate(val_loader):
        if i >= args.val_samples:
            break
        cached_val_batches.append(batch)

    train_dataset = CelebAPairedDataset(
        dataset_root=args.dataset_root,
        split="train",
        image_size=128,
        same_identity_probability=0.0,
        seed=args.seed
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    # 2. Evaluate CONDITION A: Baseline Phase 6B.2 model
    print("\n--- Evaluating Condition A: Current Baseline (Phase 6B.2 Best Checkpoint) ---")
    base_model = AdaINFaceSwapModel(
        base_channels=64,
        bottleneck_channels=512,
        embedding_dim=512,
        blur_sigma=3.0,
        arcface_checkpoint_path=args.arcface_checkpoint
    ).to(device)
    base_ckpt = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
    base_model.load_state_dict(base_ckpt["model_state_dict"])
    base_model.arcface.eval()
    for p in base_model.arcface.parameters():
        p.requires_grad = False

    cond_A_metrics, base_records = evaluate_model(base_model, cached_val_batches, device)
    print(f"Condition A Metrics (Baseline):")
    print(f"  A = cos(src, out):      {cond_A_metrics['A']:.4f}")
    print(f"  B = cos(tgt, out):      {cond_A_metrics['B']:.4f}")
    print(f"  C = cos(src, tgt) base: {cond_A_metrics['C']:.4f}")
    print(f"  D = cos(src, swap):     {cond_A_metrics['D']:.4f}")
    print(f"  A - C (Gain):           {cond_A_metrics['gain_A']:+.4f}")
    print(f"  D - C (Gain):           {cond_A_metrics['gain_D']:+.4f}")
    print(f"  Mask Inside Face:       {cond_A_metrics['mask_inside']:.4f}")

    del base_model
    torch.cuda.empty_cache()

    # 3. Initialize CONDITION B: Source Feature Conditioning Prototype
    print("\n--- Initializing Condition B: Source Feature Conditioning Prototype ---")
    model_B = SourceConditionedFaceSwapModel(
        base_channels=64,
        bottleneck_channels=512,
        embedding_dim=512,
        blur_sigma=3.0,
        arcface_checkpoint_path=args.arcface_checkpoint
    ).to(device)

    # Load compatible Phase 6B.2 weights
    model_state = model_B.state_dict()
    ckpt_state = base_ckpt["model_state_dict"]
    loaded_keys = []
    missing_keys = []
    for k in ckpt_state:
        if k in model_state:
            model_state[k] = ckpt_state[k]
            loaded_keys.append(k)
    for k in model_state:
        if not k.startswith("arcface.") and k not in ckpt_state:
            missing_keys.append(k)
    model_B.load_state_dict(model_state, strict=False)

    print(f"  Loaded existing weights:        {len(loaded_keys)}")
    print(f"  Newly initialized weights:      {len(missing_keys)}")
    print(f"  Source encoder params:          {sum(p.numel() for p in model_B.source_encoder.parameters()):,}")
    print(f"  Generator spatial proj params:  {sum(p.numel() for proj in [model_B.generator.src_proj8, model_B.generator.src_proj16, model_B.generator.src_proj32, model_B.generator.src_proj64] for p in proj.parameters()):,}")

    model_B.arcface.eval()
    for p in model_B.arcface.parameters():
        p.requires_grad = False

    discriminator = PatchGANDiscriminator(in_channels=3, base_channels=64).to(device)
    if "discriminator_state_dict" in base_ckpt and base_ckpt["discriminator_state_dict"] is not None:
        discriminator.load_state_dict(base_ckpt["discriminator_state_dict"])

    criterion_g = Stage2CompositeLoss(
        arcface_extractor=model_B.arcface,
        w_id=10.0,
        w_id_swap=8.0,
        w_struct=5.0,
        w_bg=5.0,
        w_mask=5.0,
        w_adv=0.5
    ).to(device)
    criterion_adv = AdversarialLoss().to(device)

    opt_g = optim.Adam([p for p in model_B.parameters() if p.requires_grad], lr=args.lr_g, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    scaler_g = torch.amp.GradScaler('cuda')
    scaler_d = torch.amp.GradScaler('cuda')

    # Step 0 evaluation of Condition B
    cond_B_step0, _ = evaluate_model(model_B, cached_val_batches, device)
    print(f"\nCondition B @ Step 0:")
    print(f"  A = {cond_B_step0['A']:.4f}, B = {cond_B_step0['B']:.4f}, C = {cond_B_step0['C']:.4f}, D = {cond_B_step0['D']:.4f} | A-C = {cond_B_step0['gain_A']:+.4f}, D-C = {cond_B_step0['gain_D']:+.4f}")

    # 4. Controlled 100-step training test
    print(f"\n--- Running Controlled 100-Step Diagnostic Training Test ---")
    start_time = time.time()
    global_step = 0
    model_B.train()
    discriminator.train()
    opt_g.zero_grad()
    opt_d.zero_grad()

    while global_step < args.max_steps:
        for batch in train_loader:
            if global_step >= args.max_steps:
                break
            global_step += 1

            i_source = batch["source"].to(device, non_blocking=True)
            i_target = batch["target"].to(device, non_blocking=True)
            target_mask = batch["target_mask"].to(device, non_blocking=True)
            target_l_map = batch["target_landmark_map"].to(device, non_blocking=True)

            # Discriminator
            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    outputs = model_B(i_target=i_target, l_target=target_l_map, i_source=i_source)
                    i_comp_det = outputs["i_composite"].detach()
                d_real = discriminator(i_target)
                d_fake = discriminator(i_comp_det)
                loss_d, _, _ = criterion_adv.discriminator_loss(d_real, d_fake)
                loss_d_step = loss_d / args.grad_accum_steps

            scaler_d.scale(loss_d_step).backward()
            if global_step % args.grad_accum_steps == 0:
                scaler_d.step(opt_d)
                scaler_d.update()
                opt_d.zero_grad()

            # Generator
            with torch.amp.autocast('cuda'):
                outputs_g = model_B(i_target=i_target, l_target=target_l_map, i_source=i_source)
                i_swap = outputs_g["i_swap"]
                m_pred = outputs_g["m_pred"]
                i_composite = outputs_g["i_composite"]
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

            if global_step % 25 == 0:
                print(f"Step {global_step:3d} | G Loss: {total_loss_g.item():.4f} (ID_out: {loss_dict_g['loss_id_out']:.4f}, ID_swap: {loss_dict_g['loss_id_swap']:.4f}, Struct: {loss_dict_g['loss_struct']:.4f}) | D: {loss_d.item():.4f}")

    train_time = time.time() - start_time
    print(f"Completed {global_step} steps in {train_time:.2f}s.")

    # 5. Final Condition B Evaluation & Sample Comparison
    print("\n--- Final Evaluation of Condition B (@ Step 100) ---")
    cond_B_step100, b_records = evaluate_model(model_B, cached_val_batches, device)
    print(f"Condition B @ Step 100:")
    print(f"  A = {cond_B_step100['A']:.4f}, B = {cond_B_step100['B']:.4f}, C = {cond_B_step100['C']:.4f}, D = {cond_B_step100['D']:.4f}")
    print(f"  A - C (Gain):     {cond_B_step100['gain_A']:+.4f} (Baseline: {cond_A_metrics['gain_A']:+.4f})")
    print(f"  D - C (Gain):     {cond_B_step100['gain_D']:+.4f} (Baseline: {cond_A_metrics['gain_D']:+.4f})")
    print(f"  Target Leakage B: {cond_B_step100['B']:.4f} (Baseline: {cond_A_metrics['B']:.4f})")
    print(f"  Mask Inside:      {cond_B_step100['mask_inside']:.4f}")

    # 6. Save Comparative Samples: [SOURCE, TARGET, BASELINE_OUT, CONDITION_B_OUT, CONDITION_B_MASK]
    print("\nSaving comparative validation samples...")
    for idx in range(min(4, len(base_records), len(b_records))):
        s_rec = base_records[idx]
        b_rec = b_records[idx]

        src_v = (s_rec["source"][0] + 1.0) / 2.0
        tgt_v = (s_rec["target"][0] + 1.0) / 2.0
        base_comp = (s_rec["i_composite"][0] + 1.0) / 2.0
        cond_b_comp = (b_rec["i_composite"][0] + 1.0) / 2.0
        cond_b_mask = b_rec["m_pred"][0].expand(3, -1, -1)

        grid = torch.cat([src_v, tgt_v, base_comp, cond_b_comp, cond_b_mask], dim=2)
        save_image(grid, samples_dir / f"comparison_sample_{idx}.png")
    print(f"Saved comparative samples to {samples_dir}")

    # 7. Checkpoint & Metrics Saving
    ckpt_path = out_dir / "phase6c_diagnostic_model.pt"
    torch.save({
        "step": global_step,
        "model_state_dict": model_B.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "cond_A_metrics": cond_A_metrics,
        "cond_B_step100_metrics": cond_B_step100,
        "config": vars(args)
    }, ckpt_path)

    metrics_path = out_dir / "phase6c_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "condition_A_baseline": cond_A_metrics,
            "condition_B_step0": cond_B_step0,
            "condition_B_step100": cond_B_step100,
            "delta_gain_A": cond_B_step100["gain_A"] - cond_A_metrics["gain_A"],
            "delta_gain_D": cond_B_step100["gain_D"] - cond_A_metrics["gain_D"],
            "delta_B": cond_B_step100["B"] - cond_A_metrics["B"]
        }, f, indent=2)

    # Integrity verification
    arcface_grads = [p.grad for p in model_B.arcface.parameters() if p.grad is not None]
    assert len(arcface_grads) == 0, "ArcFace received gradients!"

    print("\n" + "=" * 76)
    print("                     PHASE 6C DIAGNOSTIC COMPLETE")
    print("=" * 76)
    print(f"Condition A (Baseline Phase 6B.2): A-C = {cond_A_metrics['gain_A']:+.4f} | D-C = {cond_A_metrics['gain_D']:+.4f} | B = {cond_A_metrics['B']:.4f}")
    print(f"Condition B (Source Spatial Feats): A-C = {cond_B_step100['gain_A']:+.4f} | D-C = {cond_B_step100['gain_D']:+.4f} | B = {cond_B_step100['B']:.4f}")
    print(f"Delta: Delta(A-C) = {cond_B_step100['gain_A'] - cond_A_metrics['gain_A']:+.4f}, Delta(D-C) = {cond_B_step100['gain_D'] - cond_A_metrics['gain_D']:+.4f}, Delta(B) = {cond_B_step100['B'] - cond_A_metrics['B']:+.4f}")
    print("=" * 76)


if __name__ == "__main__":
    run_phase6c_diagnostic()
