#!/usr/bin/env python3
"""
Stage 2 Cross-Identity + Adversarial Training Smoke Test
Audits and verifies:
  1. Initialization from ml/checkpoints/stage1/best_model.pt
  2. Device placement and parameter breakdown (trainable vs frozen)
  3. ArcFace strictly frozen (0 gradients)
  4. Generator & TargetEncoder trainable
  5. PatchGAN Discriminator integration & real/fake patch logits
  6. Genuine CelebA cross-ID pair sampling (id(source) != id(target))
  7. Forward pass execution on CUDA
  8. Calculation of all Stage 2 losses (ID, Struct, BG, Mask, Adv, Disc)
  9. Backward pass execution & gradient verification across all modules
  10. ArcFace cosine similarity triad (src->out, tgt->out, src->tgt baseline)
  11. Mask behavior statistics
  12. GPU VRAM monitoring
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.data.dataset import CelebAPairedDataset
from ml.models.face_swap_model import AdaINFaceSwapModel
from ml.models.discriminator import PatchGANDiscriminator, AdversarialLoss
from ml.models.losses import Stage2CompositeLoss


def run_stage2_smoke_test(
    stage1_checkpoint_path: str = "ml/checkpoints/stage1/best_model.pt",
    arcface_checkpoint_path: str = "ml/models/weights/ms1mv2_iresnet50.pth",
    dataset_root: str = "ml/data/celeba"
):
    print("====================================================================")
    print("     PHASE 6B: STAGE 2 CROSS-ID + ADVERSARIAL SMOKE TEST            ")
    print("====================================================================")

    # 1. Device verification
    assert torch.cuda.is_available(), "CUDA is required for Stage 2 training!"
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    print(f"Device:                   {device} ({gpu_name})")
    print(f"Total GPU VRAM:           {total_vram_mb:.1f} MB")
    print("--------------------------------------------------------------------")

    # 2. Initialize Models & Load Pretrained Stage 1 Generator
    print("[1/8] Initializing AdaINFaceSwapModel & Loading Stage 1 Checkpoint...")
    model = AdaINFaceSwapModel(
        base_channels=64,
        bottleneck_channels=512,
        embedding_dim=512,
        arcface_checkpoint_path=arcface_checkpoint_path
    ).to(device)

    stage1_ckpt = torch.load(stage1_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(stage1_ckpt["model_state_dict"])
    print(f"  Loaded weights from {stage1_checkpoint_path} (trained to step {stage1_ckpt.get('global_step')})")

    # ArcFace must be strictly frozen
    model.arcface.eval()
    for p in model.arcface.parameters():
        p.requires_grad = False

    # Initialize PatchGAN Discriminator
    print("[2/8] Initializing PatchGAN Discriminator...")
    discriminator = PatchGANDiscriminator(in_channels=3, base_channels=64).to(device)
    adv_criterion = AdversarialLoss().to(device)

    # Parameter counts
    gen_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    gen_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    disc_trainable = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print(f"  Generator Trainable Params:     {gen_trainable:,}")
    print(f"  Generator Frozen Params:        {gen_frozen:,} (ArcFace)")
    print(f"  Discriminator Trainable Params: {disc_trainable:,}")
    print(f"  Total Active Parameters:        {gen_trainable + disc_trainable + gen_frozen:,}")

    # Loss suite & optimizers
    stage2_criterion = Stage2CompositeLoss(
        arcface_extractor=model.arcface,
        w_id=10.0,
        w_struct=5.0,
        w_bg=5.0,
        w_mask=2.0,
        w_adv=1.0
    ).to(device)

    opt_g = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

    # 3. Sample genuine cross-ID pair from CelebA
    print("[3/8] Fetching 1 Genuine Cross-ID Pair from CelebA...")
    dataset = CelebAPairedDataset(
        dataset_root=dataset_root,
        split="train",
        image_size=128,
        same_identity_probability=0.0,  # 100% CROSS-ID
        seed=42
    )

    batch = dataset[0]
    i_source = batch["source"].unsqueeze(0).to(device)
    i_target = batch["target"].unsqueeze(0).to(device)
    target_mask = batch["target_mask"].unsqueeze(0).to(device)
    target_l_map = batch["target_landmark_map"].unsqueeze(0).to(device)

    source_id = batch["source_identity"]
    target_id = batch["target_identity"]
    source_filename = Path(batch["source_path"]).name
    target_filename = Path(batch["target_path"]).name

    print(f"  Cross-ID Pair Verified:")
    print(f"    Source: Identity {source_id} ({source_filename})")
    print(f"    Target: Identity {target_id} ({target_filename})")
    print(f"    Pair Type: {batch['pair_type']}")
    assert source_id != target_id, f"FAIL: Source ({source_id}) and Target ({target_id}) must differ!"
    assert source_filename != target_filename, "FAIL: Source and target filenames must differ!"
    print("  [PASS] Genuine cross-identity pair confirmed.")

    # 4. Generator Forward Pass
    print("[4/8] Running Generator Forward Pass...")
    opt_g.zero_grad()
    opt_d.zero_grad()

    outputs = model(i_target=i_target, l_target=target_l_map, i_source=i_source)
    i_swap = outputs["i_swap"]
    m_pred = outputs["m_pred"]
    i_composite = outputs["i_composite"]

    assert torch.isfinite(i_swap).all(), "Non-finite values in i_swap!"
    assert torch.isfinite(m_pred).all(), "Non-finite values in m_pred!"
    assert torch.isfinite(i_composite).all(), "Non-finite values in i_composite!"
    print(f"  Output Shapes: i_composite={list(i_composite.shape)}, m_pred={list(m_pred.shape)}")

    # 5. Discriminator Forward Pass & Discriminator Backward
    print("[5/8] Running Discriminator Real/Fake Classification & Backward...")
    # D on real target face
    d_real_logits = discriminator(i_target)
    # D on fake composite face (detached for D update)
    d_fake_logits_det = discriminator(i_composite.detach())

    loss_d, loss_d_real, loss_d_fake = adv_criterion.discriminator_loss(d_real_logits, d_fake_logits_det)
    loss_d.backward()
    opt_d.step()
    print(f"  Discriminator Logits: Real Mean={d_real_logits.mean().item():.3f}, Fake Mean={d_fake_logits_det.mean().item():.3f}")
    print(f"  Loss D: Total={loss_d.item():.4f} (Real={loss_d_real.item():.4f}, Fake={loss_d_fake.item():.4f})")

    # 6. Generator Loss & Generator Backward
    print("[6/8] Running Generator Stage 2 Loss & Backward...")
    # D on fake composite (with gradients for G)
    d_fake_logits_g = discriminator(i_composite)

    loss_g, loss_dict = stage2_criterion(
        i_source=i_source,
        i_target=i_target,
        i_swap=i_swap,
        i_composite=i_composite,
        pred_mask=m_pred,
        target_mask=target_mask,
        d_fake_logits=d_fake_logits_g
    )
    loss_g.backward()
    opt_g.step()

    print(f"  Generator Stage 2 Losses:")
    for k, v in loss_dict.items():
        print(f"    - {k:15s}: {v:.6f}")

    # 7. Gradient Verification
    print("[7/8] Verifying Gradient Propagation Rules...")
    # ArcFace MUST have zero gradients
    arcface_grads = [p.grad for p in model.arcface.parameters() if p.grad is not None]
    assert len(arcface_grads) == 0, f"VIOLATION: ArcFace received gradients! Count={len(arcface_grads)}"
    print("  [PASS] ArcFace backbone strictly frozen (0 gradients).")

    # Target encoder, generator, mask head, discriminator must have nonzero gradients
    enc_grads = [p.grad for p in model.target_encoder.parameters() if p.grad is not None and torch.norm(p.grad) > 0]
    gen_grads = [p.grad for p in model.generator.parameters() if p.grad is not None and torch.norm(p.grad) > 0]
    mask_grads = [p.grad for p in model.generator.mask_head.parameters() if p.grad is not None and torch.norm(p.grad) > 0]
    disc_grads = [p.grad for p in discriminator.parameters() if p.grad is not None and torch.norm(p.grad) > 0]

    assert len(enc_grads) > 0, "TargetStructureEncoder received no gradients!"
    assert len(gen_grads) > 0, "Generator received no gradients!"
    assert len(mask_grads) > 0, "Mask head received no gradients!"
    assert len(disc_grads) > 0, "Discriminator received no gradients!"

    print(f"  [PASS] TargetStructureEncoder active gradients: {len(enc_grads)} tensors.")
    print(f"  [PASS] Generator active gradients:              {len(gen_grads)} tensors.")
    print(f"  [PASS] Mask Head active gradients:             {len(mask_grads)} tensors.")
    print(f"  [PASS] Discriminator active gradients:         {len(disc_grads)} tensors.")

    # 8. ArcFace Identity Triad & Mask Statistics
    print("[8/8] Evaluating Identity Triad & Mask Statistics...")
    with torch.no_grad():
        z_src = model.arcface(i_source)
        z_out = model.arcface(i_composite)
        z_tgt = model.arcface(i_target)

        cos_src_out = (z_src * z_out).sum().item()
        cos_tgt_out = (z_tgt * z_out).sum().item()
        cos_src_tgt = (z_src * z_tgt).sum().item()

    m_np = m_pred.detach().cpu().numpy().squeeze()
    mask_mean = float(m_np.mean())
    mask_min = float(m_np.min())
    mask_max = float(m_np.max())
    mask_pct_50 = float((m_np > 0.5).mean() * 100.0)

    alloc_mb = torch.cuda.memory_allocated() / (1024**2)
    max_alloc_mb = torch.cuda.max_memory_allocated() / (1024**2)
    res_mb = torch.cuda.memory_reserved() / (1024**2)

    print(f"  Identity Triad (Pretrained Stage 1 Init):")
    print(f"    - ArcFace Source -> Output Cosine:   {cos_src_out:.4f}")
    print(f"    - ArcFace Target -> Output Cosine:   {cos_tgt_out:.4f}")
    print(f"    - ArcFace Source -> Target Baseline: {cos_src_tgt:.4f}")
    print(f"  Mask Statistics:")
    print(f"    - Mean: {mask_mean:.4f}, Min: {mask_min:.4f}, Max: {mask_max:.4f}, Pixels > 0.5: {mask_pct_50:.2f}%")
    print(f"  VRAM Usage on {gpu_name}:")
    print(f"    - Allocated: {alloc_mb:.2f} MB, Peak: {max_alloc_mb:.2f} MB, Reserved: {res_mb:.2f} MB")

    print("\n====================================================================")
    print("          STAGE 2 SMOKE TEST PASSED (ALL CHECKS OK)                 ")
    print("====================================================================")
    return True


if __name__ == "__main__":
    run_stage2_smoke_test()
