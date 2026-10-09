"""
Phase 6G: Smoke Test for Correspondence-Aware Multi-Scale Face-Swap Prototype
Verifies forward and backward dynamics on one genuine cross-ID CelebA pair:
  - Source & target identity embeddings
  - 6D aligned source & confidence map
  - Source multi-scale features & target features
  - Decoder output, mask, and composite
  - Gradient checks: ArcFace grads = 0, Generator grads != 0, Skip grads != 0, Discriminator grads != 0
  - NaN / Inf checks
  - CUDA placement & VRAM usage
"""

import sys
import hashlib
from pathlib import Path
import random
import torch
import torch.nn.functional as F

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ml.models.phase6g_model import MultiScaleCorrespondenceFaceSwapModel
from ml.models.discriminator import PatchGANDiscriminator, AdversarialLoss
from ml.models.losses import Stage2CompositeLoss
from ml.data.dataset import CelebAPairedDataset
from ml.training.face_preprocessing import RealFacePreprocessor
from ml.inference.face_correspondence import (
    preprocess_single_face,
    warp_and_prepare_source
)

EXPECTED_ARCFACE_SHA = "2B75B93C48B01C78A4263F7295AB2DBF84F85190C51BE45B92C4C0B0AEDCEAA3"


def run_smoke_test():
    print("=" * 60, flush=True)
    print("PHASE 6G: ARCHITECTURAL SMOKE TEST", flush=True)
    print("=" * 60, flush=True)

    # 1. Device and VRAM
    assert torch.cuda.is_available(), "CUDA is required for Phase 6G smoke test"
    device = torch.device("cuda")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})", flush=True)
    torch.cuda.reset_peak_memory_stats()

    # 2. ArcFace Integrity
    arcface_path = Path("ml/models/weights/ms1mv2_iresnet50.pth")
    assert arcface_path.exists(), f"ArcFace checkpoint missing: {arcface_path}"
    with open(arcface_path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest().upper()
    print(f"ArcFace SHA-256: {sha256}", flush=True)
    assert sha256 == EXPECTED_ARCFACE_SHA, f"ArcFace SHA mismatch! Expected {EXPECTED_ARCFACE_SHA}, got {sha256}"

    # 3. Load 1 genuine cross-ID pair
    print("Loading 1 genuine cross-identity CelebA pair...", flush=True)
    preprocessor = RealFacePreprocessor(image_size=128)
    val_ds = CelebAPairedDataset(dataset_root="ml/data/celeba", split="val", seed=42)
    identities = list(val_ds.split_identity_to_images.keys())
    rng = random.Random(42)
    id_s, id_t = rng.sample(identities, 2)
    assert id_s != id_t, "Source and target identities must be different"
    p_s = rng.choice(val_ds.split_identity_to_images[id_s])
    p_t = rng.choice(val_ds.split_identity_to_images[id_t])

    s_img, s_tensor, s_mask, _, s_dense, s_5pts = preprocess_single_face(preprocessor, str(p_s))
    t_img, t_tensor, t_mask, t_lmap, t_dense, t_5pts = preprocess_single_face(preprocessor, str(p_t))

    aln_src_tensor, conf_tensor = warp_and_prepare_source(s_img, s_dense, t_dense, 128)

    i_src = torch.from_numpy(s_tensor).unsqueeze(0).to(device)
    i_tgt = torch.from_numpy(t_tensor).unsqueeze(0).to(device)
    l_tgt = torch.from_numpy(t_lmap).unsqueeze(0).to(device)
    m_tgt = torch.from_numpy(t_mask).unsqueeze(0).to(device)
    aln_src = aln_src_tensor.to(device)
    conf_map = conf_tensor.to(device)

    print(f"  Source image tensor:      {i_src.shape} on {i_src.device}")
    print(f"  Target image tensor:      {i_tgt.shape} on {i_tgt.device}")
    print(f"  Target landmark map:      {l_tgt.shape} on {l_tgt.device}")
    print(f"  Aligned source tensor:    {aln_src.shape} on {aln_src.device}")
    print(f"  Confidence map tensor:    {conf_map.shape} on {conf_map.device}")

    # 4. Instantiate Models
    print("Instantiating MultiScaleCorrespondenceFaceSwapModel and PatchGANDiscriminator...", flush=True)
    model = MultiScaleCorrespondenceFaceSwapModel(
        base_channels=64,
        bottleneck_channels=512,
        embedding_dim=512,
        blur_kernel_size=9,
        blur_sigma=3.0,
        arcface_checkpoint_path=str(arcface_path)
    ).to(device)

    # Load baseline checkpoint weights
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

    # 5. Forward Pass Checks
    print("\nExecuting forward pass checks...", flush=True)
    model.train()
    discriminator.train()

    # (a) Source and target identity embeddings
    with torch.no_grad():
        z_id_src = model.arcface(i_src)
        z_id_tgt = model.arcface(i_tgt)
    assert z_id_src.shape == (1, 512), f"Expected (1, 512), got {z_id_src.shape}"
    assert z_id_tgt.shape == (1, 512), f"Expected (1, 512), got {z_id_tgt.shape}"
    norm_src = torch.norm(z_id_src, p=2, dim=1).item()
    print(f"  [OK] Source identity vector: shape {z_id_src.shape}, L2 norm = {norm_src:.4f}")
    assert abs(norm_src - 1.0) < 1e-4, f"Identity vector must be L2 normalized, got {norm_src}"

    # (b) Multi-scale source features
    skips = model.source_encoder(aln_src)
    print(f"  [OK] Source multi-scale features:")
    for k in ["s64", "s32", "s16", "s8"]:
        assert k in skips, f"Missing skip {k}"
        print(f"       {k}: shape {skips[k].shape}, min={skips[k].min().item():.3f}, max={skips[k].max().item():.3f}")
        assert not torch.isnan(skips[k]).any(), f"NaN in {k}"
        assert not torch.isinf(skips[k]).any(), f"Inf in {k}"

    # (c) Target features
    x_tgt = model.prepare_target_input(i_tgt, l_tgt)
    f_tgt = model.target_encoder(x_tgt)
    print(f"  [OK] Target bottleneck features: shape {f_tgt.shape}")
    assert f_tgt.shape == (1, 512, 8, 8)
    assert not torch.isnan(f_tgt).any()
    assert not torch.isinf(f_tgt).any()

    # (d) Full Model Forward
    outputs = model(
        i_target=i_tgt,
        l_target=l_tgt,
        i_source=i_src,
        aligned_source=aln_src,
        confidence_map=conf_map
    )

    i_swap = outputs["i_swap"]
    m_pred = outputs["m_pred"]
    i_composite = outputs["i_composite"]

    print(f"  [OK] Decoder i_swap:      shape {i_swap.shape}, range [{i_swap.min().item():.3f}, {i_swap.max().item():.3f}]")
    print(f"  [OK] Decoder m_pred:      shape {m_pred.shape}, range [{m_pred.min().item():.3f}, {m_pred.max().item():.3f}]")
    print(f"  [OK] Compositor i_comp:   shape {i_composite.shape}, range [{i_composite.min().item():.3f}, {i_composite.max().item():.3f}]")

    assert i_swap.shape == (1, 3, 128, 128)
    assert m_pred.shape == (1, 1, 128, 128)
    assert i_composite.shape == (1, 3, 128, 128)
    assert not torch.isnan(i_swap).any() and not torch.isinf(i_swap).any()
    assert not torch.isnan(m_pred).any() and not torch.isinf(m_pred).any()
    assert not torch.isnan(i_composite).any() and not torch.isinf(i_composite).any()

    # 6. Backward Pass & Gradient Checks
    print("\nExecuting backward pass and gradient checks...", flush=True)
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

    opt_g = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

    # (a) Generator Step
    opt_g.zero_grad()
    d_fake_pred = discriminator(i_composite)

    total_g_loss, loss_dict = criterion_g(
        i_source=i_src,
        i_target=i_tgt,
        i_swap=i_swap,
        i_composite=i_composite,
        pred_mask=m_pred,
        target_mask=m_tgt,
        d_fake_logits=d_fake_pred
    )
    print(f"  Generator total loss: {total_g_loss.item():.4f}")
    assert not torch.isnan(total_g_loss) and not torch.isinf(total_g_loss)

    total_g_loss.backward()

    # Gradient assertions:
    # 1. ArcFace gradients MUST be zero / None
    arcface_grad_norms = [p.grad.norm().item() for p in model.arcface.parameters() if p.grad is not None]
    print(f"  [CHECK] ArcFace gradients: {len(arcface_grad_norms)} active gradient tensors (expected 0)")
    assert len(arcface_grad_norms) == 0, "ArcFace parameters received gradients! ArcFace must be frozen."

    # 2. Generator main weights have non-zero gradients
    gen_grad_norm = model.generator.rgb_head[0].weight.grad.norm().item()
    print(f"  [CHECK] Generator RGB head grad norm: {gen_grad_norm:.6f} (expected > 0)")
    assert gen_grad_norm > 0, "Generator RGB head received zero gradients!"

    # 3. Source skips have non-zero gradients
    src_enc_grad = model.source_encoder.layer1[0].weight.grad.norm().item()
    fuse_64_grad = model.generator.fuse_64[0].weight.grad.norm().item()
    fuse_32_grad = model.generator.fuse_32[0].weight.grad.norm().item()
    fuse_16_grad = model.generator.fuse_16[0].weight.grad.norm().item()
    fuse_8_grad  = model.generator.fuse_8[0].weight.grad.norm().item()

    print(f"  [CHECK] Source encoder layer1 grad norm: {src_enc_grad:.6f} (expected > 0)")
    print(f"  [CHECK] Generator fuse_64 grad norm:     {fuse_64_grad:.6f} (expected > 0)")
    print(f"  [CHECK] Generator fuse_32 grad norm:     {fuse_32_grad:.6f} (expected > 0)")
    print(f"  [CHECK] Generator fuse_16 grad norm:     {fuse_16_grad:.6f} (expected > 0)")
    print(f"  [CHECK] Generator fuse_8 grad norm:      {fuse_8_grad:.6f} (expected > 0)")

    assert src_enc_grad > 0, "Source encoder received zero gradients!"
    assert fuse_64_grad > 0, "fuse_64 skip received zero gradients!"
    assert fuse_32_grad > 0, "fuse_32 skip received zero gradients!"
    assert fuse_16_grad > 0, "fuse_16 skip received zero gradients!"
    assert fuse_8_grad > 0, "fuse_8 bottleneck received zero gradients!"

    opt_g.step()

    # (b) Discriminator Step
    opt_d.zero_grad()
    d_real = discriminator(i_tgt)
    d_fake = discriminator(i_composite.detach())
    loss_d, _, _ = criterion_adv.discriminator_loss(d_real, d_fake)
    print(f"  Discriminator total loss: {loss_d.item():.4f}")
    assert not torch.isnan(loss_d) and not torch.isinf(loss_d)

    loss_d.backward()
    d_grad_norm = discriminator.net[0].weight.grad.norm().item()
    print(f"  [CHECK] Discriminator block 0 grad norm: {d_grad_norm:.6f} (expected > 0)")
    assert d_grad_norm > 0, "Discriminator received zero gradients!"
    opt_d.step()

    # 7. Hardware & VRAM Usage
    alloc_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    res_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
    print(f"\nPeak Allocated VRAM: {alloc_mb:.1f} MB / Peak Reserved: {res_mb:.1f} MB", flush=True)
    assert alloc_mb < 6000, f"VRAM usage exceeded threshold: {alloc_mb:.1f} MB"

    print("\n" + "=" * 60, flush=True)
    print("ALL PHASE 6G SMOKE TEST CHECKS PASSED SUCCESSFULLY!", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    run_smoke_test()
