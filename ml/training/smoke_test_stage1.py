"""
Stage 1 Training Smoke Test for Hackathena Neural Face-Swap
Verifies:
  1. Dataset loading with genuine CelebA same-ID pair
  2. Complete forward pass with AdaINFaceSwapModel + verified ArcFace
  3. Calculation of all Stage 1 losses
  4. Backward pass execution
  5. Gradient verification:
     - ArcFace has NO gradients (frozen)
     - Generator receives gradients
     - Target encoder receives gradients
     - Mask head receives gradients
  6. Optimizer step execution
  7. No NaN, no Inf, finite values throughout
"""

import sys
import os
from pathlib import Path
from typing import Optional
import torch
import torch.optim as optim

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.data.dataset import CelebAPairedDataset
from ml.models.face_swap_model import AdaINFaceSwapModel, ArcFaceIdentityExtractor
from ml.models.losses import Stage1CompositeLoss


def run_stage1_smoke_test(
    dataset_root: str = "ml/data/celeba",
    arcface_ckpt: str = "ml/models/weights/ms1mv2_iresnet50.pth",
    device: Optional[str] = None
) -> bool:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("====================================================================")
    print("      STAGE 1 SAME-ID RECONSTRUCTION SMOKE TEST                     ")
    print(f"      Device: {device.upper()}                                      ")
    if device == "cuda":
        print(f"      GPU:    {torch.cuda.get_device_name(0)}                      ")
    print("====================================================================")

    # 1. Load Dataset
    print("[Step 1] Loading CelebA paired dataset for same-ID sampling...")
    dataset = CelebAPairedDataset(
        dataset_root=dataset_root,
        split="train",
        image_size=128,
        same_identity_probability=1.0,  # 100% same-ID for Stage 1
        seed=42
    )
    print(f"  Dataset initialized with {len(dataset.samples)} samples across {len(dataset.split_identity_to_images)} train identities.")

    # 2. Get one genuine same-ID pair
    print("[Step 2] Fetching 1 genuine same-ID pair...")
    batch = dataset[0]
    i_source = batch["source"].unsqueeze(0).to(device)         # [1, 3, 128, 128]
    i_target = batch["target"].unsqueeze(0).to(device)         # [1, 3, 128, 128]
    target_mask = batch["target_mask"].unsqueeze(0).to(device) # [1, 1, 128, 128]
    target_l_map = batch["target_landmark_map"].unsqueeze(0).to(device) # [1, 1, 128, 128]

    assert batch["pair_type"] == "same_identity", "Sample must be same-identity!"
    assert batch["source_path"] != batch["target_path"], "Source and target must be distinct image files!"
    assert batch["source_identity"] == batch["target_identity"], "Identity IDs must be identical!"
    print(f"  Sample verified: Person ID={batch['source_identity']}")
    print(f"    Source: {Path(batch['source_path']).name}")
    print(f"    Target: {Path(batch['target_path']).name}")

    # 3. Initialize Model and Loss
    print("[Step 3] Initializing AdaINFaceSwapModel with verified ArcFace...")
    model = AdaINFaceSwapModel(
        base_channels=64,
        bottleneck_channels=512,
        embedding_dim=512,
        arcface_checkpoint_path=arcface_ckpt
    ).to(device)

    # Loss manager
    criterion = Stage1CompositeLoss(
        arcface_extractor=model.arcface,
        w_recon=10.0,
        w_id=5.0,
        w_bg=5.0,
        w_mask=2.0
    ).to(device)

    # Optimizer
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        betas=(0.5, 0.999)
    )

    # 4. Forward Pass
    print("[Step 4] Running forward pass...")
    optimizer.zero_grad()
    outputs = model(
        i_target=i_target,
        l_target=target_l_map,
        i_source=i_source
    )
    i_swap = outputs["i_swap"]
    m_pred = outputs["m_pred"]
    i_composite = outputs["i_composite"]

    # Numerical sanity checks
    assert torch.isfinite(i_swap).all(), "NaN or Inf in i_swap!"
    assert torch.isfinite(m_pred).all(), "NaN or Inf in m_pred!"
    assert torch.isfinite(i_composite).all(), "NaN or Inf in i_composite!"
    assert i_swap.min() >= -1.0 and i_swap.max() <= 1.0, "i_swap out of range [-1, 1]"
    assert m_pred.min() >= 0.0 and m_pred.max() <= 1.0, "m_pred out of range [0, 1]"
    print(f"  Forward pass outputs verified: i_composite shape={i_composite.shape}, m_pred shape={m_pred.shape}")

    # 5. Compute Losses
    print("[Step 5] Computing Stage 1 losses...")
    total_loss, loss_dict = criterion(
        i_source=i_source,
        i_target=i_target,
        i_swap=i_swap,
        i_composite=i_composite,
        pred_mask=m_pred,
        target_mask=target_mask
    )
    for k, v in loss_dict.items():
        assert torch.isfinite(torch.tensor(v)), f"Non-finite loss in {k}: {v}"
        print(f"  {k}: {v:.6f}")

    # 6. Backward Pass
    print("[Step 6] Running backward pass...")
    total_loss.backward()

    # 7. Verify Gradients
    print("[Step 7] Verifying gradient propagation rules...")
    # ArcFace MUST have zero gradients
    arcface_grads = [p.grad for p in model.arcface.parameters() if p.grad is not None]
    assert len(arcface_grads) == 0, f"VIOLATION: ArcFace received gradients! Count={len(arcface_grads)}"
    print("  [PASS] ArcFace backbone has STRICTLY zero gradients (remains frozen).")

    # TargetStructureEncoder must have non-zero gradients
    encoder_grads = [p.grad for p in model.target_encoder.parameters() if p.grad is not None and torch.norm(p.grad) > 0]
    assert len(encoder_grads) > 0, "TargetStructureEncoder received no gradients!"
    print(f"  [PASS] TargetStructureEncoder received non-zero gradients across {len(encoder_grads)} tensors.")

    # Generator must have non-zero gradients
    gen_grads = [p.grad for p in model.generator.parameters() if p.grad is not None and torch.norm(p.grad) > 0]
    assert len(gen_grads) > 0, "Generator received no gradients!"
    print(f"  [PASS] Generator received non-zero gradients across {len(gen_grads)} tensors.")

    # Mask head must have non-zero gradients
    mask_head_grads = [p.grad for p in model.generator.mask_head.parameters() if p.grad is not None and torch.norm(p.grad) > 0]
    assert len(mask_head_grads) > 0, "Mask head received no gradients!"
    print(f"  [PASS] Mask head received non-zero gradients across {len(mask_head_grads)} tensors.")

    # 8. Optimizer Step
    print("[Step 8] Executing optimizer step...")
    optimizer.step()
    print("  [PASS] Optimizer step completed successfully.")

    if device == "cuda":
        alloc_mb = torch.cuda.memory_allocated() / (1024**2)
        reserved_mb = torch.cuda.memory_reserved() / (1024**2)
        max_alloc_mb = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"\n[VRAM Metrics on {torch.cuda.get_device_name(0)}]:")
        print(f"  Allocated VRAM:     {alloc_mb:.2f} MB")
        print(f"  Peak Allocated:     {max_alloc_mb:.2f} MB")
        print(f"  Reserved VRAM:      {reserved_mb:.2f} MB")
        print(f"  Total GPU Memory:   {torch.cuda.get_device_properties(0).total_memory / (1024**2):.2f} MB")
        print(f"  VRAM Utilization:   {(max_alloc_mb / (torch.cuda.get_device_properties(0).total_memory / (1024**2))) * 100:.2f}%")

    print("\n====================================================================")
    print("            STAGE 1 SMOKE TEST PASSED (ALL CHECKS OK)               ")
    print("====================================================================")
    return True


if __name__ == "__main__":
    success = run_stage1_smoke_test()
    sys.exit(0 if success else 1)
