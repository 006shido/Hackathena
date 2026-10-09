#!/usr/bin/env python3
"""
Model Architecture Verification Test Suite for Hackathena Face-Swap
Tests all 15 Phase 3B verification criteria:
  1. Model construction succeeds.
  2. Source input [B,3,128,128] -> ArcFace preprocessing -> [B,512].
  3. z_id is L2 normalized.
  4. Target low-frequency image has shape [B,3,128,128].
  5. Landmark map has shape [B,1,128,128].
  6. TargetStructureEncoder produces [B,512,8,8].
  7. AdaIN receives z_id [B,512].
  8. Generator produces I_swap [B,3,128,128] and M_pred [B,1,128,128].
  9. I_swap is bounded to [-1,1].
  10. M_pred is bounded to [0,1].
  11. Soft compositor produces [B,3,128,128].
  12. A scalar test loss can be backpropagated.
  13. Relevant trainable parameters have non-zero gradients.
  14. ArcFace checkpoint absence still raises a clear FileNotFoundError.
  15. No fake/random model or checkpoint is substituted.

Note: Random tensors are used strictly as in-memory numerical inputs for testing
tensor shapes and gradient flow. This is NOT training data.
"""

import sys
import os
import json
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def run_tests():
    print("====================================================================")
    print("     Hackathena Neural Face-Swap Architecture Test Suite            ")
    print("====================================================================")

    if not TORCH_AVAILABLE:
        print("[BLOCKED] PyTorch ('torch') is not installed in the active Python environment.")
        print("          Please install PyTorch: pip install torch torchvision")
        print("====================================================================")
        return False

    from ml.models.modules import (
        gaussian_blur_2d,
        AdaIN2d,
        AdaINResBlock2d,
        TargetStructureEncoder,
        IdentityConditionedGenerator,
        SoftCompositor
    )
    from ml.models.face_swap_model import (
        AdaINFaceSwapModel,
        ArcFaceIdentityExtractor
    )
    from ml.models.iresnet import iresnet50

    batch_size = 2
    height = 128
    width = 128

    # Numerical test inputs (strictly for verifying dimensional consistency and gradient flow)
    torch.manual_seed(42)
    test_i_source = torch.randn(batch_size, 3, height, width).clamp(-1.0, 1.0)
    test_i_target = torch.randn(batch_size, 3, height, width).clamp(-1.0, 1.0)
    test_l_target = torch.rand(batch_size, 1, height, width).clamp(0.0, 1.0)
    test_z_id = torch.randn(batch_size, 512)
    test_z_id = test_z_id / torch.norm(test_z_id, p=2, dim=1, keepdim=True)

    test_results = {}

    # ------------------------------------------------------------------
    # Criterion 1: Model construction succeeds
    # ------------------------------------------------------------------
    print("\n[Criterion 1] Testing model construction...")
    try:
        model = AdaINFaceSwapModel(
            base_channels=64,
            bottleneck_channels=512,
            embedding_dim=512,
            blur_kernel_size=9,
            blur_sigma=3.0
        )
        assert isinstance(model.target_encoder, TargetStructureEncoder)
        assert isinstance(model.generator, IdentityConditionedGenerator)
        assert isinstance(model.compositor, SoftCompositor)
        print("  [PASS] AdaINFaceSwapModel instantiated successfully with all submodules.")
        test_results["1. Model Construction"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] Model construction failed: {e}")
        test_results["1. Model Construction"] = f"FAIL ({e})"

    # ------------------------------------------------------------------
    # Criterion 2: Source input [B,3,128,128] -> ArcFace preprocessing -> [B,512]
    # ------------------------------------------------------------------
    print("\n[Criterion 2] Testing Source input [B,3,128,128] -> ArcFace preprocessing -> [B,512]...")
    try:
        iresnet = iresnet50(num_features=512)
        # ArcFace input preprocessing: bilinear resize 128x128 -> 112x112
        source_112 = F.interpolate(test_i_source, size=(112, 112), mode='bilinear', align_corners=False)
        assert source_112.shape == (batch_size, 3, 112, 112)

        z_id_raw = iresnet(source_112)
        assert z_id_raw.shape == (batch_size, 512), f"Expected {(batch_size, 512)}, got {z_id_raw.shape}"
        print(f"  [PASS] ArcFace backbone produced embedding of shape: {z_id_raw.shape}")
        test_results["2. Source Input Preprocessing to [B, 512]"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] Source preprocessing failed: {e}")
        test_results["2. Source Input Preprocessing to [B, 512]"] = f"FAIL ({e})"

    # ------------------------------------------------------------------
    # Criterion 3: z_id is L2 normalized
    # ------------------------------------------------------------------
    print("\n[Criterion 3] Testing z_id L2 normalization...")
    try:
        norms = torch.norm(z_id_raw, p=2, dim=1)
        expected_norms = torch.ones(batch_size)
        torch.testing.assert_close(norms, expected_norms, atol=1e-5, rtol=1e-5)
        print(f"  [PASS] z_id L2 norm verified: {norms.tolist()} (strictly unit length).")
        test_results["3. z_id L2 Normalization"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] z_id L2 normalization failed: {e}")
        test_results["3. z_id L2 Normalization"] = f"FAIL ({e})"

    # ------------------------------------------------------------------
    # Criterion 4: Target low-frequency image has shape [B,3,128,128]
    # ------------------------------------------------------------------
    print("\n[Criterion 4] Testing Target low-frequency image shape [B,3,128,128]...")
    try:
        i_target_low = gaussian_blur_2d(test_i_target, kernel_size=9, sigma=3.0)
        assert i_target_low.shape == (batch_size, 3, height, width), f"Expected {(batch_size, 3, height, width)}, got {i_target_low.shape}"
        assert not torch.allclose(i_target_low, test_i_target), "Gaussian blur must attenuate high-frequency pixel variations"
        print(f"  [PASS] Target low-frequency image shape verified: {i_target_low.shape}")
        test_results["4. Target Low-Frequency Image Shape"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] Target low-frequency image shape failed: {e}")
        test_results["4. Target Low-Frequency Image Shape"] = f"FAIL ({e})"

    # ------------------------------------------------------------------
    # Criterion 5: Landmark map has shape [B,1,128,128]
    # ------------------------------------------------------------------
    print("\n[Criterion 5] Testing Landmark map shape [B,1,128,128]...")
    try:
        assert test_l_target.shape == (batch_size, 1, height, width), f"Expected {(batch_size, 1, height, width)}, got {test_l_target.shape}"
        # Combined 4-channel input
        x_tgt = torch.cat([i_target_low, test_l_target], dim=1)
        assert x_tgt.shape == (batch_size, 4, height, width), f"Expected {(batch_size, 4, height, width)}, got {x_tgt.shape}"
        print(f"  [PASS] Landmark map shape verified: {test_l_target.shape} (Combined X_tgt: {x_tgt.shape})")
        test_results["5. Landmark Map Shape [B, 1, 128, 128]"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] Landmark map test failed: {e}")
        test_results["5. Landmark Map Shape [B, 1, 128, 128]"] = f"FAIL ({e})"

    # ------------------------------------------------------------------
    # Criterion 6: TargetStructureEncoder produces [B,512,8,8]
    # ------------------------------------------------------------------
    print("\n[Criterion 6] Testing TargetStructureEncoder output shape [B,512,8,8]...")
    try:
        f_tgt = model.target_encoder(x_tgt)
        expected_bottleneck = (batch_size, 512, 8, 8)
        assert f_tgt.shape == expected_bottleneck, f"Expected {expected_bottleneck}, got {f_tgt.shape}"
        print(f"  [PASS] TargetStructureEncoder bottleneck verified: {f_tgt.shape}")
        test_results["6. TargetStructureEncoder Shape [B, 512, 8, 8]"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] TargetStructureEncoder failed: {e}")
        test_results["6. TargetStructureEncoder Shape [B, 512, 8, 8]"] = f"FAIL ({e})"

    # ------------------------------------------------------------------
    # Criterion 7: AdaIN receives z_id [B,512]
    # ------------------------------------------------------------------
    print("\n[Criterion 7] Testing AdaIN receiving z_id [B,512]...")
    try:
        adain = AdaIN2d(num_features=512, embedding_dim=512)
        modulated = adain(f_tgt, test_z_id)
        assert modulated.shape == f_tgt.shape, f"Expected {f_tgt.shape}, got {modulated.shape}"
        print(f"  [PASS] AdaIN correctly received z_id [B, 512] and modulated features: {modulated.shape}")
        test_results["7. AdaIN Receives z_id [B, 512]"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] AdaIN test failed: {e}")
        test_results["7. AdaIN Receives z_id [B, 512]"] = f"FAIL ({e})"

    # ------------------------------------------------------------------
    # Criterion 8: Generator produces I_swap [B,3,128,128] and M_pred [B,1,128,128]
    # ------------------------------------------------------------------
    print("\n[Criterion 8] Testing Generator output shapes I_swap and M_pred...")
    try:
        i_swap, m_pred = model.generator(f_tgt, test_z_id)
        assert i_swap.shape == (batch_size, 3, height, width), f"Expected {(batch_size, 3, height, width)}, got {i_swap.shape}"
        assert m_pred.shape == (batch_size, 1, height, width), f"Expected {(batch_size, 1, height, width)}, got {m_pred.shape}"
        print(f"  [PASS] Generator produced I_swap {i_swap.shape} and M_pred {m_pred.shape}")
        test_results["8. Generator Produces I_swap and M_pred"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] Generator output test failed: {e}")
        test_results["8. Generator Produces I_swap and M_pred"] = f"FAIL ({e})"

    # ------------------------------------------------------------------
    # Criterion 9: I_swap is bounded to [-1,1]
    # ------------------------------------------------------------------
    print("\n[Criterion 9] Testing I_swap bounds in [-1.0, 1.0]...")
    try:
        min_val = i_swap.min().item()
        max_val = i_swap.max().item()
        assert min_val >= -1.0 and max_val <= 1.0, f"I_swap out of bounds: [{min_val}, {max_val}]"
        print(f"  [PASS] I_swap is strictly bounded in [-1.0, 1.0] (observed: [{min_val:.4f}, {max_val:.4f}])")
        test_results["9. I_swap Bounded in [-1, 1]"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] I_swap bounds test failed: {e}")
        test_results["9. I_swap Bounded in [-1, 1]"] = f"FAIL ({e})"

    # ------------------------------------------------------------------
    # Criterion 10: M_pred is bounded to [0,1]
    # ------------------------------------------------------------------
    print("\n[Criterion 10] Testing M_pred bounds in [0.0, 1.0]...")
    try:
        min_mask = m_pred.min().item()
        max_mask = m_pred.max().item()
        assert min_mask >= 0.0 and max_mask <= 1.0, f"M_pred out of bounds: [{min_mask}, {max_mask}]"
        print(f"  [PASS] M_pred is strictly bounded in [0.0, 1.0] (observed: [{min_mask:.4f}, {max_mask:.4f}])")
        test_results["10. M_pred Bounded in [0, 1]"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] M_pred bounds test failed: {e}")
        test_results["10. M_pred Bounded in [0, 1]"] = f"FAIL ({e})"

    # ------------------------------------------------------------------
    # Criterion 11: Soft compositor produces [B,3,128,128]
    # ------------------------------------------------------------------
    print("\n[Criterion 11] Testing Soft compositor output shape [B,3,128,128]...")
    try:
        i_composite = model.compositor(i_swap, m_pred, test_i_target)
        assert i_composite.shape == (batch_size, 3, height, width), f"Expected {(batch_size, 3, height, width)}, got {i_composite.shape}"
        assert i_composite.min().item() >= -1.0 and i_composite.max().item() <= 1.0

        # Test boundary mathematical exactness: when M_pred == 0, output must be identically I_target
        zero_mask = torch.zeros_like(m_pred)
        exact_target = model.compositor(i_swap, zero_mask, test_i_target)
        assert torch.allclose(exact_target, test_i_target), "Zero mask must preserve target exactly"
        print(f"  [PASS] Soft compositor produced {i_composite.shape} in [-1.0, 1.0] with verified boundary math.")
        test_results["11. Soft Compositor Shape [B, 3, 128, 128]"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] Soft compositor test failed: {e}")
        test_results["11. Soft Compositor Shape [B, 3, 128, 128]"] = f"FAIL ({e})"

    # ------------------------------------------------------------------
    # Criterion 12: A scalar test loss can be backpropagated
    # ------------------------------------------------------------------
    print("\n[Criterion 12] Testing scalar test loss backpropagation...")
    try:
        model.train()
        outputs = model(
            i_target=test_i_target,
            l_target=test_l_target,
            z_id=test_z_id
        )
        test_loss = outputs["i_composite"].mean() + outputs["m_pred"].mean()
        test_loss.backward()
        print(f"  [PASS] Scalar test loss {test_loss.item():.4f} successfully backpropagated.")
        test_results["12. Scalar Loss Backpropagation"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] Backpropagation test failed: {e}")
        test_results["12. Scalar Loss Backpropagation"] = f"FAIL ({e})"

    # ------------------------------------------------------------------
    # Criterion 13: Relevant trainable parameters have non-zero gradients
    # ------------------------------------------------------------------
    print("\n[Criterion 13] Testing non-zero gradient propagation to trainable parameters...")
    try:
        generator_params_with_grad = [
            (name, p.grad.abs().sum().item())
            for name, p in model.generator.named_parameters()
            if p.grad is not None and p.grad.abs().sum().item() > 0
        ]
        encoder_params_with_grad = [
            (name, p.grad.abs().sum().item())
            for name, p in model.target_encoder.named_parameters()
            if p.grad is not None and p.grad.abs().sum().item() > 0
        ]

        assert len(generator_params_with_grad) > 0, "No generator parameters received gradients"
        assert len(encoder_params_with_grad) > 0, "No encoder parameters received gradients"
        print(f"  [PASS] Confirmed non-zero gradients across {len(generator_params_with_grad)} generator parameter tensors.")
        print(f"  [PASS] Confirmed non-zero gradients across {len(encoder_params_with_grad)} encoder parameter tensors.")
        test_results["13. Relevant Trainable Parameters Non-Zero Gradients"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] Gradient verification failed: {e}")
        test_results["13. Relevant Trainable Parameters Non-Zero Gradients"] = f"FAIL ({e})"

    # ------------------------------------------------------------------
    # Criterion 14: ArcFace checkpoint absence raises clear FileNotFoundError
    # ------------------------------------------------------------------
    print("\n[Criterion 14] Testing ArcFace checkpoint absence raises FileNotFoundError...")
    try:
        non_existent_ckpt = "ml/models/weights/non_existent_arcface_weights_9999.pth"
        try:
            _ = ArcFaceIdentityExtractor(checkpoint_path=non_existent_ckpt)
            assert False, "Expected FileNotFoundError for non-existent checkpoint"
        except FileNotFoundError as e:
            assert "ArcFace pretrained checkpoint not found" in str(e)
            print(f"  [PASS] Correctly raised FileNotFoundError on missing checkpoint:")
            print(f"         '{e.args[0].splitlines()[0]}'")
            test_results["14. ArcFace Absence Raises FileNotFoundError"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] ArcFace absence test failed: {e}")
        test_results["14. ArcFace Absence Raises FileNotFoundError"] = f"FAIL ({e})"

    # ------------------------------------------------------------------
    # Criterion 15: No fake/random model or checkpoint is substituted
    # ------------------------------------------------------------------
    print("\n[Criterion 15] Verifying zero fake/random model substitution...")
    try:
        # Verify that without checkpoint, ArcFace extractor does NOT invent random weights
        extractor = ArcFaceIdentityExtractor(checkpoint_path=None)
        assert extractor.checkpoint_path is None
        # Verify AdaINFaceSwapModel forward raises clear error if asked to extract identity without weights
        try:
            model_no_weights = AdaINFaceSwapModel(arcface_checkpoint_path=None)
            model_no_weights(i_target=test_i_target, l_target=test_l_target, i_source=test_i_source)
            assert False, "Expected error when attempting to extract identity without ArcFace weights"
        except RuntimeError as e:
            assert "ArcFace extractor is not initialized" in str(e)
            print(f"  [PASS] Verified zero fake model fallback: {e}")
            test_results["15. Zero Fake/Random Model Substitution"] = "PASS"
    except Exception as e:
        print(f"  [FAIL] Fake model verification failed: {e}")
        test_results["15. Zero Fake/Random Model Substitution"] = f"FAIL ({e})"

    # ------------------------------------------------------------------
    # Real ArcFace Pretrained Checkpoint Embedding Test (Configured Path)
    # ------------------------------------------------------------------
    print("\n[Real ArcFace Checkpoint Test] Checking configured weights on disk...")
    config_path = Path("ml/training/config.json")
    configured_ckpt = "ml/models/weights/ms1mv2_iresnet50.pth"
    if config_path.is_file():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                configured_ckpt = cfg.get("arcface_checkpoint_path", configured_ckpt)
        except Exception:
            pass

    ckpt_file = Path(configured_ckpt)
    if ckpt_file.is_file():
        try:
            extractor_real = ArcFaceIdentityExtractor(checkpoint_path=str(ckpt_file))
            real_z = extractor_real(test_i_source)
            assert real_z.shape == (batch_size, 512)
            print(f"  [PASS] Real ArcFace weights loaded from '{ckpt_file}' and produced verified embeddings.")
            test_results["Real ArcFace Pretrained Embedding Test"] = "PASS"
        except Exception as e:
            print(f"  [FAIL] Real ArcFace loading failed: {e}")
            test_results["Real ArcFace Pretrained Embedding Test"] = f"FAIL ({e})"
    else:
        print(f"  [BLOCKED] Real ArcFace checkpoint file '{configured_ckpt}' is NOT present on disk.")
        print("            In accordance with strict safety rules, model weights are NOT downloaded automatically.")
        print("            No fake checkpoint was created.")
        test_results["Real ArcFace Pretrained Embedding Test"] = f"BLOCKED (Checkpoint not found: {configured_ckpt})"

    # ------------------------------------------------------------------
    # Final Summary Table
    # ------------------------------------------------------------------
    print("\n====================================================================")
    print("                     PHASE 3B TEST SUMMARY                          ")
    print("====================================================================")
    passed_count = sum(1 for v in test_results.values() if v == "PASS")
    total_criteria = len(test_results)

    for test_name, status in test_results.items():
        if status == "PASS":
            print(f"  [PASS]    {test_name}")
        elif "BLOCKED" in status:
            print(f"  [BLOCKED] {test_name}: {status.replace('BLOCKED ', '')}")
        else:
            print(f"  [FAIL]    {test_name}: {status}")

    print("--------------------------------------------------------------------")
    print(f"Architecture/Runtime Tests: {passed_count}/{total_criteria} PASSED")
    print("====================================================================")

    return all(v == "PASS" for v in test_results.values())


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
