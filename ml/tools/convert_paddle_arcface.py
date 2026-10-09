#!/usr/bin/env python3
"""
Reproducible PaddlePaddle to PyTorch Checkpoint Converter for InsightFace ArcFace iResNet-50
Converts the official InsightFace MS1MV2 iResNet-50 Paddle parameter file (.pdparams)
inside 'arcface_iresnet50_v1.0_pretrained.tar' to a standard PyTorch state_dict.

Conversion Details:
  - Preserves the original .tar archive completely untouched.
  - Extracts 'arcface_iresnet50_v1.0_pretrained.pdparams' in-memory via tarfile.
  - Maps Paddle parameter naming conventions to PyTorch iResNet-50:
      * '._mean'     -> '.running_mean'
      * '._variance' -> '.running_var'
      * '._weight'   -> '.weight' (PReLU parameter)
  - Transposes Linear layer weight:
      * 'fc.weight' Paddle shape (25088, 512) -> PyTorch shape (512, 25088)
  - Adds 'num_batches_tracked' buffers to each BatchNorm layer so strict=True loading works.
  - Validates full key and tensor shape alignment against ml.models.iresnet.iresnet50().
  - Saves the resulting PyTorch checkpoint to:
      ml/models/weights/ms1mv2_iresnet50.pth
"""

import sys
import os
import io
import pickle
import argparse
import tarfile
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    import torch
    from ml.models.iresnet import iresnet50
except ImportError as e:
    print(f"Error importing PyTorch or iResNet model: {e}")
    sys.exit(1)


def convert_paddle_to_pytorch(
    tar_path: str,
    output_path: str = "ml/models/weights/ms1mv2_iresnet50.pth"
) -> bool:
    tar_file = Path(tar_path)
    if not tar_file.is_file():
        print(f"[ERROR] Source tar archive not found at: '{tar_path}'")
        return False

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Opening archive: '{tar_file}' (Size: {tar_file.stat().st_size / (1024*1024):.2f} MB)...")
    with tarfile.open(str(tar_file), "r") as tf:
        target_member_name = "arcface_iresnet50_v1.0_pretrained/arcface_iresnet50_v1.0_pretrained.pdparams"
        try:
            member = tf.getmember(target_member_name)
        except KeyError:
            # Search for pdparams file in archive
            candidates = [m for m in tf.getmembers() if m.name.endswith(".pdparams")]
            if not candidates:
                print(f"[ERROR] No .pdparams file found inside '{tar_path}'")
                return False
            member = candidates[0]
            print(f"Using found pdparams member: '{member.name}'")

        print(f"Extracting '{member.name}' in-memory ({member.size / (1024*1024):.2f} MB)...")
        extracted_file = tf.extractfile(member)
        if extracted_file is None:
            print("[ERROR] Could not extract member file.")
            return False

        raw_bytes = extracted_file.read()

    print("Deserializing Paddle parameters from pickle stream...")
    paddle_dict = pickle.loads(raw_bytes)
    if not isinstance(paddle_dict, dict):
        print(f"[ERROR] Expected dict from pdparams, got {type(paddle_dict)}")
        return False

    print(f"Loaded {len(paddle_dict)} entries from Paddle parameter dictionary.")

    # Target PyTorch model state_dict for reference
    ref_model = iresnet50(num_features=512)
    ref_state = ref_model.state_dict()

    pytorch_state_dict = {}
    converted_count = 0

    for k, v in paddle_dict.items():
        if k == "StructuredToParameterName@@":
            # Metadata dictionary in Paddle, skip
            continue

        # Parameter name mapping
        py_k = k
        if py_k.endswith("._mean"):
            py_k = py_k[:-6] + ".running_mean"
        elif py_k.endswith("._variance"):
            py_k = py_k[:-10] + ".running_var"
        elif py_k.endswith("._weight"):
            py_k = py_k[:-8] + ".weight"

        tensor_val = torch.from_numpy(v)

        # Transpose linear weight from (in_features, out_features) to (out_features, in_features)
        if py_k == "fc.weight" and tensor_val.shape == (25088, 512):
            tensor_val = tensor_val.t()

        if py_k not in ref_state:
            print(f"[ERROR] Mapped key '{py_k}' does not exist in PyTorch iResNet-50!")
            return False

        expected_shape = ref_state[py_k].shape
        if tensor_val.shape != expected_shape:
            print(f"[ERROR] Shape mismatch for '{py_k}': got {tensor_val.shape}, expected {expected_shape}")
            return False

        pytorch_state_dict[py_k] = tensor_val
        converted_count += 1

    print(f"Successfully mapped {converted_count} parameters.")

    # Add num_batches_tracked scalar buffers for strict=True compatibility
    for k in ref_state.keys():
        if k.endswith(".num_batches_tracked"):
            pytorch_state_dict[k] = torch.tensor(0, dtype=torch.long)

    # Validate strict loading into PyTorch model
    print("Testing strict=True state_dict loading into iResNet-50...")
    try:
        load_result = ref_model.load_state_dict(pytorch_state_dict, strict=True)
        print(f"Model load successful: {load_result}")
    except Exception as e:
        print(f"[ERROR] Failed to load converted state_dict with strict=True: {e}")
        return False

    # Test numerical forward pass
    ref_model.eval()
    test_tensor = torch.randn(1, 3, 112, 112).clamp(-1.0, 1.0)
    with torch.no_grad():
        out = ref_model(test_tensor)

    assert out.shape == (1, 512), f"Expected (1, 512), got {out.shape}"
    assert torch.isfinite(out).all(), "Output contains non-finite numbers"
    l2_norm = torch.norm(out, p=2, dim=1).item()
    assert abs(l2_norm - 1.0) < 1e-4, f"L2 norm not unit length: {l2_norm}"
    print(f"Forward pass verified: shape={out.shape}, L2-norm={l2_norm:.6f}, finite=True.")

    # Save checkpoint
    print(f"Saving PyTorch checkpoint to '{out_file}'...")
    torch.save(pytorch_state_dict, str(out_file))
    print(f"[SUCCESS] Checkpoint saved successfully! Size: {out_file.stat().st_size / (1024*1024):.2f} MB")
    return True


def main():
    default_tar = Path(os.path.expanduser("~")) / "Downloads" / "arcface_iresnet50_v1.0_pretrained.tar"
    default_out = project_root / "ml" / "models" / "weights" / "ms1mv2_iresnet50.pth"

    parser = argparse.ArgumentParser(description="Convert official InsightFace Paddle ArcFace checkpoint to PyTorch.")
    parser.add_argument("--tar-path", type=str, default=str(default_tar), help="Path to arcface_iresnet50_v1.0_pretrained.tar")
    parser.add_argument("--output-path", type=str, default=str(default_out), help="Output path for PyTorch .pth checkpoint")
    args = parser.parse_args()

    success = convert_paddle_to_pytorch(args.tar_path, args.output_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
