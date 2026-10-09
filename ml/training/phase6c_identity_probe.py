#!/usr/bin/env python3
"""
Phase 6C Identity Information Probe.
Tests whether source spatial feature maps contain measurable identity-discriminative
information by comparing same-identity vs. cross-identity pair representations
across multiple resolutions on held-out CelebA validation identities.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.data.dataset import CelebAPairedDataset
from ml.models.face_swap_model import ArcFaceIdentityExtractor
from ml.models.source_feature_encoder import SourceFeatureEncoder


def compute_feature_cosine(f1: torch.Tensor, f2: torch.Tensor) -> float:
    """Computes cosine similarity between two feature tensors."""
    v1 = f1.flatten(start_dim=1)
    v2 = f2.flatten(start_dim=1)
    cos = F.cosine_similarity(v1, v2, dim=1)
    return cos.mean().item()


def run_identity_probe(
    dataset_root: str = "ml/data/celeba",
    arcface_path: str = "ml/models/weights/ms1mv2_iresnet50.pth",
    output_dir: str = "ml/checkpoints/stage2/phase6c_diagnostic",
    num_pairs: int = 50,
    seed: int = 42
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Identity Probe on {device} ({num_pairs} same-ID & {num_pairs} cross-ID pairs)...")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Models
    arcface = ArcFaceIdentityExtractor(checkpoint_path=arcface_path).to(device)
    arcface.eval()

    encoder = SourceFeatureEncoder(in_channels=3, base_channels=64).to(device)
    encoder.eval()

    # 2. Datasets from held-out validation split
    # Same-identity pairs
    same_ds = CelebAPairedDataset(
        dataset_root=dataset_root,
        split="val",
        image_size=128,
        same_identity_probability=1.0,
        seed=seed
    )
    # Cross-identity pairs
    cross_ds = CelebAPairedDataset(
        dataset_root=dataset_root,
        split="val",
        image_size=128,
        same_identity_probability=0.0,
        seed=seed
    )

    same_loader = DataLoader(same_ds, batch_size=1, shuffle=False)
    cross_loader = DataLoader(cross_ds, batch_size=1, shuffle=False)

    metrics = {
        "arcface": {"same": [], "cross": []},
        "encoder_f8": {"same": [], "cross": []},
        "encoder_f16": {"same": [], "cross": []},
        "encoder_f32": {"same": [], "cross": []},
        "encoder_f64": {"same": [], "cross": []}
    }

    with torch.no_grad():
        # Evaluate Same-Identity Pairs
        print("  Evaluating same-identity pairs...")
        for i, batch in enumerate(same_loader):
            if i >= num_pairs:
                break
            img1 = batch["source"].to(device)
            img2 = batch["target"].to(device)

            z1 = arcface(img1)
            z2 = arcface(img2)
            metrics["arcface"]["same"].append(F.cosine_similarity(z1, z2).item())

            feat1 = encoder(img1)
            feat2 = encoder(img2)
            metrics["encoder_f8"]["same"].append(compute_feature_cosine(feat1["f8"], feat2["f8"]))
            metrics["encoder_f16"]["same"].append(compute_feature_cosine(feat1["f16"], feat2["f16"]))
            metrics["encoder_f32"]["same"].append(compute_feature_cosine(feat1["f32"], feat2["f32"]))
            metrics["encoder_f64"]["same"].append(compute_feature_cosine(feat1["f64"], feat2["f64"]))

        # Evaluate Cross-Identity Pairs
        print("  Evaluating cross-identity pairs...")
        for i, batch in enumerate(cross_loader):
            if i >= num_pairs:
                break
            img1 = batch["source"].to(device)
            img2 = batch["target"].to(device)

            z1 = arcface(img1)
            z2 = arcface(img2)
            metrics["arcface"]["cross"].append(F.cosine_similarity(z1, z2).item())

            feat1 = encoder(img1)
            feat2 = encoder(img2)
            metrics["encoder_f8"]["cross"].append(compute_feature_cosine(feat1["f8"], feat2["f8"]))
            metrics["encoder_f16"]["cross"].append(compute_feature_cosine(feat1["f16"], feat2["f16"]))
            metrics["encoder_f32"]["cross"].append(compute_feature_cosine(feat1["f32"], feat2["f32"]))
            metrics["encoder_f64"]["cross"].append(compute_feature_cosine(feat1["f64"], feat2["f64"]))

    summary = {}
    print("\n" + "=" * 76)
    print("                 PHASE 6C IDENTITY PROBE RESULTS")
    print("=" * 76)
    print(f"{'Feature Layer':<16} | {'Same-ID Mean':<14} | {'Cross-ID Mean':<14} | {'Separation (Delta)':<18}")
    print("-" * 76)

    for k, v in metrics.items():
        same_m = float(np.mean(v["same"]))
        cross_m = float(np.mean(v["cross"]))
        sep = same_m - cross_m
        summary[k] = {
            "same_id_mean": same_m,
            "cross_id_mean": cross_m,
            "separation": sep,
            "same_id_std": float(np.std(v["same"])),
            "cross_id_std": float(np.std(v["cross"]))
        }
        print(f"{k:<16} | {same_m:14.4f} | {cross_m:14.4f} | {sep:+18.4f}")

    print("=" * 76)

    results_file = out_path / "identity_probe_results.json"
    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {results_file}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="ml/data/celeba")
    parser.add_argument("--arcface-path", default="ml/models/weights/ms1mv2_iresnet50.pth")
    parser.add_argument("--output-dir", default="ml/checkpoints/stage2/phase6c_diagnostic")
    parser.add_argument("--num-pairs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_identity_probe(
        dataset_root=args.dataset_root,
        arcface_path=args.arcface_path,
        output_dir=args.output_dir,
        num_pairs=args.num_pairs,
        seed=args.seed
    )
