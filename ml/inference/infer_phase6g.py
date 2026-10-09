"""
Phase 6G: Standalone Local Inference Entry Point
Runs correspondence-aware multi-scale neural face swapping on CUDA:
  Input:
    - source image path
    - target image path
  Output:
    - swapped composite output image
    - raw swapped face image
    - predicted soft blending mask
    - aligned source warp
    - full visual comparison grid
    - diagnostic metrics JSON (ArcFace cosine similarities, landmark error, runtime)
"""

import os
import sys
import time
import json
import hashlib
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F

from ml.models.phase6g_model import MultiScaleCorrespondenceFaceSwapModel
from ml.training.face_preprocessing import (
    RealFacePreprocessor,
    denormalize_image_tensor
)
from ml.inference.face_correspondence import (
    preprocess_single_face,
    warp_and_prepare_source
)

EXPECTED_ARCFACE_SHA = "2B75B93C48B01C78A4263F7295AB2DBF84F85190C51BE45B92C4C0B0AEDCEAA3"


class Phase6GInferenceEngine:
    def __init__(
        self,
        checkpoint_path: str = "ml/checkpoints/stage2/phase6g/best_model.pt",
        arcface_path: str = "ml/models/weights/ms1mv2_iresnet50.pth",
        device: Optional[torch.device] = None,
        model_variant: str = 'phase6g',
        correspondence: str = 'legacy',
        refine: bool = False
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_variant not in ('phase6g', 'fullres') or correspondence not in ('legacy', 'robust', 'reliable', 'visibility'):
            raise ValueError('Unsupported model variant or correspondence mode.')
        self.correspondence = correspondence
        self.model_variant = model_variant
        self.refine = refine
        print(f"Initializing Phase 6G Inference Engine on {self.device}...", flush=True)

        # 1. Verify ArcFace Checkpoint Integrity
        arcface_p = Path(arcface_path)
        assert arcface_p.exists(), f"ArcFace checkpoint missing: {arcface_p}"
        with open(arcface_p, "rb") as f:
            sha256 = hashlib.sha256(f.read()).hexdigest().upper()
        assert sha256 == EXPECTED_ARCFACE_SHA, f"ArcFace SHA-256 mismatch! Got {sha256}"
        print(f"  ArcFace SHA-256 confirmed: {sha256}", flush=True)

        # 2. Preprocessor
        self.preprocessor = RealFacePreprocessor(image_size=128)

        # 3. Model
        model_class = MultiScaleCorrespondenceFaceSwapModel
        extra = {}
        if model_variant == 'fullres':
            from ml.experiments.phase6g_1_fullres_skip.model_phase6g1 import MultiScaleCorrespondenceFaceSwapModel6G1
            model_class = MultiScaleCorrespondenceFaceSwapModel6G1
            extra['fullres_channels'] = 32
        self.model = model_class(
            base_channels=64,
            bottleneck_channels=512,
            embedding_dim=512,
            blur_kernel_size=9,
            blur_sigma=3.0,
            arcface_checkpoint_path=str(arcface_p),
            **extra
        ).to(self.device)

        ckpt_p = Path(checkpoint_path)
        if not ckpt_p.exists():
            # Fallback to latest_model.pt if best_model.pt is not present
            fallback = ckpt_p.parent / "latest_model.pt"
            if fallback.exists():
                ckpt_p = fallback
            else:
                raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path} or {fallback}")

        state = torch.load(str(ckpt_p), map_location=self.device, weights_only=False)
        model_weights = state["model_state_dict"] if "model_state_dict" in state else state
        self.model.load_state_dict(model_weights)
        self.checkpoint_path = str(ckpt_p.resolve())
        with ckpt_p.open('rb') as checkpoint_file:
            self.checkpoint_sha256 = hashlib.file_digest(checkpoint_file, 'sha256').hexdigest()
        self.model.eval()
        self.model.arcface.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        print(f"  Loaded Phase 6G weights from: {ckpt_p}", flush=True)

    def infer(
        self,
        source_path: str,
        target_path: str,
        output_dir: Optional[str] = None,
        pair_prefix: str = "swap"
    ) -> Dict[str, Any]:
        """
        Executes complete face-swap inference on a genuine image pair.
        """
        t0 = time.time()

        # 1. Preprocess source and target
        s_img, s_tensor, s_mask, _, s_dense, s_5pts = preprocess_single_face(self.preprocessor, source_path)
        t_img, t_tensor, t_mask, t_lmap, t_dense, t_5pts = preprocess_single_face(self.preprocessor, target_path)

        # 2. 6D piecewise-affine correspondence warp
        geometry_diagnostics = None
        if self.correspondence in ('robust', 'reliable', 'visibility'):
            from ml.training.robust_correspondence import robust_warp, reliable_legacy_warp, visibility_warp
            warp = {'robust': robust_warp, 'reliable': reliable_legacy_warp,
                    'visibility': lambda *a: visibility_warp(*a,strength=0.25)}[self.correspondence]
            aln_src_tensor, conf_tensor, geometry_diagnostics = warp(s_img, s_dense, t_dense)
        else:
            aln_src_tensor, conf_tensor = warp_and_prepare_source(s_img, s_dense, t_dense, 128)

        # 3. Prepare tensors on device
        i_src = torch.from_numpy(s_tensor).unsqueeze(0).to(self.device)
        i_tgt = torch.from_numpy(t_tensor).unsqueeze(0).to(self.device)
        l_tgt = torch.from_numpy(t_lmap).unsqueeze(0).to(self.device)
        aln_src = aln_src_tensor.to(self.device)
        conf_map = conf_tensor.to(self.device)

        # 4. Neural Forward Pass
        t_net_start = time.time()
        with torch.no_grad():
            outputs = self.model(
                i_target=i_tgt,
                l_target=l_tgt,
                i_source=i_src,
                aligned_source=aln_src,
                confidence_map=conf_map,
                disable_skips=False
            )
            i_swap = outputs["i_swap"]
            m_pred = outputs["m_pred"]
            i_comp = outputs["i_composite"]

            # Compute diagnostic ArcFace cosine similarities
            z_src = self.model.arcface(i_src)
            z_tgt = self.model.arcface(i_tgt)
            z_comp = self.model.arcface(i_comp)
            z_swap = self.model.arcface(i_swap)

            cos_A = float(F.cosine_similarity(z_src, z_comp).item())
            cos_B = float(F.cosine_similarity(z_tgt, z_comp).item())
            cos_C = float(F.cosine_similarity(z_src, z_tgt).item())
            cos_D = float(F.cosine_similarity(z_src, z_swap).item())

        net_latency_ms = (time.time() - t_net_start) * 1000.0
        total_latency_ms = (time.time() - t0) * 1000.0

        # 5. Convert outputs to PIL & uint8
        def to_u8(t):
            return ((t.squeeze(0).cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

        comp_np = to_u8(i_comp)
        swap_np = to_u8(i_swap)
        aln_np  = to_u8(aln_src)
        src_u8  = to_u8(i_src)
        tgt_u8  = to_u8(i_tgt)
        m_np    = (m_pred.squeeze().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        mask_u8 = np.repeat(m_np[:, :, None], 3, axis=2)

        refinement = None
        if self.refine:
            from ml.inference.face_refinement import refine_face
            def identity_score(rgb):
                tensor = torch.from_numpy(rgb.astype(np.float32) / 127.5 - 1).permute(2, 0, 1).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    return float(F.cosine_similarity(z_src, self.model.arcface(tensor)).item())
            refined, refinement = refine_face(
                swap_np, tgt_u8, m_pred.squeeze().cpu().numpy(), t_mask,
                t_dense, self.preprocessor, identity_score=identity_score
            )
            if refinement['accepted']:
                comp_np = refined
                tensor = torch.from_numpy(comp_np.astype(np.float32) / 127.5 - 1).permute(2, 0, 1).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    z_comp = self.model.arcface(tensor)
                    cos_A = float(F.cosine_similarity(z_src, z_comp).item())
                    cos_B = float(F.cosine_similarity(z_tgt, z_comp).item())

        # 6. Re-detect landmarks on composite output
        comp_pil = Image.fromarray(comp_np)
        redetect_ok = False
        lm_error = None
        try:
            det_comp = self.preprocessor.detect_landmarks(comp_pil)
            if det_comp is not None and det_comp.dense_landmarks is not None:
                redetect_ok = True
                lm_error = float(np.linalg.norm(det_comp.key_landmarks_5pts - t_5pts, axis=1).mean())
        except Exception:
            pass

        # 7. Save outputs if directory requested
        out_paths = {}
        if output_dir is not None:
            out_p = Path(output_dir)
            out_p.mkdir(parents=True, exist_ok=True)

            comp_path = out_p / f"{pair_prefix}_composite.png"
            swap_path = out_p / f"{pair_prefix}_swap.png"
            mask_path = out_p / f"{pair_prefix}_mask.png"
            aln_path  = out_p / f"{pair_prefix}_aligned_source.png"
            grid_path = out_p / f"{pair_prefix}_comparison_grid.png"
            diag_path = out_p / f"{pair_prefix}_diagnostic.json"

            Image.fromarray(comp_np).save(comp_path)
            Image.fromarray(swap_np).save(swap_path)
            Image.fromarray(m_np).save(mask_path)
            Image.fromarray(aln_np).save(aln_path)

            # Build comparison grid: [Source | Target | 6D Aligned | Swap Pred | Mask | Composite]
            row = np.concatenate([src_u8, tgt_u8, aln_np, swap_np, mask_u8, comp_np], axis=1)
            h, w_col = 32, 128
            header = np.full((h, w_col * 6, 3), 30, dtype=np.uint8)
            lbls = ["SOURCE", "TARGET", "6D ALIGNED", "SWAP PRED", "MASK PRED", "COMPOSITE"]
            for idx, lbl in enumerate(lbls):
                cv2.putText(header, lbl, (idx * w_col + 14, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            full_grid = np.concatenate([header, row], axis=0)
            Image.fromarray(full_grid).save(grid_path)

            out_paths = {
                "composite": str(comp_path),
                "swap": str(swap_path),
                "mask": str(mask_path),
                "aligned_source": str(aln_path),
                "grid": str(grid_path),
                "diagnostic": str(diag_path)
            }

        total_latency_ms = (time.time() - t0) * 1000.0
        diag = {
            "refinement": refinement,
            "correspondence": self.correspondence,
            "geometry_diagnostics": geometry_diagnostics,
            "source_path": str(source_path),
            "target_path": str(target_path),
            "output_paths": out_paths,
            "metrics": {
                "A_cosine_source_composite": cos_A,
                "B_cosine_target_composite": cos_B,
                "C_cosine_source_target": cos_C,
                "D_cosine_source_swap": cos_D,
                "A_minus_C_gain": cos_A - cos_C,
                "D_minus_C_gain": cos_D - cos_C,
                "mask_mean": float(m_np.mean() / 255.0),
                "face_redetected": redetect_ok,
                "landmark_error_px": lm_error
            },
            "performance": {
                "network_latency_ms": net_latency_ms,
                "total_pipeline_latency_ms": total_latency_ms,
                "device": str(self.device),
                "vram_alloc_mb": float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0
            }
        }

        if output_dir is not None:
            with open(diag_path, "w") as f:
                json.dump(diag, f, indent=2)

        return diag


def main():
    parser = argparse.ArgumentParser(description="Phase 6G Standalone Face-Swap Inference")
    parser.add_argument("--source", type=str, required=True, help="Path to source face image")
    parser.add_argument("--target", type=str, required=True, help="Path to target face image")
    parser.add_argument("--checkpoint", type=str, default="ml/checkpoints/stage2/phase6g/best_model.pt", help="Phase 6G checkpoint path")
    parser.add_argument("--arcface", type=str, default="ml/models/weights/ms1mv2_iresnet50.pth", help="ArcFace checkpoint path")
    parser.add_argument("--output-dir", type=str, default="ml/inference/outputs/phase6g", help="Output directory")
    parser.add_argument("--prefix", type=str, default="test_pair", help="Output filename prefix")
    parser.add_argument('--model-variant', choices=['phase6g', 'fullres'], default='phase6g')
    parser.add_argument('--correspondence', choices=['legacy', 'robust', 'reliable', 'visibility'], default='legacy')
    parser.add_argument('--refine', action='store_true', help='Apply identity-guarded opaque face blending')
    args = parser.parse_args()

    engine = Phase6GInferenceEngine(
        checkpoint_path=args.checkpoint,
        arcface_path=args.arcface,
        model_variant=args.model_variant,
        correspondence=args.correspondence,
        refine=args.refine
    )
    result = engine.infer(
        source_path=args.source,
        target_path=args.target,
        output_dir=args.output_dir,
        pair_prefix=args.prefix
    )
    print("\nInference Complete:", flush=True)
    print(f"  Composite Output: {result['output_paths'].get('composite', 'None')}", flush=True)
    print(f"  A-C Identity Gain: {result['metrics']['A_minus_C_gain']:+.4f}", flush=True)
    print(f"  Network Latency:   {result['performance']['network_latency_ms']:.1f} ms", flush=True)
    print(f"  Total Latency:     {result['performance']['total_pipeline_latency_ms']:.1f} ms", flush=True)


if __name__ == "__main__":
    main()
