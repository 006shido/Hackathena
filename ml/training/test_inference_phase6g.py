"""
Phase 6G: Test script for standalone local inference on 3 genuine image pairs.
"""

import sys
import random
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ml.data.dataset import CelebAPairedDataset
from ml.inference.infer_phase6g import Phase6GInferenceEngine

val_ds = CelebAPairedDataset(dataset_root="ml/data/celeba", split="val", seed=42)
identities = list(val_ds.split_identity_to_images.keys())
rng = random.Random(999)

engine = Phase6GInferenceEngine(
    checkpoint_path="ml/checkpoints/stage2/phase6g/best_model.pt",
    arcface_path="ml/models/weights/ms1mv2_iresnet50.pth"
)

out_dir = Path("ml/inference/outputs/phase6g")
out_dir.mkdir(parents=True, exist_ok=True)

print("Starting inference on 3 genuine validation pairs...", flush=True)

for i in range(3):
    id_s, id_t = rng.sample(identities, 2)
    p_s = rng.choice(val_ds.split_identity_to_images[id_s])
    p_t = rng.choice(val_ds.split_identity_to_images[id_t])
    prefix = f"test_pair_{i+1}"
    print(f"\n--- Testing Pair {i+1}: {p_s.name} -> {p_t.name} ---", flush=True)
    res = engine.infer(str(p_s), str(p_t), output_dir=str(out_dir), pair_prefix=prefix)
    m = res["metrics"]
    p = res["performance"]
    print(f"  Source image:  {p_s.name}", flush=True)
    print(f"  Target image:  {p_t.name}", flush=True)
    print(f"  Composite Out: {res['output_paths']['composite']}", flush=True)
    print(f"  A (src-comp):  {m['A_cosine_source_composite']:.4f}", flush=True)
    print(f"  C (src-tgt):   {m['C_cosine_source_target']:.4f}", flush=True)
    print(f"  A-C Gain:      {m['A_minus_C_gain']:+.4f}", flush=True)
    print(f"  D-C Gain:      {m['D_minus_C_gain']:+.4f}", flush=True)
    print(f"  Mask Mean:     {m['mask_mean']:.3f}", flush=True)
    print(f"  Face Detect:   {m['face_redetected']}", flush=True)
    print(f"  Lm Error:      {m['landmark_error_px']:.2f} px" if m['landmark_error_px'] is not None else "  Lm Error: N/A", flush=True)
    print(f"  Net Latency:   {p['network_latency_ms']:.1f} ms", flush=True)
    print(f"  Total Latency: {p['total_pipeline_latency_ms']:.1f} ms", flush=True)

print("\nStandalone inference successfully completed on all 3 genuine pairs!", flush=True)
