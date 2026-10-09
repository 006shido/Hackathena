# Quality fine-tuning pilot

Completed 2026-10-05: 300 gradient steps from the full-dataset step-41000 selected checkpoint, legacy correspondence, 1,213 target images processed, 13 preprocessing skips, approximately 180 seconds of training. This is a completed bounded pilot, not another full epoch. The status field `completed: false` refers to full-epoch coverage; the log confirms the configured step limit was reached normally.

Changes: self-reconstruction sampling increased from 20% to 40%; reconstruction weight increased from 5 to 12; exact self-pair horizontal/vertical edge supervision added with weight 3; raw face-core opacity penalty added with weight 3; low-frequency structure weight increased from 5 to 7. Cross-ID identity losses and margin remain enabled. Cross-ID faces have no invented pixel ground truth, so edge matching is restricted to genuine self pairs. Face-core supervision excludes the feathered perimeter.

## Same 300-pair validation comparison

Identity-disjoint validation split, seed 20261005, matching the prior evaluation pairs. FINAL in the comparison images means the pilot's step-300 checkpoint; SELECTED means the original full-training checkpoint.

| Metric | Selected original | Quality pilot |
|---|---:|---:|
| Source cosine | 0.67085 | 0.67287 |
| Target cosine | 0.09019 | 0.09101 |
| Source identity wins | 99.00% | 98.67% |
| Landmark error | 2.21995 px | 2.10608 px |
| Composite sharpness | 276.59 | 279.86 |
| Raw swap sharpness | 372.69 | 378.10 |
| Face redetection | 99.67% | 99.00% |

Landmark error improves approximately 5.1%; composite sharpness improves approximately 1.2%. Source identity cosine improves slightly, but source identity wins and face redetection decline. Visual review confirms significant profile distortion and color/feature artifacts remain. Outcome: partial geometric improvement; not approved for production promotion. No existing checkpoints were overwritten and live-call behavior was not changed.

Pilot checkpoint: `quality_pilot/latest_model.pt`. Metrics: `quality_evaluation/results.json`. Comparison images: `quality_evaluation/comparison_01.png` through `comparison_04.png`.

Reproduction from the repository root using the working Python runtime and project site-packages:

```powershell
python ml/experiments/phase6g_3_full_dataset/train.py --quality-tuning --initialization ml/experiments/phase6g_3_full_dataset/run/best_model.pt --output NEW_OUTPUT_DIRECTORY --max-steps 300 --val-interval 100
python ml/experiments/phase6g_3_full_dataset/evaluate.py --candidate ml/experiments/phase6g_3_full_dataset/quality_pilot/latest_model.pt --output NEW_EVALUATION_DIRECTORY
```

Remaining model-quality work should address unreliable cross-pose source correspondence and provide stronger geometric supervision; larger training runs alone do not establish that these artifacts are fixed. Video temporal stability and live-call throughput still need evaluation after image quality passes.
