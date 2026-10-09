# Pose curriculum and reliability-mask pilot

Completed 2026-10-05. Status: evaluated experiment; not promoted.

Implemented a bounded three-stage pose curriculum in the resumable training entry point. For a 300-step pilot, steps 0-99 admit five-point yaw/pitch ratio disparity <=0.15 and unreliable pixel fraction <=10%; steps 100-199 admit disparity <=0.35 and unreliable fraction <=20%; steps 200-299 remove those sampling limits. These are dimensionless landmark-based pose proxies, not angles or ground-truth head rotations. All stages mask invalid geometry. The existing 20% self-pair proposal rate stays configured; rejection of cross-ID proposals can increase the realized self-pair fraction in early stages.

Reliable correspondence retains the trained legacy warp coordinates in valid regions. Signed-area and singular-value checks identify folded or severely stretched triangles. Invalid source pixels become neutral normalized RGB before the encoder, and their confidence remains zero at feature gates. The same preprocessing is exposed in evaluation and the offline inference CLI via `--correspondence reliable`. Existing production defaults remain legacy.

The pilot starts from the original full-dataset step-41000 selected checkpoint, not from the prior quality pilot. This isolates the curriculum/reliability experiment from stronger reconstruction changes. It completed 300 steps, 1,933 target attempts, 706 curriculum rejections, 27 preprocessing skips, and 1,200 accepted training examples in approximately 192 seconds. Target-attempt counts include curriculum rejections. `completed: false` in status means a full epoch was not covered; the configured pilot step limit was reached normally.

## Fresh same-pair comparison

300 pairs, seed 20261005, validation identities disjoint from training. SELECTED means original checkpoint with legacy inputs; FINAL means step-300 curriculum checkpoint with reliable inputs. This measures the combined model/preprocessing change; it does not isolate curriculum causality.

| Metric | Original | Curriculum + masking |
|---|---:|---:|
| Source cosine | 0.67085 | 0.63631 |
| Target cosine | 0.09019 | 0.08791 |
| Source identity wins | 99.00% | 98.67% |
| Composite sharpness | 276.59 | 295.71 |
| Raw swap sharpness | 372.69 | 426.49 |
| Landmark error | 2.21995 px | 2.13995 px |
| Face redetection | 99.67% | 98.33% |

Sharpness improves 6.9% and landmark error decreases 3.6%, while source similarity falls 5.1% and face redetection declines. Visual inspection still shows distorted profiles, mouth artifacts, and missing source detail caused by masked regions. This is not an overall quality fix. Do not replace the original selected checkpoint or extend to full training on this evidence.

On the separate 100-pair training-time validation set, reliable-input source similarity was 0.63672 before adaptation and 0.63920 at step 300; this suggests the input change already removes identity information and the short adaptation does not recover it. Five-point sampling alone does not solve expression/occlusion correspondence.

Tests: identity mapping preserved; inverted mesh suppressed; invalid input pixels and confidence remain exactly zero. Training and fresh evaluation completed with finite losses. No existing checkpoints, dataset images, or live-call behavior were overwritten.

Files: `curriculum_pilot/latest_model.pt`, `curriculum_evaluation/results.json`, `curriculum_evaluation/comparison_01.png` through `comparison_04.png`, and `curriculum_pilot.log`.

Reproduction from the repository root with the working Python runtime:

```powershell
python ml/experiments/phase6g_3_full_dataset/train.py --pose-curriculum --correspondence reliable --initialization ml/experiments/phase6g_3_full_dataset/run/best_model.pt --output NEW_PILOT_DIRECTORY --max-steps 300 --val-interval 100
python ml/experiments/phase6g_3_full_dataset/evaluate.py --candidate ml/experiments/phase6g_3_full_dataset/curriculum_pilot/latest_model.pt --candidate-correspondence reliable --output NEW_EVALUATION_DIRECTORY
```

Remaining work requires source-detail-preserving occlusion handling and better correspondence supervision, rather than assuming that longer training or harder zero masking solves the artifacts. Temporal video stability and throughput remain unverified.
