# Full-epoch review and fresh evaluation

Date: 2026-10-05. Classification: identity transfer improved; visual and geometry tuning required before live-call integration.

The epoch completed at step 42,449 after traversing 172,111 training targets. 2,315 attempted pairs were skipped during preprocessing. No further training or production integration was performed during this review.

## Fresh evaluation

300 cross-identity pairs were sampled from the held-out identity split using seed 20261005. This seed differs from the training-time validation seed 137. It is a fresh validation sample, not a separately reserved test split. Full pair filenames and measurements are in `evaluation/results.json`.

| Metric | Phase 6G.2 baseline | Selected epoch checkpoint (41,000) | Final checkpoint (42,449) |
|---|---:|---:|---:|
| Source similarity | 0.4827 | 0.6709 | 0.6692 |
| Target similarity | 0.1376 | 0.0902 | 0.0927 |
| Source identity wins | 95.33% | 99.00% | 99.00% |
| Face redetection | 97.33% | 99.67% | 98.67% |
| Landmark error | 1.53 px | 2.22 px | 2.19 px |
| Composite sharpness | 293.4 | 276.6 | 281.8 |
| Raw swap sharpness | 354.9 | 372.7 | 370.0 |
| Model forward median | 14.88 ms | 15.74 ms | 14.28 ms |
| Model forward p95 | 16.99 ms | 17.22 ms | 15.99 ms |

Timing used five warm-up passes and 30 synchronized CUDA forwards, batch one. Timings include the model's ArcFace source embedding and device transfer in the evaluation helper, but exclude face detection, geometric warping, image encoding, network transport and WebRTC. They do not establish end-to-end real-time frame rate.

## Visual review

Reviewed the final training-time six-pair image and fresh comparison grids 01 and 03 (12 fresh examples). Four grids contain 24 examples in total, with SOURCE, TARGET, ALIGNED, BASELINE, SELECTED, FINAL columns.

The full-epoch models substantially reduce the baseline's false-color artifacts and strengthen source features. However, mouth/teeth ghosting, unnatural contours, inconsistent skin color and incorrect geometry remain visible. Severe source-target pose mismatches cause tearing already visible in the deterministic aligned source; additional epochs alone cannot remove a defective correspondence signal. Some outputs retain target hair, glasses or expression incompatibly with transferred source features.

The selected checkpoint is retained as the current offline candidate because it slightly improves source similarity and face redetection relative to final. The final checkpoint is slightly sharper and has marginally lower landmark error. Neither passes a photorealism or live-video stability criterion.

Training-time checkpoint selection allowed a landmark regression of up to 0.5 px on its fixed validation sample. On this fresh sample the selected regression is about 0.69 px, exceeding that tolerance. This is a concrete generalization limitation despite the 99% identity win rate. Cosine wins are not probabilities of authenticity or detection accuracy.

## Next concrete work

1. Address faulty correspondence for large pose differences: confidence/occlusion masks and pose-based rejection, verified by independent redetection and regional landmark metrics.
2. Evaluate color consistency and mouth/eye boundary treatment, using real self-reconstruction supervision and preserving source identity; compare candidate composites on the fixed grids before another long run.
3. Test temporal stability and the complete preprocessing/inference pipeline on actual video before connecting a new stream to WebRTC.
4. Independent receiver-side video anomaly detection and actual audio-detector validation remain separate work. The existing face alert uses tester telemetry.

Artifacts: `evaluation/results.json`, `evaluation/comparison_01.png` through `comparison_04.png`, `evaluation.log`, and `evaluate.py`. Prior checkpoints, dataset and live client/server code were preserved.
