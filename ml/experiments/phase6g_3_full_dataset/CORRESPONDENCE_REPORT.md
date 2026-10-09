# Orientation-aware correspondence and controlled adaptation

Implemented and evaluated 2026-10-05. Status: useful geometry reliability checks; insufficient overall improvement for production promotion.

## Changes and reasons

`ml/training/robust_correspondence.py` implements signed-area foldover rejection, singular-value stretch bounds, exclusion of unstable iris-interior constraints, and confidence zero for invalid triangles. The earlier confidence used absolute triangle areas and a minimum confidence of 0.1, allowing inverted geometry to influence source skips. Rejected regions use a global similarity mapping instead of folded local triangles. A tested optional transition-smoothing variant is disabled because it worsened measured alignment.

Training accepts `--correspondence robust` and explicit `--initialization`; resume checks the correspondence mode to prevent accidental pipeline changes. A 300-step pilot initialized from the full-epoch selected checkpoint, using new streaming training pairs, bfloat16 generator/discriminator, fp32 frozen ArcFace with identity gradients, and the existing reconstruction/identity objectives. Outputs are in `robust_pilot/`; no protected checkpoint was overwritten.

Standalone inference now accepts explicit `--model-variant fullres` and `--correspondence robust` options, making the pilot concretely usable for offline diagnosis. Default Phase 6G inference remains the legacy pipeline; the FastAPI app and live client/server were not switched.

## Evidence

Identity-map and inverted-mesh tests pass on genuine CelebA face landmarks. Same-image mapping preserves pixels; an inverted mesh reduces geometric confidence. Standalone full-resolution robust inference successfully generated an image and diagnostic JSON for the known preset pair. Its first cold-run latency was about 16.7 seconds, including initialization-related overhead; this is not a warmed video throughput measurement.

Initial 100-pair input-only diagnostic improved warped-source cosine from 0.6156 to 0.6250 but increased output landmark error from 2.13 to 2.39 px. Thus the new input could not be enabled without adaptation.

The pilot's own 100-pair seed-137 validation reduced robust-input landmark error from 2.39 to 2.21 px and maintained 100% source-over-target wins. On 300 fresh seed-20261005 pairs:

| Metric | Original selected model + legacy warp | Pilot final weights + robust warp |
|---|---:|---:|
| Source cosine | 0.6709 | 0.6493 |
| Target cosine | 0.0902 | 0.0897 |
| Source wins | 99.00% | 98.67% |
| Warped-source cosine | 0.6262 | 0.6322 |
| Redetection | 99.67% | 98.33% |
| Landmark error | 2.220 px | 2.179 px |

This is about 1.8% better landmark error, with lower source similarity and redetection. Visual review still shows malformed mouth details and pose artifacts. It is a tradeoff, not a complete fix.

The smoothing variant scored source cosine 0.6402 and landmark error 2.356 px on the same 300 pairs; it was rejected and retained only as an explicit disabled experiment. Its measurements are saved as `pilot_evaluation/smoothed_results.json`.

## Decision

Keep the original full-epoch selected model as the main offline candidate. Preserve the robust pilot for diagnosis; do not promote it to live calls or start another full epoch using this configuration. Geometry confidence checks address a real failure mode, but this experiment does not establish accurate photorealistic swapping. A future improvement should constrain expression and occluded-region reconstruction during training and be evaluated per anatomical region, rather than assuming lower aggregate landmark error implies a convincing video.

## Reproduce

Use bundled Python with `.venv/Lib/site-packages` on PYTHONPATH from the repository root. Run `test_correspondence.py` for reliability tests, or `evaluate_correspondence.py --pilot --pairs 300` for the comparison. `infer_phase6g.py --model-variant fullres --correspondence robust --checkpoint ml/experiments/phase6g_3_full_dataset/robust_pilot/latest_model.pt --source ... --target ...` runs the explicit pilot pipeline.
