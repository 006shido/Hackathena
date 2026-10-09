# Phase 6G.2: restored identity gradients and source-target margin

Completed 2026-10-05. Classification: improved identity transfer; visual tuning still required.

## Recovered stopping point

Phase 6D established deterministic geometric correspondence. Later local artifacts include Phase 6G.1, whose full-resolution source skip improved sharpness but achieved only 58% source-over-target identity wins on its validation set. This continuation starts from its best checkpoint, not from random weights or the degraded final weights.

## Root cause and correction

`ArcFaceIdentityExtractor.forward` encloses its backbone in `torch.no_grad()`. Consequently the previous identity losses could report meaningful values without supplying gradients to generated pixels. This experiment overrides only its own extractor instance: the backbone remains frozen and in evaluation mode, but generated-image gradients are enabled. An executable gradient assertion verified finite, nonzero image gradients and frozen ArcFace parameters before training.

Added mean hinge loss on both raw swap and composite: `relu(target_cosine - source_cosine + 0.2)`, weighted by 10. Retained existing structure, background, mask and adversarial losses. Learning rate 2e-5, batch four, 300 training pairs, 300 steps, validation every 25 steps on 100 pairs. Training and validation identities are disjoint. Best-checkpoint selection rejects substantial sharpness, landmark and redetection regressions. Step 100 was selected; later steps improved some identity metrics while sacrificing excessive sharpness.

## Results

| Metric | Original validation baseline | Selected checkpoint | Fresh baseline | Fresh candidate |
|---|---:|---:|---:|---:|
| Source cosine | 0.2272 | 0.4697 | 0.2603 | 0.4936 |
| Target cosine | 0.1836 | 0.1259 | 0.1857 | 0.1280 |
| Source identity wins | 58% | 97% | 68% | 97% |
| Composite sharpness | 301.6 | 276.9 | 288.5 | 264.1 |
| Raw swap sharpness | 354.1 | see metrics.json | 359.4 | 357.1 |
| Face redetection | 98% | 100% | 100% | 100% |
| Landmark error | 1.84 px | 1.62 px | 1.83 px | 1.70 px |

Fresh evaluation uses seed 137 with 100 pairs from the held-out identity split; it is a second validation sample, not a separately reserved test split. Pair filenames are recorded in `checkpoints/fresh_evaluation.json`. These are cosine metrics, not probabilities of photorealism or detection accuracy.

Visual inspection of comparison images 01, 02 and 05 confirms changed facial identity but also reveals color mismatch, blended ghosting and occasional background/contour artifacts. The model is not yet ready to claim accurate photorealistic live swapping. Twelve fresh comparison strips are saved with columns SOURCE, TARGET, GEOMETRIC ALIGNMENT, RAW SWAP, COMPOSITE.

## Artifacts and integrity

Selected checkpoint: `checkpoints/best_model.pt`. Final checkpoint: `checkpoints/latest_model.pt`. Full validation history: `checkpoints/metrics.json`. Fresh results: `checkpoints/fresh_evaluation.json`. Training process completed successfully; protected ArcFace, Phase 6G, and Phase 6G.1 checkpoint hashes were rechecked and unchanged. Client, server, dataset and production inference configuration were not edited.

Runtime: existing virtual environment launcher points at an unavailable Microsoft Store Python. Bundled Python 3.12.14 successfully uses `.venv/Lib/site-packages` through PYTHONPATH, with torch 2.7.1+cu128 and RTX 5060 CUDA. Training and evaluation logs record the completed runs.

## Remaining work toward the user's intended call experience

1. Tune compositing/color consistency and border artifacts; assess a larger identity-disjoint sample and video temporal stability. Compare identity against the stronger geometric alignment reference, not just the earlier neural checkpoint.
2. Integrate the selected architecture into an isolated inference option and measure sustained frame throughput before sending generated video through WebRTC. Current production service still loads the older Phase 6G model.
3. Implement and validate independent video anomaly analysis on the receiver's actual incoming frames. Current face alert in SecurityPanel uses peerAttackState.faceSwap telemetry, which is simulation disclosure rather than independent detection.
4. Validate the existing incoming-audio DSP detector using original and transformed speech through actual WebRTC, including benign noise and compression. Face generator training does not train an audio detector or establish its accuracy.

Run from repository root with the bundled Python executable and PYTHONPATH set to `.venv/Lib/site-packages`. `evaluate.py` can regenerate fresh evaluation outputs; `train.py` refuses to overwrite an existing metrics report.
