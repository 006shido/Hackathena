# Blending fix evaluation

Updated 2026-10-05. The inference CLI now supports `--refine`: independently check generated face geometry, apply bounded LAB lighting correction, keep the face interior opaque, and feather its perimeter. Additional landmark warping is disabled by default because it reduced source identity. An ArcFace guard rejects changes losing more than 0.02 source cosine. Rejected changes retain the original float-model composite in the inference engine.

On the same 300 validation pairs and selected step-41000 checkpoint, 124 refinements were accepted:

| Metric | Original | Refined |
|---|---:|---:|
| Source cosine | 0.6709 | 0.6683 |
| Source identity wins | 99% | 99% |
| Landmark error | 2.220 px | 2.199 px |
| Sharpness | 276.6 | 286.0 |
| Face redetection | 99.67% | 99.33% |

Sharpness increased 3.4%; geometry changed only slightly. Visual inspection still shows distorted faces and residual ghosting generated inside the model. This is a limited compositing improvement, not a complete face-quality fix or a justification for live deployment. Redetection slightly declined. Existing checkpoints remain unchanged.

The 300-pair evaluation and a complete CLI inference run passed. The CLI run correctly rejected an identity-regressing refinement and retained the original output. Evaluation fallbacks are rebuilt from uint8 images and can differ slightly from original float composites.

Usage from repository root with the configured Python runtime:

```powershell
python ml/inference/infer_phase6g.py --source SOURCE.jpg --target TARGET.jpg --checkpoint ml/experiments/phase6g_3_full_dataset/run/best_model.pt --model-variant fullres --refine --output-dir ml/inference/outputs/refined
```

Results: `refinement_evaluation/results.json`; visual comparisons: `refinement_evaluation/comparison_01.png` through `comparison_04.png`. The production service is not switched to this optional path. Temporal stability and live-call throughput remain unverified.
