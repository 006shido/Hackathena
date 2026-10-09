# ML runtime and research artifacts

The current launcher is `scripts/Start-ResearchPreview.ps1 -EnableMediaDetection -EnableWebRTCVideo`.

## Required by the current platform

- `api/`: inference, neural video/WebRTC sessions, received-media detection.
- `inference/`: neural engines, frame processing, refinement, and shared face correspondence.
- `inference/face_correspondence.py`: unchanged helpers extracted from Phase 6D/6E. Training and diagnostics now share this implementation. Legacy Phase 6D/6E imports remain compatible.
- `models/`: shared architectures, identity extraction, modules, and pretrained weights. Earlier-phase names do not imply unused code.
- `training/face_preprocessing.py` and `training/robust_correspondence.py`: shared preprocessing/correspondence still used during inference despite their directory name.
- `experiments/phase6g_1_fullres_skip/model_phase6g1.py`: architecture of the currently selected full-resolution model.
- `experiments/phase6g_3_full_dataset/run/best_model.pt`: active Phase 6G checkpoint selected by the launcher.
- `experiments/reference_backend/`: optional pretrained video backend and its assets, selected with `-VideoBackend research-reference`; the default live backend uses the Phase 6G checkpoint.
- `detection/`: received-face/audio detector implementations and weights.
- `runtime_webrtc/`: installed WebRTC runtime used by the launcher.
- Retained CelebA images: UI test pairs and startup warmup inputs.
- The frontend's face-landmarker asset is also required.

Inference no longer imports the old Phase 6D diagnostic or Phase 6E training script merely to obtain geometry helpers.

## Historical research retained

Training scripts, diagnostic scripts, experiment reports, metrics, and comparison images remain available. Baseline Stage 1/Stage 2 and pilot checkpoints still have training/audit references. The default standalone Phase 6G checkpoint remains available for tools that do not use the preview launcher.

## Removed artifacts (2026-10-09)

No runtime, training resume, evaluation, or diagnostic code loaded these nine old outputs:

- `checkpoints/stage2/phase6c_diagnostic/phase6c_diagnostic_model.pt`
- `checkpoints/stage2/phase6e_prototype/step_000.pt`, `step_025.pt`, `step_050.pt`, `step_075.pt`, `step_100.pt`
- `checkpoints/stage2/phase6f_reconstruction/condition_A/model_step300.pt`
- `checkpoints/stage2/phase6f_reconstruction/condition_B/model_step300.pt`
- `checkpoints/stage2/phase6f_reconstruction/condition_C/model_step300.pt`

These generated checkpoints freed 2.09 GiB. Their reports and sample grids were retained; recovering those exact historical weights requires a backup or rerunning the corresponding experiment with the original dataset. File sizes and hashes were recorded in ignored `.run-logs/production-cleanup/removed-historical-checkpoints.json`.

The current checkpoint, scoring logic, architecture, and geometry algorithms were not changed. AST comparison verified the four extracted helper bodies were identical. Fresh-service inference/detection checks should be used whenever runtime code is reorganized.
