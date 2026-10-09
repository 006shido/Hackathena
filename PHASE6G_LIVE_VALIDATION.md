# Trained Phase 6G live video validation

Validated on 2026-10-09, Windows, NVIDIA RTX 5060 Laptop GPU.

`Start-ResearchPreview.ps1` now defaults to `trained-model`. Both HTTP frame sessions and WebRTC video sessions select `NeuralFrameSession`, using the existing full-resolution Phase 6G model. The pretrained reference backend remains an explicit launcher option, not an automatic replacement for the trained generator.

Checkpoint SHA-256: `156e922187b56c3d4fcddc8bec48bc5fd85813f78739a6801b09dcb4e3d5a4d9`.

The pipeline caches source identity, detects and aligns each target face, smooths small landmark movements, runs the trained generator, and projects its masked output back into the camera frame. Missing face detection returns the original frame. Shared FP32 settings preserve the existing photo metrics across backend selection. Training weights and architecture were not changed.

## Checks passed

- Real HTTP frames: dimensions, modified-face output, owner isolation, stale timestamp rejection, session teardown; measured processing 67.17, 71.42 and 74.19 ms on retained test images.
- All three original Express photo integrations: identity gains 1.0636, 0.7785 and 0.6639, plus input validation, unavailable-service handling and concurrency rejection.
- Browser peer acceptance: 12 checks covering received video/audio, late join, source switching, screen-track replacement and stopping effects while preserving original tracks. Recorded clip preview measured 12.40 FPS; receiving peer decoded 127 frames.
- Browser WebRTC adapter: trained frames reported modified, source switching preserved the output track, and original camera fixture remained live after stop. Measured about 6.86–7.82 FPS.
- Simulated WebRTC offer failure: the browser switched to HTTP transport using the same trained backend, continued modified frames at about 12.28–12.51 FPS, preserved source switching and kept the original camera fixture live.
- Repository lint, frontend/backend type checks, all 11 automated test suites and frontend/backend production builds passed. Network fixture tests require permission to open localhost sockets in a restricted execution environment.

Browser fixtures use prerecorded video and generated audio. They verify transport and lifecycle, not physical webcams, microphones or cross-network deployment. The trained face crop is 128 pixels; facial detail, difficult poses, occlusion and temporal consistency still require wider evaluation. These checks do not establish production-quality identity transfer for every person. Received-media detectors remain experimental and are not calibrated for reliable live-call decisions.

## Run

```powershell
./ml/scripts/Start-ResearchPreview.ps1 -EnableMediaDetection -EnableWebRTCVideo
```

Open the normal app at http://127.0.0.1:5173/ in separate browser profiles, join the same room as tester/user, and select a permitted source face. ML health should report `video_backend: trained-model`.
