# Local research preview

From the repository root in PowerShell:

```powershell
./ml/scripts/Start-ResearchPreview.ps1
```

Open http://127.0.0.1:5173 and use normal tester and user accounts in separate
browser profiles. This starts the normal app, authenticated server, and warmed
trained Phase 6G video backend. Select a source face in the tester's Face panel.
Ctrl+C stops the processes started by this launcher. Existing services are
preserved; occupied ports cause startup to fail.

The launcher verifies model hashes, binds all services to localhost, and records
health and logs under `ml/experiments/reference_backend/local_preview/`.
It requires existing server build, client dependencies, CUDA Python packages,
and downloaded model assets. It does not install dependencies or start training.
Use `-PythonExe <path>` to select another compatible Python runtime,
`-CheckOnly` to verify required files, or `-SmokeSeconds 2` for a bounded startup
test. Rebuild server/dist after changing server code.

For a two-device VS Code forwarded demo, use `-ForwardedDemo -AppPort 5002
-MlPort 8002`. This builds an isolated production frontend, serves it with the
API/signaling on 5002, and generates unique demo credentials. Forward only the
application port using HTTPS. See EVENING_DEMO.md for complete instructions,
TURN configuration and the distinction between this local GPU demo and Render
deployment. The default local development mode is unchanged.

The default `-VideoBackend trained-model` uses the selected Phase 6G checkpoint
for live frames as well as photo inference. `-VideoBackend research-reference`
selects the separate pretrained InsightFace backend. Health reports the selected
backend. Both HTTP and optional WebRTC transport use that selection.

Reference-backend tracking is optional (`-EnableTracking`). It improves detection and reduces
fallback transitions on the two recorded clips, but its motion-compensated
residual results are mixed. Default tracking stays off.

The models include pretrained identity weights restricted to noncommercial
research; see MODEL_NOTICES.md. Missing face detection returns camera video;
hard poses, occlusion, facial detail, physical-device performance and cross-network
calls still require evaluation. Reported modification flags are not independent
deepfake classifier results. Voice presets use DSP transformations.
