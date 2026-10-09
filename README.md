# DeepTrace — Phase 6G research platform

DeepTrace includes a React frontend, Express/Socket.IO backend, GPU ML service, Phase 6G neural face-swap test panel, live face-swap research backend, and experimental face/audio anomaly detection.

## Download

- **Source code:** [GitHub repository](https://github.com/006shido/Hackathena). Select **Code → Download ZIP**, then extract it; or clone using Git:

```powershell
git clone https://github.com/006shido/Hackathena.git
cd Hackathena
```

- **Model weights:** [research-v1.0.0 release](https://github.com/006shido/Hackathena/releases/tag/research-v1.0.0). The setup below downloads all seven required files automatically and verifies their SHA-256 checksums. Model downloads total approximately 1.7 GiB.

If you downloaded the source ZIP, open PowerShell inside the extracted folder containing `package.json` instead of running the clone commands.

## Requirements

The tested full-stack setup uses:

- Windows and PowerShell.
- Python **3.12**, with the Windows `py` launcher.
- Node.js **24** and npm, available on PATH.
- An NVIDIA GPU and a driver supporting **CUDA 12.8**. The release was tested on an RTX 5060 Laptop GPU; other GPU configurations have not been validated.
- Internet access for dependency and model downloads.

The full ML stack requires CUDA. Other operating systems and CPU-only execution are not validated. The Dockerfile covers the web application, not the GPU ML stack.

## Install once

Read [MODEL_NOTICES.md](MODEL_NOTICES.md). These model assets are for **noncommercial research**, with separate upstream terms.

From the repository root:

```powershell
./ml/scripts/Setup-Research.ps1 -AcceptResearchTerms
```

Setup creates the Python environment, installs CUDA PyTorch and runtime dependencies, downloads and verifies models, installs frontend/backend dependencies, builds the application, and checks GPU availability. Keep PowerShell open until it completes.

## Start the complete platform

```powershell
./ml/scripts/Start-ResearchPreview.ps1 -EnableMediaDetection -EnableWebRTCVideo
```

Wait for `Research preview ready`, then open:

| Service | Address |
|---|---|
| Frontend | http://127.0.0.1:5173/ |
| Phase 6G neural test | http://127.0.0.1:5173/?test=phase6g |
| Backend and ML health proxy | http://127.0.0.1:5001/api/ml/health |
| Private GPU ML health | http://127.0.0.1:8001/health |

Keep the launcher running. Press **Ctrl+C** in that terminal to stop the services it started. For later sessions, run the start command again; setup is only needed once unless dependencies or release assets change.

## Test Phase 6G with photos

1. Open the Phase 6G neural test address above.
2. Upload a source face photo and a target face photo you have permission to use.
3. Run inference and inspect the result and measurements.

The source supplies the identity to transfer; the target supplies the destination face/image. Phase 6G image inference uses the trained full-resolution checkpoint. Live video uses the configured research reference backend.

The three CelebA preset pairs are optional. Their images are not included in the repository or release. Obtain them separately under the dataset's terms and place these files in `ml/data/celeba/img_align_celeba/`:

```text
004831.jpg  004865.jpg
004931.jpg  004842.jpg
004893.jpg  004925.jpg
```

You do not need the full training dataset to upload your own photos or run ordinary inference.

## Try live face swapping and anomaly detection

1. Open the frontend in two separate browser profiles, so each has an independent login session.
2. Select **Regular User Demo** in one profile and **Tester Attack Simulator** in the other.
3. Create a room and use the same room code in the other profile.
4. Allow camera/microphone access, check the device preview, and join the call.
5. In the tester profile, open the **AI Face Swap & Attack Simulator** panel and choose a source face using its controls.
6. In the receiving profile, open **Security Monitor & Visualizer** to inspect received-media analysis.

The start flags enable face/audio detection and neural WebRTC processing. Model scores are experimental and are not proof that media is real or fake. Reported tester controls are separate from independent received-media measurements.

For calls between devices or networks, follow [EVENING_DEMO.md](ml/scripts/EVENING_DEMO.md). HTTPS and an appropriate TURN relay may be required. Keep the Python ML service private.

## Verify the installation

```powershell
./.venv/Scripts/python.exe ml/scripts/download_models.py --verify-only
npm run lint
npm run typecheck
npm test
npm run build
```

With the platform running and the six original CelebA preset images installed, run the exact three-pair neural integration suite:

```powershell
npm run test:ml --prefix server
```

See [RELEASE_VALIDATION.md](RELEASE_VALIDATION.md) for the checks performed and their limits.

## Troubleshooting

- **Python or npm not found:** install the required versions and reopen PowerShell; check that `py -3.12 --version`, `node --version`, and `npm --version` work.
- **CUDA unavailable:** check the NVIDIA driver and GPU support. Setup reports a failure rather than silently switching this full stack to CPU.
- **Model missing or checksum mismatch:** run the downloader again. Existing files with mismatched checksums are left untouched; inspect them before replacing them with the matching release assets.
- **Port already in use:** the launcher preserves existing services and stops startup. Use the existing running platform or stop the process you started on ports 5173, 5001, or 8001.
- **Preset photos unavailable:** upload your own permitted photos or install the six separately obtained fixtures listed above.
- **First inference is slow:** model loading and GPU warmup happen during startup; wait for the ready message.

## Research scope and deployment

GitHub hosts the code and downloads; users run this platform on their own compatible computer. Installed environments, datasets, training outputs, credentials, and model weights are excluded from ordinary Git commits. Runtime models are distributed separately through the release.

Review [MODEL_NOTICES.md](MODEL_NOTICES.md) for model restrictions and attribution, [GETTING_STARTED.md](GETTING_STARTED.md) for additional setup notes, and [PRODUCTION_CLEANUP_REPORT.md](PRODUCTION_CLEANUP_REPORT.md) before considering an internet deployment. The demo's account management, persistence, public test endpoints, and experimental detector accuracy still require production review.
