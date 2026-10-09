# Run the research platform

This release includes the frontend, backend, Phase 6G neural inference/test screen, live face-swap backend, and experimental received-face/audio anomaly detectors. Model files are separate GitHub Release assets, downloaded and checksum-verified by setup.

Tested platform: Windows, PowerShell, Python 3.12, Node.js 24, an NVIDIA GPU and a driver supporting CUDA 12.8. The full stack requires CUDA; the existing Dockerfile is for the web application only. Other platforms have not been validated.

Read [model terms](MODEL_NOTICES.md), then from the repository root:

```powershell
./ml/scripts/Setup-Research.ps1 -AcceptResearchTerms
./ml/scripts/Start-ResearchPreview.ps1 -EnableMediaDetection -EnableWebRTCVideo
```

Open http://127.0.0.1:5173/ for the application and http://127.0.0.1:5173/?test=phase6g for the neural test panel. Use the local Regular User Demo and Tester Attack Simulator in separate browser profiles. Frontend runs on 5173, backend on 5001, and private ML on 8001. Ctrl+C in the launcher stops its processes.

Upload two permitted face photos in the neural test panel. The three CelebA preset pairs require these separately obtained images in `ml/data/celeba/img_align_celeba/`: 004831.jpg, 004865.jpg, 004931.jpg, 004842.jpg, 004893.jpg, 004925.jpg. Exact golden-metric integration tests require those original inputs. There is no need to download the full dataset for ordinary inference.

For a live call, log in as tester/user in separate browser profiles, join the same room, permit camera/microphone access, and select a source face in the tester's face panel. Independent face/audio analysis is displayed to the receiver when enabled. Scores are experimental and do not prove authenticity.

Verify models and application checks:

```powershell
./.venv/Scripts/python.exe ml/scripts/download_models.py --verify-only
npm run lint
npm run typecheck
npm test
npm run build
# With the stack running and original CelebA preset fixtures installed:
npm run test:ml --prefix server
```

Do not expose the private ML port. For remote-device research demos, see `ml/scripts/EVENING_DEMO.md`; HTTPS and a suitable TURN relay are needed. Before internet deployment, review the production gaps in `PRODUCTION_CLEANUP_REPORT.md`, configure unique signing secrets/passwords, and replace demo account/session management as appropriate.

The repository excludes installed dependencies, model weights, datasets, secrets, and generated training output. A release weighs about 1.7 GiB of runtime models rather than the 30 GiB development workspace. The release includes no training optimizer/resume-only checkpoints beyond the existing selected Phase 6G file.
