# Independent detection work

Face swapping and detecting manipulated media are separate tasks. The app's
existing tester status flags and swap identity guard are not deepfake detectors.

This directory contains reproducible inference adapters for independently
trained public models, not a newly trained or validated production detector:

- DeepfakeBench Xception, official v1.0.1 checkpoint trained on FaceForensics++.
  Source code is CC-BY-NC-4.0 with upstream third-party notices.
- NAVER AASIST, official ASVspoof2019 LA pretrained checkpoint; MIT source license.

Exact upstream commits, licenses, notices and asset SHA-256 checksums are under
assets/. The Xception adapter removes only the training registry import and
decorator; architecture and strict checkpoint loading are retained. PyTorch
weights_only loading avoids executing checkpoint pickle objects.

Face input is RGB 256x256 normalized with mean/std .5. Current MediaPipe square
crops are an adaptation from upstream face preprocessing and require domain
evaluation. Audio input is mono 16 kHz, 64600 samples; resampling uses an
anti-aliasing polyphase filter. Short speech is repeated as in upstream code.
Face class 1 is fake; audio class 0 is spoof. Scores are uncalibrated model
outputs, not probabilities of truth, and are never sourced from tester flags.
Missing/small/multiple faces and silent audio yield no score.

## Run the evaluations

Use the existing compatible Python runtime and .venv package path:

```powershell
$env:PYTHONPATH = "$PWD/.venv/Lib/site-packages"
& "$env:USERPROFILE/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe" -m ml.detection.evaluate_local
& "$env:USERPROFILE/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe" -m ml.detection.evaluate_face_holdout
```

The local pilot uses 12 previously inspected original/swapped photo pairs and
one natural speech recording with DSP effects. It is not a held-out audio
deepfake benchmark. At an uncalibrated .5 cutoff, Xception made two false alarms
and missed three swaps. AASIST assigned the clean recorded speech a spoof score
of .99896. Neither result justifies an accurate fake/real badge. These models
are intentionally not enabled in the live application.

The larger photo evaluation partitions source and target identities together
into calibration/test folds and excludes cross-fold pairs. It applies matched
JPEG85 encoding and the same original-derived crop to each real/swapped pair.
Calibration thresholds never use test scores. This tests only this swapper and
photo domain, not unknown manipulations or real WebRTC video.

Before live promotion: fix preprocessing/domain failures, evaluate labeled
natural and synthesized speech from held-out speakers and manipulation methods,
measure false alarms and missed detections, evaluate video compression and
temporal aggregation, then test the actual received media on two devices.
The CelebA photo dataset alone cannot train or validate audio detection.

## Results from the larger checks

The original face detector's identity-disjoint comparison had 71 calibration
pairs and 81 test pairs. A threshold above every calibration negative produced
zero observed false positives but detected only 31/81 swaps (AUC .89514).

A separate domain-adaptation experiment used 500 training, 150 calibration and
150 test pairs, with source/target identity partitions disjoint and every prior
photo-pool identity excluded. A trained linear head on frozen Xception features
detected only 21/150 test swaps at zero observed false positives (AUC .8684).
It is not promoted. Actual network fine-tuning is a separate experiment under
swap_domain_finetune; status/results are written there. Its test reuses the
linear experiment's holdout for comparison, requiring fresh confirmation later.

The official ASVspoof researchers' published evaluation showcase supplied 148
unique clips: 18 natural and 130 synthetic/converted. At the uncalibrated .5
cutoff, AASIST made zero observed false alarms and detected 104/130 spoofs,
with AUC .95897. This curated showcase is not the full/random benchmark;
shared speakers/texts and the small natural set limit conclusions. The separate
false alarm on the natural VOiCES speech recording remains unresolved. Neither
these examples nor the app's DSP effects are used to train AASIST.

Downloaded weight/media and generated tensor files are excluded from Git.
Licenses, pinned metadata, scripts, manifests and reports remain reviewable.
Deploying source alone will require provisioning the corresponding assets.

## Evening demo update — 2026-10-06

Independent receiver-media research analysis is now integrated: received video JPEG samples go to the Xception swap-domain robust candidate; received mono PCM segments go to NAVER AASIST. JWT-authenticated user and tester roles are accepted, sender effect flags are excluded from model input, GPU work shares the swap inference lock, busy/unavailable/silent samples produce no fabricated score, and client cleanup does not stop call tracks. Enable with `-EnableMediaDetection` in `ml/scripts/Start-ResearchPreview.ps1`. Scores are explicitly uncalibrated research outputs, not authenticity verdicts.

Robust candidate six-epoch training completed. Reused identity-disjoint photo comparison: AUC 0.9826, 5/150 false positives, 134/150 true positives at threshold 0.7698. Received-video diagnostic: 12/115 natural frames falsely flagged, 67/112 swapped frames detected. Blur radius 0/1/2 false positives: 2/50, 12/50, 2/49. These correlated previously inspected clips and reused photos are not a fresh blind benchmark. Candidate is NOT promoted to reliable live classification.

AASIST official curated examples: 0/18 natural false positives and 104/130 spoof positives; limited curated scope. A natural browser-call speech sample falsely scored 0.999 despite original speech scoring 0.023. Audio boundary trimming did not fix this codec/domain failure. Reliable voice fake/real classification remains unresolved.

Both local demo roles and actual face inference passed HTTP checks; malformed audio rejected, silence returned insufficient with null score, unauthorized detection returned 401. Proxy ownership spoofing test and panel independence/empty-state tests pass. TypeScript and production client build pass. Home demo buttons now use the localhost auth endpoint; external forwarded links route to the password form. Unsupported zero-latency/browser-only/clone-certification marketing claims were corrected.

Dedicated updated demo: http://localhost:5002 ; private ML service: 127.0.0.1:8002. Existing 5001/5173 services preserved. No TURN relay is configured; two-device different-network connectivity and physical camera/mic quality still require testing. Research weights retain their license restrictions; do not present this as production certification.
