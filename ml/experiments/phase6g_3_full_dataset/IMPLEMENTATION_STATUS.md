# Face-swap implementation status — 2026-10-06

**Status: quality and call acceptance remain open. No replacement is enabled by default.**

The original full CelebA epoch finished at 42,449 steps. Later bounded runs are
experiments, not another completed full epoch. Dataset and original checkpoints
are preserved. There is no claim of perfect output or a calibrated video detector.

## Own-model experiments

Fresh 300 held-out cross-identity pairs, seed 20261005 unless specified:

| Candidate | Source similarity | Source wins | Re-detection | Landmark error |
|---|---:|---:|---:|---:|
| Original full-epoch selection | 0.67085 | 99.00% | 99.67% | 2.220 px |
| Quality tuning, 300 steps | 0.67287 | 98.67% | 99.00% | 2.106 px |
| Reliable warp curriculum, 300 steps | 0.63632 | 98.67% | 98.33% | 2.140 px |
| Visibility curriculum, 1,000 steps | 0.67020 | 99.67% | 99.00% | 2.155 px |
| Different-photo same-ID supervision, 1,500 steps | 0.63728 | 99.00% | 100.00% | 2.040 px |
| 50% checkpoint interpolation with original | 0.66620 | 98.67% | 99.67% | 2.135 px |

Cross-view supervision uses genuine different photos of the same identity as
reconstruction targets. It improved geometry but reduced source similarity and
sharpness. Visual mouth/profile defects remain. The second fresh 300-pair seed
20261006 also showed a geometry improvement (2.008 vs 2.162 px), but lower source
similarity (0.651 vs 0.681). None of these candidates was promoted.

## Independent research reference

Official InsightFace assets were downloaded into `ml/experiments/reference_backend/assets`
and independently verified against published release SHA-256 digests. The requested
ONNX dependency installation was declined; no packages were installed. Existing
OpenCV provides the independent CPU runner. A restricted, hash-pinned PyTorch
executor supports only this graph, rejects unexpected operators/types/semantics,
and is isolated from the live service.

Fresh 300 pairs, seed 20261006, cached-source-landmarks-v2 protocol:

| Metric | Original full-epoch model | OpenCV research reference |
|---|---:|---:|
| Source similarity | 0.68121 | 0.75768 |
| Target similarity | 0.09059 | 0.14403 |
| Source wins | 100.00% | 99.67% |
| Independent re-detection | 99.67% | 99.67% |
| Landmark error | 2.162 px | 1.647 px |
| Laplacian sharpness | 282.83 | 254.37 |

Reference faces have fewer obvious doubled features and better pose matching in
reviewed grids. Softer output, hard poses and one source-identity failure remain.
Mean metrics do not establish defect-free output. The initial 100-pair screen used
an extra source-crop re-detection; the fresh protocol reuses source landmarks from
successful original-image preprocessing to avoid an unnecessary failure point.

PyTorch fp32 parity against OpenCV on three fixed inputs: maximum absolute error
below 0.00006, mean below 0.000006. Median swap-plus-transfer time was 58.14 ms,
p95 73.27 ms. This excludes detection, source recognition, compositing and network.
CPU swap was approximately 1.1 seconds. GPU real-face and video results are stored
separately under the reference experiment, not silently substituted into CPU results.

The GPU fresh-300 evaluation reproduced CPU source similarity within 0.000001,
with identical source-win and detection rates. Full 640x480 synthetic processing
detected all 60 frames: median 74.84 ms, p95 83.35 ms (about 13 processed frames/s,
excluding encoding and network). Original-model comparison at the same frame size
was median 72.46 ms, p95 79.42 ms. These are upscaled still-image motion probes,
not recordings of speech or occlusion. Results and videos are saved under
`reference_backend/gpu_holdout`, `reference_backend/video640` and this experiment's
`video640` directory.

Official pretrained assets have non-commercial research restrictions; deployment
rights must be established before distributing a reference-backed product.
Sources: https://github.com/deepinsight/insightface and
https://github.com/deepinsight/insightface/releases/expanded_assets/model-zoo.

## Integration implemented and verified

- Per-call source identity caching, in-memory frame processing and inverse affine
  projection of premultiplied face color. Pixels outside the face remain exact.
- Missing-face fallback preserves the camera frame and resets tracking. Finite,
  increasing timestamps are required.
- Opt-in `/video/sessions` service routes, ownership, bounded sessions, expiry,
  serialized inference, image-size limits, JPEG frames and session teardown.
- Express authenticates JWT tester roles and forwards trusted owner identity.
- Browser preview keeps one outstanding frame, rejects stale identity/stop responses,
  closes bitmaps, bounds request timeouts and preserves the original camera track.
- Preset errors are surfaced; failed restoration after screen sharing falls back
  to camera and tells the receiver that face simulation stopped.
- Receiver panel displays video/audio tester-reported changes separately from
  incoming audio analysis. A reported face change is not labelled as independent
  landmark detection; low audio risk is not presented as verified authenticity.

Checks passed: client and server builds; projection, correspondence and missing-face
unit checks; service metadata; real Express proxy authorization/ownership/binary
responses; browser lifecycle simulation; receiver panel rendered-state checks.
Actual temporary FastAPI HTTP test returned three neural frames, correct dimensions,
owner rejection, duplicate timestamp rejection and teardown. The temporary service
was stopped. This is not an end-to-end two-peer WebRTC acceptance test.

An additional temporary HTTP test passed these same checks with the GPU research
backend. Local research preview can be selected explicitly with
`ML_VIDEO_BACKEND=research-reference` plus the two preview enable flags. Selection
is reported in health, session creation and `X-Video-Backend`; the proxy forwards
it. Source recognition is cached once per session. The research backend only
affects opted-in video sessions; still inference keeps its configured trained model.
No research server remains running after the checks.

### Subsequent call-status and real-video checks

Authenticated Socket.IO integration now verifies late-join status snapshots,
tester room membership, role rejection, canonical face/voice modes, stop updates
and peer teardown. The client re-sends its current simulation flags after joining
and clears the remote state on signaling disconnection. Client/server builds pass.

The synthetic probe now compensates known motion before measuring consecutive
RGB changes inside the eroded face region. At three times the original motion,
90/90 frames were detected; output mean MAE was 0.351/255 versus input interpolation
residual 0.262/255. This is evidence about rigid synthetic motion only.

Real OpenCV tracking test videos were downloaded from the official repository;
URLs, frame counts and computed SHA-256 digests are in
`reference_backend/videos/provenance.json`. They are evaluation data, not training.
Source: https://github.com/opencv/opencv_extra/tree/4.x/testdata/cv/tracking.

- David: 770 real frames, 320x240, detection 95.06%, median 67.36 ms,
  p95 70.93 ms. Lighting changes expose a detection reliability gap.
- FaceOcc2: 812 real frames, 320x240, detection 87.93%, median 67.01 ms,
  p95 70.97 ms. Reviewed fully covered frames returned original pixels; sampled
  partial occlusions were visually reviewed. This does not prove all occluders are
  preserved or all output identities are correct.

Videos and contact sheets are in `reference_backend/david_video` and
`reference_backend/occlusion_video`. No real-expression flicker pass is inferred
from raw consecutive frame deltas.

Cold GPU setup could take several seconds in the first frame, exceeding the
browser's two-second frame timeout. The opted-in research backend now loads and
warms during service startup, before readiness. Session creation refuses an
unwarmed reference rather than loading it inside a timed request. A fresh actual
HTTP test verified all three frames below two seconds, including the first;
individual timings are in `reference_backend/warm_http_test/results.json`.
The test service was stopped.

### Browser media and independent identity gates

The recorded-video browser harness uses the production NeuralVideoPreview,
VoiceTransformationPipeline, authenticated Express ML proxy, Socket.IO signaling
and real RTCPeerConnections. The acceptance runner binds only loopback, uses a
fresh temporary JWT secret, and serves fixed dataset fixtures. It does not access
camera/microphone or create normal application accounts. See
`client/tests/neural-call.html`, `client/tests/neural-call.ts` and
`server/tests/run-browser-acceptance.ts`.

The latest guarded browser run passed late join, video reception, non-silent
transformed audio, screen-track replacement and effect restoration, source
switching, original-track preservation and stop signaling. Receiver decoded 112
frames, with 10.68 preview frames/s and nonzero received audio RMS. The source is
a recorded real face clip; audio is generated, so this does not validate natural
speech detection, physical devices, two-computer connectivity or the complete
Call UI. Results/screenshot are in `reference_backend/browser_call`.

Codec-decoded received snapshots were scored with frozen original ms1mv2 ArcFace:

| Received snapshot | Source A | Source B | Original target |
|---|---:|---:|---:|
| Source A active | 0.586 | 0.046 | 0.295 |
| Source B after switch | 0.084 | 0.765 | 0.361 |
| Effects stopped | 0.114 | 0.249 | 0.888 |

The scorer is independent of the reference generator; it is also the scorer used
by the new identity guard. These three snapshots do not establish all-frame
identity fidelity. The verification script is `reference_backend/verify_browser_identity.py`.

Research sessions now cache a source embedding for an independent frame identity
gate. Swaps need source similarity >=0.30 and source-minus-target >=0.05, otherwise
the original RGB frame is returned. These are conservative operational thresholds,
not calibrated identity probabilities. The known held-out failure
186165.jpg -> 070860.jpg was rejected (source 0.261, target 0.684), preserving
the original pixels exactly; a valid control was accepted (source 0.627, target
0.174). An actual HTTP regression verified rejection metadata and exactly the
same JPEG bytes as encoding the original frame with the service settings.

`X-Face-Changed` distinguishes actual generated output from detection-only camera
fallback; the proxy forwards it and the preview telemetry labels camera fallback.
At 640x480, the guarded synthetic test accepted all 60 frames with median 92.77 ms,
p95 102.15 ms. This performance cost is measured, not hidden in swap-only timing.

Screen sharing now suspends face processing, publishes the screen in local and
remote streams, reports face simulation inactive while sharing, and restores the
requested face effect when sharing stops manually or via the browser. Effect
toggles/reset while sharing preserve the screen track. These hook changes build
successfully; the browser harness verifies track replacement and effect restart
at the media level, not the actual display-picker UI.

Overlapping preview startup is guarded by a startup generation. An older cancelled
startup cannot create a duplicate stream or stop the newer one. The lifecycle
regression test covers this race and preservation of the camera.

Low-light retry remains experimental: it recovered only two of 770 David frames
(95.32% vs 95.06%). Most misses occurred on small faces in bright frames, so this
is not promoted as a general detector fix. Detection-only retry never replaces
original synthesis RGB pixels.

Original-model synthetic 178x218 motion probe: 60/60 faces detected, median 47.67 ms,
p95 52.52 ms. Small synthetic images do not establish webcam performance or stability
during speech, expression changes, occlusion or fast head movement.

## Remaining acceptance work

### Opt-in per-call tracking implementation (2026-10-06)

`ML_VIDEO_TRACKING=1`, with the existing explicit research preview flags, now gives
each ReferenceFrameSession its own preprocessor and VIDEO detector. The shared
source IMAGE detector and GPU weights remain shared; tracked target landmarks and
timestamps do not. Incoming browser milliseconds are rounded and strictly
increasing integer timestamps are maintained for MediaPipe. Defaults remain off.

Session deletion, idle expiration and shutdown close tracking resources. Expiration
skips a busy session; frame handling rechecks registry membership after decoding.
Cancelled requests await their inference thread before releasing the GPU lock;
cancelled session construction closes the otherwise orphaned tracking resource.

Actual tracked HTTP frames took 106.05, 73.92 and 75.19 ms in the regression. The
test passed owner isolation, stale timestamp rejection, deletion and exactly
original JPEG output on the known identity-rejected pair. Results are in
`reference_backend/tracked_http_test`. Real detector/session isolation and expiry
tests passed in `test_video_tracking.py`. `test_worker_cancellation.py` passed
outside the sandbox; Windows asyncio setup stalled inside the restricted sandbox.
The stalled test processes were terminated before the approved retry.

The production media browser harness passed against this tracked service: late
join, received video/audio, screen replacement, camera preservation, effect
restoration, source switching and cleanup. It decoded 123 frames at 11.19 preview
frames/s with received audio RMS 0.0498. This harness does not mount the complete
Call UI or use physical devices. Evidence is in `reference_backend/tracked_browser_call`.
Independent-generator received identity scores (same scorer as the guard) were:
source A 0.593 vs target 0.321; source B 0.764 vs target 0.318; camera restored
target 0.894. These are three snapshots, not all-frame fidelity. The scorer now
accepts `--output` to reproduce this check without overwriting older runs.

Opt-in tracking is implemented and these regressions pass. Broader visual stability,
perceptual defects, physical/cross-device acceptance and applicable usage gates
remain open. Test service and fixture servers were stopped afterward.

### Real-video temporal landmark experiment (2026-10-06)

Guarded reference baseline was evaluated over all 812 faceocc2 frames and all 770
David frames. `inspect_video.py` records actual swap/rejection/missing-face
decisions, transition indices, longest fallback runs and accepted identity margins.
The baseline makes 34 and 52 decision transitions respectively; this is an observed
continuity limitation, not a motion-compensated flicker score.

An offline-only `--video-landmarks` option tests the official MediaPipe VIDEO mode
with num_faces=1 smoothing, using an independent detector per sequence. Source
identity is still built from IMAGE detection; thresholds remain A>=0.30 and
A-B>=0.05. Training, still evaluation and service behavior remain unchanged.

| Clip / mode | Detection | Swapped frames | Identity rejects | Decision transitions | Warm median / p95 |
|---|---:|---:|---:|---:|---:|
| faceocc2 IMAGE | 87.93% | 673 / 812 | 41 | 34 | 79.22 / 83.55 ms |
| faceocc2 VIDEO | 92.24% | 699 / 812 | 50 | 21 | 78.55 / 82.50 ms |
| David IMAGE | 95.06% | 680 / 770 | 52 | 52 | 79.53 / 83.37 ms |
| David VIDEO | 95.45% | 689 / 770 | 46 | 38 | 78.51 / 81.39 ms |

Tracking gained 30 swaps and lost 4 on faceocc2, gained 20 and lost 11 on David.
Selected changed decisions are compared original/baseline/tracked by
`compare_video.py`, with exact frame indices preserved. Reviewed faceocc2 additions
include severe head roll and a book beside the cheek; coverage improves there,
while identity-rejected mouth occlusion retains the camera. Long full occlusion
still causes fallback. These selected visuals do not prove every partial occluder
is preserved or all-frame perceptual quality.

Results are in `reference_backend/guarded_{occlusion,david}_video` and
`reference_backend/tracked_{occlusion,david}_video`. The candidate's accepted
minimum source score / source-target margin were 0.403 / 0.0527 (faceocc2),
0.370 / 0.0529 (David). These use the guard scorer, not an additional independent
recognizer. Both inference processes completed normally. Next is per-call tracking
isolation, actual timestamps and video stability/call regression before promotion.

### Recorded speech transport and voice lifecycle (2026-10-06)

`client/tests/speech-audio.html` sends an official PyTorch/VOiCES 3.4-second speech
fixture through the production VoiceTransformationPipeline and real loopback
WebRTC. Fixture provenance and SHA are in `reference_backend/videos/speech-provenance.json`.
The test captures received PCM, rather than inferring audibility from signaling.
The first baseline capture was silent because the receiver media had no playback
element; after attaching playback, both input and received clean audio were nonzero.
The failed baseline report is preserved as `initial-capture-failure.json`.

All presets passed non-silent speech and <1% near-clipped-sample gates. Received
RMS was clean 0.1205, robotic 0.0535, deep resonance 0.3664, synthetic resonance
0.1690. Near-clipped samples (absolute PCM >=0.99) were respectively 0%, 0.00047%,
0.0150%, 0.0822%. These are level/transport checks on one fixture, not intelligibility
scores or evidence that the presets use neural voice cloning or actual pitch shifting.
Received WAVs and `audio-results.json` are in `reference_backend/browser_speech_call`.

Voice startup now stops previous derived tracks and cancels superseded asynchronous
startup. Preset changes disconnect anonymous input branches before rebuilding;
stop clears automatic-resume callbacks. The browser confirms previous output
tracks end and original speech remains live. `server/tests/voice-lifecycle.test.ts`
also verifies branch cleanup and overlapping startup ownership. Client build and
test-entry TypeScript checks pass. Natural speech intelligibility, varied speakers
and calibrated audio anomaly classification remain unverified.

### Full Call UI regression (2026-10-06)

The new `client/tests/call-ui.html` fixture mounts the actual App with separate
tester/user origins and ephemeral authenticated sessions. Media acquisition alone
uses the recorded faceocc2 clip and generated audio. Run the local fixture server
with `npx tsx tests/run-browser-acceptance.ts --full-ui` from server/ while the
opted-in research ML service is running. Normal preview flags remain disabled.

Browser interaction verified both callers join and receive decoded 320x240 video,
gallery uploads and active source switching, combined face/voice reported flags,
screen replacement, face-processing suspension, browser-ended share restoration,
reset preserving an active screen, reset clearing flags, and peer cleanup through
the actual end-call slider. Evidence and scope are in
`reference_backend/full_call_ui/observations.json` and its screenshots.

The full interface exposed and now fixes fabricated 56 FPS/478-landmark standby
claims and WebGL/blending claims while the neural backend runs. Unsupported neural
blend controls are hidden. Measured FPS is rounded; actual detection, fallback and
suspension states are shown. Suspended neural processing no longer inherits stale
browser-engine landmarks. Returning from Voice to Face now reattaches the preview
video, fixing a persistent blank thumbnail caused by an incomplete effect dependency.

Generated audio checks reported flags only here; natural speech/intelligibility and
the heuristic audio detector are still unverified. Screen capture is simulated,
not a native picker. Brief unable-to-play placeholders occurred during stopped-track
replacement and cleared on restoration. These UI observations do not establish
all-frame visual fidelity. Test services and browser tabs were stopped afterward.

1. Fix/reject identity failures and residual visual artifacts before choosing a
   replacement. More full-dataset training alone is not an acceptance criterion.
2. Test real speech, expression, profile, occlusion, lighting and motion clips;
   measure motion-compensated flicker and complete pipeline latency.
3. Complete authenticated tester-to-user calls, source switching, screen sharing,
   audio transforms, reconnects and teardown in real browsers.
4. Promote only a backend that passes quality, runtime and applicable usage gates.

Normal flags remain disabled: `ML_ENABLE_VIDEO_PREVIEW` defaults off and
`VITE_ENABLE_NEURAL_VIDEO_PREVIEW` defaults false. The original service checkpoint
and browser renderer defaults are preserved.

### Motion-compensated residual and normal-app launcher (2026-10-06)

`reference_backend/flow_stability.py` evaluates encoded RGB residual using
bidirectional Farneback flow, consistency and input-photometric gates, and eroded
effect support. Only consecutive swaps accepted in both modes contribute;
fallback transitions are excluded. The known-translation/brightness-pulse unit
check passed. This approximate metric includes flow and codec error and is not a
calibrated perceptual flicker score.

On faceocc2, baseline/tracked output MAE was 3.9769/3.6247 across 655 pairs.
On David it was 3.7625/3.7776 across 569 evaluated pairs. Results are mixed;
tracking remains opt-in. Full records are in each tracked clip's flow_stability.json.

`ml/scripts/Start-ResearchPreview.ps1` now starts the normal app with normal
accounts and the verified research backend on localhost. Its bounded smoke run
became ready with CUDA, research-reference and tracking=false, exited successfully,
and left ports 8001/5001/5173 free. Startup evidence is in
`reference_backend/local_preview/20261006-025931-103/startup.json`.
Instructions and limitations are in `ml/scripts/RESEARCH_PREVIEW.md`.
This establishes local startup, not physical-camera or cross-device acceptance.

### Lossless occluder diagnostic (2026-10-06)

The offline video evaluator now constructs the same per-session VIDEO tracker as
the live service, uses actual frame timestamps, closes it on exit, and supports
`--save-raw-frames` for lossless original/output PNGs. Prior tracking reports used
an older shared detector setup; retain them as historical evidence rather than
assuming they prove current live-session behavior.

A fresh 151-frame faceocc2 run with identity guard and per-session tracking
completed: 131 swapped frames, 20 identity rejections, median 79.06 ms and p95
83.87 ms. On accepted frame 150, a manually annotated book interior (8363 pixels
after 9px erosion) had mean RGB change 0.5782 and 2.786% of pixels changed by more
than 8 levels in at least one channel. These are raw inference differences without
output codec error. The landmark oval compositor therefore does not fully preserve
this occluder, even when the identity gate accepts the face.

Evidence is in `reference_backend/raw_occluder_probe/occluder_frame150.json`,
the lossless panels and heatmap. `inspect_occluder.py` reproduces the measurement;
its polygon is a manual diagnostic, not an automatic occlusion segmentation model.
Occluder-aware composition remains an unresolved quality requirement; do not
weaken identity gates or claim this measurement fixes it.

### XSeg occlusion candidate, not promoted (2026-10-06)

Downloaded the official FaceFusion models-3.1.0 XSeg asset with user approval;
its published SHA256 matches c4d1498b8a03b5fe2a3a5d2ef2a0402ab03bd51edaf5b2d8d5fb764702a97dd3.
Vendor/license metadata is DeepFaceLab/GPL-3.0. Provenance is recorded alongside
the asset. Existing OpenCV can execute it; no packages were installed.

`probe_occlusion_mask.py` applies a predicted visibility mask to two lossless
composites, using IMAGE landmarks for this isolated probe. The annotated book
interior changes fell from MAE 0.5782 / 2.786% >8 levels to exactly zero.
However, clear-frame source similarity fell from 0.5786 to 0.4976 and target
similarity increased from 0.3430 to 0.4381. Both clear outputs pass the current
identity gate, but the candidate retains substantially more target identity.
Book-frame baseline and candidate both fail the independent IMAGE-aligned gate;
this alignment differs from the VIDEO landmarks that accepted the original frame.
Do not equate that mismatch to a live gate regression without same-alignment testing.

CPU mask inference took 154–187 ms on these two cold calls, which is too expensive
to simply add to the current frame loop. Artifacts and identity scores are in
`reference_backend/raw_occluder_probe/xseg_probe.json` and comparison PNGs.
The candidate is not promoted. Further work must evaluate alignment and mask
thresholds against both occluder preservation and source identity, then measure
warm runtime and temporal behavior on broader clips.

### Occlusion mask threshold sweep (2026-10-06)

`probe_occlusion_mask.py --sweep` now tests soft and binary visibility masks at
thresholds 0.1–0.5, keeping the original probe report separate. All ten candidates
preserved the annotated book interior exactly. Soft threshold 0.1 improved clear
source similarity to 0.5259, but still below baseline 0.5786. Binary masks produced
roughly 0.53–0.54 on the clear frame and preserved more source/target separation
than the original soft setting. All book candidates failed the IMAGE-aligned
identity gate; baseline also fails under that alignment as previously recorded.

Six forwards per input establish warm CPU cost: about 156–157 ms median in the
final sweep. Thus initial setup alone does not explain the added latency.
`raw_occluder_probe/xseg_sweep.json` and threshold comparison PNGs preserve results.
These two frames are tuning inputs, not a held-out validation set. No threshold
is promoted: same-live-alignment identity testing, runtime improvement and video
edge/flicker review remain required before adopting occluder-aware composition.

### Same-tracking-alignment occlusion verification (2026-10-06)

Offline selected-frame diagnostics now capture the actual VIDEO five-point
landmarks. The normal HTTP path does not enable this capture. Lossless frame
exports include geometry JSON, and `probe_occlusion_mask.py --sweep --live-geometry`
refuses comparison unless baseline source and target scores reproduce within
1e-4. Candidate images use a separate live_alignment directory so prior IMAGE
alignment artifacts remain distinct.

The fresh 151-frame run again produced 131 swaps and 20 identity rejections.
The same-alignment baseline at book frame150 passes with roughly source 0.53 /
target 0.46. Every occlusion-mask candidate fails, with source about 0.40–0.44 /
target 0.58–0.66, while preserving the annotated book interior exactly. The
rejection therefore persists after removing the earlier alignment mismatch.
Clear-frame candidates still pass but lose source similarity against baseline.
Full precision evidence is in raw_occluder_probe/xseg_live_sweep.json.

Do not lower or bypass the guard to accept these candidates. Book preservation
changes the appearance seen by the whole-crop recognizer; the present experiment
does not establish how much visible facial identity survives. Occlusion-conditioned
identity assessment would need independent calibration before changing acceptance.
The current candidate remains unpromoted and live defaults remain unchanged.

### Occlusion boundary expansion experiment (2026-10-06)

The isolated probe now tests binary visibility expansion at radii 0/3/7/11 in
the aligned 256px mask, with pixels outside the modeled crop retaining the
baseline composite. At radius11 the clear-frame source score recovers to about
0.58, nearly its original 0.5786, while the manually annotated book interior
remains pixel-exact at all radii. The occluded frame still fails at every radius:
source about 0.42–0.44 and target about 0.49–0.58. This experiment recovers clear
face detail but does not establish sufficient visible identity under occlusion.

Artifacts are separate in raw_occluder_probe/boundary_sweep and full precision
scores in xseg_boundary_sweep.json. Radius selection is tuning on two frames,
not held-out validation; edge pixels outside the annotated interior are not
proven preserved. CPU performance remains unresolved. No setting is enabled in
the service, and rejection thresholds are unchanged.

### XSeg GPU execution experiment (2026-10-06)

Added a restricted SHA-pinned Torch executor for XSeg using existing packages;
it rejects unexpected operators/interfaces and compares against OpenCV. Three
random NHWC BGR inputs had max error <=0.000180 and mean error <=0.000004.
Warm mask inference including input/output transfers measured median 14.67 ms,
p95 16.82 ms across ten calls, versus the earlier ~156 ms CPU measurements.
This benchmark excludes detection, composition and identity scoring.

The real-crop probe supports --gpu and verifies raw masks against OpenCV before
testing thresholds. GPU artifacts/reports are separate from CPU results.
This improves the runtime path for further occlusion research; it does not
resolve the candidate's identity rejection or authorize live promotion.
Evidence: raw_occluder_probe/xseg_gpu_parity.json and
xseg_boundary_sweep_gpu.json. No dependencies were installed and service
defaults remain unchanged.

### Controlled foreground effect on identity scoring (2026-10-06)

`occluded_identity_control.py` evaluates 20 cross-ID pairs from the existing
held-out evaluation. Positive controls are actual selected-source photos, not
generated swaps; negative controls are target photos. Identical seeded textured
foreground covers the lower 0/20/35/50/65% of aligned source and target images.
The current score compares the clean source embedding with the occluded output
and occluded target, isolating foreground-induced false rejection.

Known-source acceptance was 100/100/100/70/0% across those coverage levels.
Mean source similarity fell 1.0000/0.8626/0.6427/0.4451/0.1614 while shared-object
target similarity rose 0.0171/0.0583/0.1141/0.2882/0.5596. All unchanged-target
negative controls were rejected. These are controlled nuisance results, not
real-object generalization or false-accept calibration against arbitrary wrong
source identities. At 65% coverage insufficient visible evidence is also plausible.

Evidence: reference_backend/occluded_identity_control/results.json and example
panels. The result explains why preserving an occluder can reduce whole-crop
source/target separation without proving any particular swap's visible identity.
Live thresholds remain unchanged. An occlusion-conditioned scorer requires
different-photo same-ID positives, cross-ID negatives, multiple occluder shapes,
and a disjoint calibration/evaluation split before adoption.

### Disjoint different-photo occlusion verification (2026-10-06)

`calibrate_occluded_identity.py` used 120 identities from the original validation
partition, excluding all original training identities. Calibration and evaluation
each contain 60 distinct people with different-photo same-ID positives and cyclic
different-ID negatives. Seven fixed masks (clean, lower35/50/65, left35, right35,
center35) compare whole-crop scoring with oracle conditioning: the exact known
occluded region is set to neutral gray in every enrollment/comparison image.
Per-mask thresholds are selected above the maximum calibration negative only.
No service threshold is changed.

At evaluation, whole/oracle genuine acceptance was respectively: clean
91.7/91.7%, lower35 88.3/91.7%, lower50 73.3/73.3%, lower65 10.0/21.7%,
left35 91.7/93.3%, right35 91.7/88.3%, center35 68.3/50.0%.
Whole lower50 falsely accepted 1/60 negatives; other tested configurations
accepted 0/60. That small negative sample does not establish a deployment FAR.
Oracle masks are more accurate than predicted masks and do not generalize to
arbitrary shapes, real objects, target poses or generated swaps by assumption.

Evidence: reference_backend/occlusion_calibration/results.json includes names,
identity splits, skips, raw scores and calibration thresholds. The outcomes are
mixed and insufficient to replace the live guard. Occlusion-aware scoring remains
experimental; heavy occlusion and central/eye loss are unresolved limitations.

### Guarded occlusion video candidate (2026-10-06)

The offline evaluator now supports --occlusion-mask, using GPU XSeg binary
visibility with threshold0.1/radius11 (tuned earlier) and exact original-pixel
restoration in hidden regions. ReferenceFrameSession refuses this option without
the existing identity guard. No HTTP configuration enables it. Compositor tests
pass hidden-pixel, visible-baseline and untouched-background preservation.

Full 812-frame faceocc2 candidate completed: 697 swaps, 52 identity rejections,
63 missing detections, median full pipeline97.01 ms, p95 104.22 ms. Results and
selected raw panels are in reference_backend/masked_occlusion_video.
This is not a promotion decision: a fresh unmasked baseline using current
per-session tracking is being evaluated separately in current_baseline_occlusion_video
before matched decisions, visibility artifacts and motion residual are compared.
Additional clips and end-to-end browser runtime remain required.

### Matched occlusion video comparison (2026-10-06)

Fresh current-code baseline completed 812 frames: 699 swaps, 50 identity
rejections, median79.75 ms/p9585.42 ms. The binary-mask candidate made 697
swaps, rejected52, and added about17 ms median pipeline time. Its only decision
changes were new camera fallbacks at frames150 and737; neither previously rejected
frame became accepted. Decision transitions fell21 to19.

On688 common consecutive accepted frame pairs, approximate optical-flow
compensated encoded output residual increased from3.7618 to4.0955 (~8.9%).
This includes codec/flow error and excludes fallback transitions, but is sufficient
evidence against promoting the current hard-boundary variant. Selected lossless
original/baseline/candidate panels are in masked_occlusion_video/raw_comparison.png.

An offline visible-side feather option now uses distance-to-hidden-boundary
weights. Hidden pixels remain exactly original; updated compositor checks pass.
A3px candidate is evaluating separately in feathered_occlusion_video. This
experimental option remains absent from normal HTTP configuration and is not
a promotion until matched identity, mask preservation and stability checks pass.

### Visible-side feather video result (2026-10-06)

The3px feather candidate finished812 frames:697 swaps/52 rejections, median
97.14 ms/p95101.98 ms. Its only decision differences against the fresh baseline
remain camera fallbacks at150 and737. On688 common accepted consecutive pairs,
output residual was baseline3.7502/candidate4.0396 (~7.7% worse). Input support
differs slightly between comparisons; do not compare hard/soft absolute residuals
as if evaluated on exactly identical pixels.

The continuity evaluator now reports encoded original-panel mismatch and rejects
unequal dimensions. Mean original-panel discrepancy was0.000891 RGB levels,
maximum-frame mean0.1559. Codec effects remain, but average original mismatch
does not explain the observed residual increase by itself.

Evidence: feathered_occlusion_video/results.json, changed_decisions.json and
flow_stability.json. Both binary and feather variants remain unpromoted. Simply
softening current-frame boundaries is insufficient; further work needs temporal
mask stability while retaining exact hidden pixels and unchanged identity gates.

### Motion-compensated temporal mask candidate (2026-10-06)

An offline temporal option warps the previous visibility weight using current-to-
previous Farneback flow, applies a20-level photometric/in-bounds consistency gate,
and blends newly visible weights65/35. The result is capped by current visibility,
so currently hidden pixels cannot inherit prior generated color. State belongs to
each ReferenceFrameSession and clears on missing detection and close.
Pixel tests verify gradual reveal and immediate hidden-pixel restoration; real
session tests verify isolation and one-session close preserving another's state.

Full812-frame result:696 swaps/53 identity rejections, median107.82 ms/p95117.27 ms.
Compared with699 baseline swaps, camera fallbacks are added at83,150,737.
On687 common accepted consecutive pairs, residual was baseline3.7493 versus
candidate3.9085 (~4.2% worse). Mean encoded input-panel mismatch0.001333 levels.
This improves the excess residual relative to the static feather candidate but
does not beat baseline; support/frame sets differ, preventing an exact paired
hard/soft/temporal ranking without further matched-region analysis.

Evidence: reference_backend/temporal_occlusion_video/results.json,
flow_stability.json and changed_decisions.json. The temporal option remains
unpromoted and is not configurable through the normal HTTP service. Tests pass
the preservation/cleanup invariants, not general visual-quality acceptance.

### Spatial exclusion diagnostic (2026-10-06)

`inspect_mask_coverage.py` samples26 frames using independent IMAGE landmarks and
the same XSeg threshold/dilation. It separates the actual landmark oval from its
15px-eroded inner region. At clear frame0,436 aligned face pixels were excluded;
344 (~79%) lay outside the inner region and92 inside (~3% of the inner region).
Visualization places most exclusions at the forehead/hair boundary. At book
frame150,2034 face pixels were excluded, including783 inner pixels (~22% of the
inner region). Other sampled frames show variable perimeter exclusion; frames478,
480,720 fail independent IMAGE detection and are recorded rather than silently
replacing geometry. VIDEO tracking can still recognize some of these frames.

Evidence: reference_backend/mask_coverage/results.json and frame0/frame150 PNGs
(orange perimeter exclusion, magenta inner exclusion, green eroded-region outline).
This distinguishes spatial behavior on sampled frames; it is not ground-truth
segmentation, same-live-alignment evaluation, or proof of the continuity cause.
It motivates separating natural face-boundary segmentation from genuine foreground
occlusion before further smoothing/tuning, while leaving current gates unchanged.

### Semantic parsing alternative, isolated probe (2026-10-06)

Downloaded official FaceFusion/yakhyo BiSeNet ResNet18 parser with approval and
verified published SHA2562218b6183c26ca5c83303232d682a536c670c13ea9695f716c777d1f244eefe9.
Provenance records upstream MIT code/model metadata and CelebAMask-HQ training;
this does not remove the existing swap-weight usage restrictions. Existing
OpenCV executes19x512x512 logits without new packages.

`probe_semantic_mask.py` uses saved live VIDEO landmarks and verifies baseline
scores before comparing hard skin/eye/brow/nose/mouth/lip classes. Hair, glasses
and background preserve original pixels. Clear frame0 passes at source~0.55 /
target~0.35; frame478 passes at source~0.46 / target~0.16. Book frame150 still
fails (source~0.49 / target~0.48) and annotated book MAE remains0.41 rather than
zero. Thus the semantic parser alone also mislabels part of this occluder.
Warm CPU inference116–138 ms adds unacceptable direct per-frame overhead.

Evidence: reference_backend/semantic_mask_probe/results.json, logits-class label
PNGs and comparisons. This is three tuning frames, not a held-out quality pass.
No live parser is enabled. Any combined semantic/visibility approach must still
prove object preservation, identity, temporal continuity and full-frame runtime.

### Combined semantic/XSeg point probe (2026-10-06)

`probe_semantic_mask.py --combine-xseg` tests XSeg visibility with semantic hair
retaining the baseline composite. It uses saved VIDEO landmarks, verifies baseline
identity scores, and stores artifacts separately in combined_mask_probe. The
parser512 crop is resized to256 for XSeg in this point experiment, so it is not
an exact reproduction of the separate direct256 XSeg video sampling path.

Clear frame0 passes at source~0.57/target~0.35; profile frame478 passes at
source~0.50/target~0.10. Annotated book interior at150 is pixel-exact, but that
frame still fails source identity (~0.43 versus target~0.49). Retaining semantic
hair does not resolve this identity failure. These are three tuning frames and
do not prove improved continuity, general occluder preservation or runtime.

No combined model is enabled in the service. Parser CPU cost remains an additional
constraint; broader evaluation and a calibrated visible-identity method are still
needed before adopting the combined approach. Live guards remain unchanged.

### Semantic parser GPU path (2026-10-06)

Added separate restricted SHA-pinned TorchSemantic executor for the official
parser graph, including dynamic shape/resize operations and auxiliary heads.
OpenCV primary output is explicitly selected; its previous default was verified
identical to the primary output. Unexpected interfaces/operators fail closed.
No new packages were installed.

Two random normalized inputs matched primary logits with max error<=0.0000072,
mean error<=0.00000072 and100% label agreement. Warm execution including transfers
measured12.67 ms median (six calls after warmup), excluding crop preparation and
composition. Evidence: semantic_mask_probe/gpu_parity.json.

`probe_semantic_mask.py --combine-xseg --gpu` additionally verifies real-crop
logits against OpenCV and stores separate GPU artifacts. This addresses parser
execution overhead for broader research; it does not change the combined mask's
occluded identity rejection, prove video continuity or enable a live parser.

### Combined semantic/XSeg full-video ablation (2026-10-06)

The offline --semantic-hair option now evaluates GPU parser hair labels with
direct256 XSeg crops and saved VIDEO geometry. It retains baseline treatment in
parser-labeled hair while XSeg determines visibility elsewhere. This is an
experimental exception to the visibility mask, not a guarantee that erroneous
hair labels cannot alter an object. A compositor test verifies the intended
override; no HTTP configuration enables it.

The812-frame run finished normally:695 swaps/54 identity rejections, median
114.14 ms/p95120.79 ms. Four previously accepted frames83,150,179,737 fall back;
no formerly rejected frame becomes accepted. On685 valid common consecutive
pairs, approximate output residual was baseline3.7160/candidate3.8710 (~4.2%
worse), with mean encoded original-panel mismatch0.000891 levels.

Evidence: reference_backend/combined_occlusion_video/results.json,
flow_stability.json and changed_decisions.json. The natural hair/perimeter
distinction alone does not fix the regression. Combined, temporal, feather and
binary variants remain unpromoted; the existing research preview is unchanged.
Static point-mask improvements are insufficient evidence for full-video rollout.

### Complete frame-session photo holdout (2026-10-06)

The same existing 300 held-out cross-identity pairs now passed through the
default IMAGE research frame session using original full images, cached source
identity, full-frame rendering and the unchanged identity guard. The run
completed normally: 298 accepted swaps, two identity rejections, zero source
detection failures and zero missing targets. Median pipeline time was 85.90 ms,
p95 94.60 ms; accepted mean source similarity was 0.74652. Rejected outputs were
checked for exact original-frame fallback. Evidence:
reference_backend/live_holdout/results.json.

This closes the aligned-crop versus full-frame verification gap on these same
pairs. It does not establish video stability, perceptual quality, JPEG/WebRTC
performance or physical-device call readiness. No model or rollout default was
changed. Full training is complete; visual defects and physical call validation
still prevent a reliable completion date for general implementation.

### Evening two-device demo preparation (2026-10-06)

User specified different networks and testing via VS Code forwarding before
their GitHub/Render deployment. Start-ResearchPreview.ps1 now supports
-ForwardedDemo with configurable application/ML ports. It builds an isolated
production frontend and serves frontend/API/signaling together, keeping GPU
HTTP private on localhost. Each run generates unique JWT/demo passwords;
machine-local builds/logs/credentials are excluded by ml/.gitignore. Public
demo mode rejects default JWT secrets and weak/shared account passwords.
Client TURN URLs/username/credential and optional relay-only mode are configured
through build-time environment variables; no relay account was fabricated.

TypeScript checks and server build passed. Existing authenticated socket test
and new public-demo-auth test passed. Bounded actual startup on 5002/8002
confirmed CUDA research backend ready and exited normally after two seconds;
evidence local_preview/20261006-063937-688/startup.json. Existing services on
5001/5173 were preserved. Nothing was publicly forwarded or deployed by these
checks. Actual different-network media connectivity remains unverified and
may require TURN credentials. Instructions: ml/scripts/EVENING_DEMO.md.

This prepares a runnable research demo, not a defect-free release. Render
frontend/server deployment still needs an independently reachable compatible
ML backend and appropriate model usage rights.

## Evening demo update � 2026-10-06

Independent receiver-media research analysis is now integrated: received video JPEG samples go to the Xception swap-domain robust candidate; received mono PCM segments go to NAVER AASIST. JWT-authenticated user and tester roles are accepted, sender effect flags are excluded from model input, GPU work shares the swap inference lock, busy/unavailable/silent samples produce no fabricated score, and client cleanup does not stop call tracks. Enable with `-EnableMediaDetection` in `ml/scripts/Start-ResearchPreview.ps1`. Scores are explicitly uncalibrated research outputs, not authenticity verdicts.

Robust candidate six-epoch training completed. Reused identity-disjoint photo comparison: AUC 0.9826, 5/150 false positives, 134/150 true positives at threshold 0.7698. Received-video diagnostic: 12/115 natural frames falsely flagged, 67/112 swapped frames detected. Blur radius 0/1/2 false positives: 2/50, 12/50, 2/49. These correlated previously inspected clips and reused photos are not a fresh blind benchmark. Candidate is NOT promoted to reliable live classification.

AASIST official curated examples: 0/18 natural false positives and 104/130 spoof positives; limited curated scope. A natural browser-call speech sample falsely scored 0.999 despite original speech scoring 0.023. Audio boundary trimming did not fix this codec/domain failure. Reliable voice fake/real classification remains unresolved.

Both local demo roles and actual face inference passed HTTP checks; malformed audio rejected, silence returned insufficient with null score, unauthorized detection returned 401. Proxy ownership spoofing test and panel independence/empty-state tests pass. TypeScript and production client build pass. Home demo buttons now use the localhost auth endpoint; external forwarded links route to the password form. Unsupported zero-latency/browser-only/clone-certification marketing claims were corrected.

Dedicated updated demo: http://localhost:5002 ; private ML service: 127.0.0.1:8002. Existing 5001/5173 services preserved. No TURN relay is configured; two-device different-network connectivity and physical camera/mic quality still require testing. Research weights retain their license restrictions; do not present this as production certification.

## WebRTC inference transport � 2026-10-06

Added opt-in `-EnableWebRTCVideo` to the research launcher. aiortc 1.14.0 and binary dependencies are isolated in ignored `ml/runtime_webrtc/`; tested versions are recorded in `ml/requirements-webrtc-lock.txt`. Existing ML packages and trained weights are unchanged.

Tester-only JWT-authenticated SDP signaling goes through `/api/ml/video/sessions/{id}/offer`. Media travels as encrypted WebRTC video to the GPU service and back, with original call audio untouched. Input uses a single newest-frame slot while the existing model, blending, face fallback and source identity gate remain active. Face detection/changed status and processing FPS travel over a small telemetry data channel.

The client preserves one outgoing canvas track across source changes and runtime fallback. New decoded frames trigger canvas capture; stopped tracks and failed connections are cleaned up. Failed streaming startup or stalled media falls back to the original HTTP pipeline. Independent receiver detection remains enabled.

Validation: tester authorization, session ID and parsed JSON forwarding tests pass; existing HTTP authorization and media lifecycle regressions pass. Actual local RTP/DTLS benchmark: HTTP frame-request median 108.2 ms and 9.25 processing FPS, WebRTC 11.53 received FPS with median capture-to-return frame age 271.9 ms (p95 309.0 ms). These are different timing measures, so this does NOT establish lower latency. Browser prerecorded-video tests passed WebRTC streaming, source change while retaining WebRTC, stable outgoing track, original track preservation, and simulated signaling-failure HTTP recovery. Browser WebRTC FPS was about 6�7 in the background fixture, versus about 10 in HTTP fallback; this is not a proven smoothness improvement. Reports: `ml/experiments/reference_backend/webrtc_transport/`.

Cleanup review caught lingering frame-reader tasks and canceled-handshake errors; explicit reader cancellation and negotiation cleanup were added, and the subsequent browser test completed without those errors in the new server log. Physical two-device forwarded-network latency is not validated. VS Code forwarding does not carry WebRTC UDP media; configured STUN is tried, but difficult NAT/firewall paths may need TURN. No TURN account is currently configured. Transport UI distinguishes WebRTC versus HTTP neural preview. Do not describe this as lag-free or a proven cross-network fix.

Run: `./ml/scripts/Start-ResearchPreview.ps1 -ForwardedDemo -SimpleDemoPasswords -EnableMediaDetection -EnableWebRTCVideo -AppPort 5002 -MlPort 8002`. Demo credentials remain user/user123 and tester/tester123 as explicitly requested.

## Receiver detector starvation fix � 2026-10-06

User observed both learned detectors stuck on Waiting for GPU during a live swap. The shared lock was being checked nonblockingly, so continuous swap processing repeatedly rejected samples with HTTP 429. Changed detection to acquire the shared FIFO lock with a two-second bound, limited to four pending samples total and one per receiver/media kind. The latest-frame swap pipeline stays active; classifier weights and quality thresholds are unchanged. Canceled waiting requests cannot release someone else's lock, and canceled running inference holds the lock until its worker finishes. Three focused scheduling/cancellation tests pass.

Actual RTP/DTLS recorded-media contention test returned HTTP 200 and measured results for all three video and three speech requests while swapping continued: 40 received frames, 11.30 FPS, frame-age median 270.9 ms, p95 311.6 ms. Report: `ml/experiments/reference_backend/webrtc_transport/benchmark_with_detection.json`. This confirms local scheduling, not cross-device classification accuracy or zero latency. The monitor header now says Low audio anomaly instead of implying an overall Low risk verdict while face analysis is pending. Dedicated demo was restarted with this fix on port 5002.
