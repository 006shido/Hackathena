
## WebRTC inference transport — 2026-10-06

Added opt-in `-EnableWebRTCVideo` to the research launcher. aiortc 1.14.0 and binary dependencies are isolated in ignored `ml/runtime_webrtc/`; tested versions are recorded in `ml/requirements-webrtc-lock.txt`. Existing ML packages and trained weights are unchanged.

Tester-only JWT-authenticated SDP signaling goes through `/api/ml/video/sessions/{id}/offer`. Media travels as encrypted WebRTC video to the GPU service and back, with original call audio untouched. Input uses a single newest-frame slot while the existing model, blending, face fallback and source identity gate remain active. Face detection/changed status and processing FPS travel over a small telemetry data channel.

The client preserves one outgoing canvas track across source changes and runtime fallback. New decoded frames trigger canvas capture; stopped tracks and failed connections are cleaned up. Failed streaming startup or stalled media falls back to the original HTTP pipeline. Independent receiver detection remains enabled.

Validation: tester authorization, session ID and parsed JSON forwarding tests pass; existing HTTP authorization and media lifecycle regressions pass. Actual local RTP/DTLS benchmark: HTTP frame-request median 108.2 ms and 9.25 processing FPS, WebRTC 11.53 received FPS with median capture-to-return frame age 271.9 ms (p95 309.0 ms). These are different timing measures, so this does NOT establish lower latency. Browser prerecorded-video tests passed WebRTC streaming, source change while retaining WebRTC, stable outgoing track, original track preservation, and simulated signaling-failure HTTP recovery. Browser WebRTC FPS was about 6–7 in the background fixture, versus about 10 in HTTP fallback; this is not a proven smoothness improvement. Reports: `ml/experiments/reference_backend/webrtc_transport/`.

Cleanup review caught lingering frame-reader tasks and canceled-handshake errors; explicit reader cancellation and negotiation cleanup were added, and the subsequent browser test completed without those errors in the new server log. Physical two-device forwarded-network latency is not validated. VS Code forwarding does not carry WebRTC UDP media; configured STUN is tried, but difficult NAT/firewall paths may need TURN. No TURN account is currently configured. Transport UI distinguishes WebRTC versus HTTP neural preview. Do not describe this as lag-free or a proven cross-network fix.

Run: `./ml/scripts/Start-ResearchPreview.ps1 -ForwardedDemo -SimpleDemoPasswords -EnableMediaDetection -EnableWebRTCVideo -AppPort 5002 -MlPort 8002`. Demo credentials remain user/user123 and tester/tester123 as explicitly requested.
