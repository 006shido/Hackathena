# Evening demo

Use the existing verified research preview for a tester-to-user demo.
Training has finished. The selected backend has passed 298 of 300 full-frame
photo identity checks; this is not a measure of visual perfection.

## Start

From the repository root in PowerShell:

```powershell
./ml/scripts/Start-ResearchPreview.ps1 -ForwardedDemo -AppPort 5002 -MlPort 8002
```

In VS Code's Ports panel, forward **5002**, then copy its **HTTPS** forwarded
address. Use that same address on both devices. Private forwarding requires
authorized Microsoft/GitHub access; Public visibility allows the other device
to reach the app without that tunnel sign-in. The app still requires login.
Do not forward 8002: it is the private GPU service. Do not expose the Vite
development server on 5173.

Sign in as `tester` on the tester device and `user` on the other. Every launcher
run generates new passwords in `demo-access.local.txt` under the log directory
printed at startup. Share only the appropriate password with your demo partner.
This directory is excluded from Git. Previous passwords and tokens no longer
apply after restarting the launcher. Keep the launcher and VS Code forwarding
running; Ctrl+C stops the launcher's services. Stop port forwarding afterward.

The machine running this launcher must stay online and has the GPU backend.
Changing AppPort is supported with -ForwardedDemo if another service occupies
5002; forward the selected application port.

## Different-network media connection

HTTPS forwarding carries the website, API and signaling. WebRTC media uses its
own connection. STUN is configured by default; restrictive networks may need a
TURN relay. Configure provider-issued temporary TURN credentials in your local
PowerShell environment before launching:

```powershell
$env:VITE_TURN_URLS = 'turn:YOUR_RELAY:3478,turns:YOUR_RELAY:5349'
$env:VITE_TURN_USERNAME = 'YOUR_TEMPORARY_USERNAME'
$env:VITE_TURN_CREDENTIAL = 'YOUR_TEMPORARY_TURN_PASSWORD'
# Optional: verify relay connectivity by forcing it during rehearsal.
$env:VITE_WEBRTC_RELAY_ONLY = 'true'
```

Use the exact URLs and transports supplied by your provider. TURN credentials
are delivered to browsers in the demo build; use limited-lived credentials,
never an account API key. Do not commit credentials or put them in chat.
Without an actual two-device test, different-network connectivity is unverified.

## Rehearsal

1. Join the same call with the tester and user accounts. Verify ordinary video
   and microphone audio first.
2. Select a clear, front-facing source photo in the tester's Face panel.
   Keep the target face well lit, mostly frontal and unobstructed.
3. Verify the changed video reaches the user and the video-modification status
   updates. A rejected face frame returns the original camera image.
4. Enable a voice preset, speak a sentence and verify changed audio reaches
   the user along with the audio-modification status.
5. Switch the source photo, disable effects and verify restoration.
6. End the call and verify camera/microphone indicators clear.

This sequence still needs a rehearsal with real camera and microphone devices.
The automated call checks used recorded video and audio.

## What can be demonstrated accurately

- Face substitution and voice DSP presets through the research preview.
- Tester-to-user modification flags, original-frame fallback and effect reset.
- Completed training and the separate evaluation results.

Modification flags describe active tester effects. They are not independently
validated video/audio anomaly classifier predictions. Voice presets are DSP,
not a trained voice-cloning model. Difficult poses and occlusion remain weak.

The launcher binds to localhost; VS Code supplies remote HTTPS access.
The pretrained face-swap weights are limited to noncommercial research use.
See RESEARCH_PREVIEW.md for startup requirements and log locations.

## Render deployment after testing

Committing the frontend/server does not move this computer's GPU or downloaded
model assets into Render. The deployed server must be able to reach a compatible
ML service via ML_SERVICE_HOST/ML_SERVICE_PORT, with networking and access
controls verified separately. A frontend/server-only deploy cannot perform
this GPU face swap by itself. Keep deployment separate from the evening demo.
