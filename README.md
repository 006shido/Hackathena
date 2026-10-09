# DeepTrace 🛡️ — Secure 1-to-1 Video Calling with Attack Simulation

**Complete research setup:** [GETTING_STARTED.md](GETTING_STARTED.md) covers frontend, backend, Phase 6G, live face swapping, and face/audio anomaly detection. Model downloads have separate [research terms](MODEL_NOTICES.md).

**DeepTrace** is a modern, browser-based, peer-to-peer 1-to-1 video calling application built for hackathon security demonstrations. It features role-based access control and an isolated **Attack Simulator** that enables authorized testers to inject simulated deepfake facial manipulations and synthetic voice transformations directly into outgoing WebRTC streams in real time.

The application architecture includes designated integration points for a future multi-modal AI deepfake detection pipeline (Voice + Face + Lip-Sync analysis).

---

## 🌟 Key Features

- **Real Browser-to-Browser WebRTC**: Full peer-to-peer audio and video streaming using modern WebRTC (`RTCPeerConnection`, `RTCRtpSender.replaceTrack()`), STUN servers, and dynamic stream renegotiation.
- **Strict Role-Based Security**:
  - **`USER`**: Normal meeting participant. Can create/join calls, toggle camera/mic, share screen, and view the remote participant. Normal users **cannot** see or trigger any attack simulation controls.
  - **`TESTER`**: Security evaluation persona. Granted access to the **Attack Simulator** panel to simulate adversarial deepfake attacks for testing security monitoring.
- **Server-Side Enforcement**: Role validation is enforced on both HTTP sessions and WebSocket connections. Attack telemetry from unauthorized accounts is strictly rejected server-side.
- **2-Participant Room Limit**: Rooms strictly enforce a maximum of 2 participants per room. Third parties attempting to enter receive `"This room is full."`.
- **Pre-Call Audio & Video Check**: Live device preview allowing camera/mic verification and hardware toggling before entering active calls.
- **Controlled Outgoing Media Attacks**:
  - **Face Simulation**: Offscreen HTML5 Canvas pipeline rendering synthetic facial persona overlays, facial landmark tracking meshes, and neural boundary blending seamlessly converted to outgoing `MediaStreamTrack` via `canvas.captureStream(30)`.
  - **Voice Transformation**: Web Audio API audio graph (ring modulation, formant frequency shifting, wave-shaper harmonic distortion) transforming microphone input into synthetic/robotic vocoder timbres sent across WebRTC.
  - **Combined Attack**: Simultaneously triggers face swap and voice transformation with high-visibility audit telemetry.
  - **Instant Reset**: Restores original camera and microphone media tracks instantly.
- **Future DeepTrace Detection Interface**: Dedicated "DeepTrace Monitoring" panel illustrating where upcoming AI detection models (Face, Voice, Lip-Sync, and Risk Engine) will ingest media feeds.

---

## 🏗️ Architecture

```
                    ┌─────────────────────────┐
                    │ DeepTrace Server (5001) │
                    │ - JWT Authentication    │
                    │ - Room Manager (Max 2)  │
                    │ - WebRTC Signaling (WS) │
                    │ - Tester Role Auditing  │
                    └───────────┬─────────────┘
                                │
               ┌────────────────┴────────────────┐
               │ Signaling (SDP Offer/Answer/ICE)│
               ▼                                 ▼
    ┌───────────────────────┐         ┌───────────────────────┐
    │  Browser Tab 1 (Host) │◄───────►│ Browser Tab 2 (Peer)  │
    │  WebRTC Media Stream  │ Direct  │  WebRTC Media Stream  │
    │  (Camera + Audio)     │ P2P     │  (Camera + Audio)     │
    └───────────┬───────────┘         └───────────┬───────────┘
                │                                 │
    ┌───────────▼───────────┐         ┌───────────▼───────────┐
    │ TESTER Attack Engine  │         │ DeepTrace Monitoring  │
    │ - Canvas Face Swap    │         │ Placeholder (Planned) │
    │ - WebAudio Vocoder    │         │ - Face Anomaly        │
    │ - replaceTrack()      │         │ - Voice Anomaly       │
    └───────────────────────┘         │ - Lip-Sync Anomaly    │
                                      │ - Risk Engine         │
                                      └───────────────────────┘
```

---

## 🔑 Demo Credentials

| Role | Username | Password | Capabilities |
| :--- | :--- | :--- | :--- |
| **USER** | `user` | `user123` | Normal 1-to-1 video calling. No attack controls. |
| **TESTER** | `tester` | `tester123` | Video calling + In-Call **Attack Simulator** panel. |

*Tip: The login screen contains 1-click buttons to auto-populate either account instantly.*

---

## 🚀 Quick Start & Installation

### Prerequisites
- **Node.js** (v18.0.0 or higher, tested on v24)
- **npm** (v9.0.0 or higher)

### 1. Install Dependencies
Run the installation command from the repository root:

```bash
npm run install:all
```

Or install separately:
```bash
npm install
npm install --prefix server
npm install --prefix client
```

### 2. Start the Application
Run both the signaling server and the Vite React frontend concurrently with a single command:

```bash
npm run dev
```

The services will start at:
- **Signaling Server**: `http://localhost:5001`
- **Frontend Client**: `http://localhost:5173`

---

## 🧪 Demonstration Walkthrough (Step-by-Step)

To reproduce the exact hackathon presentation flow:

1. **Step 1: Tester Login**
   - Open your browser to `http://localhost:5173`.
   - Click the **TESTER** demo account button (or type `tester` / `tester123`).
   - Click **Login**. You will land on the **Tester Dashboard** showing the `TESTER MODE` security card.

2. **Step 2: Create Test Call**
   - Click **Create Test Call**.
   - Review your camera and microphone in the **Audio & Video Check** preview.
   - Click **Join Call**.
   - Note the generated Room ID displayed in the top header (e.g. `ABC-123`). Click the copy icon to copy it.

3. **Step 3: User Login (Second Window)**
   - Open a second browser window (or Incognito tab) to `http://localhost:5173`.
   - Click the **USER** demo account button (or type `user` / `user123`).
   - Click **Login**. You will land on the **User Dashboard** with standard calling options.

4. **Step 4: User Joins Call**
   - Paste the Tester's Room ID into the **Join Call** input and click **Join Room**.
   - Review device preview and click **Join Call**.

5. **Step 5: Active WebRTC Communication**
   - Both participants are now connected via WebRTC!
   - Remote video and audio stream between the two tabs.
   - Verify that the normal `user` does **NOT** see an Attack Simulator panel.

6. **Step 6: Tester Activates Attack Simulation**
   - On the **Tester's** screen, click **Attack Simulator** (or view the right drawer).
   - Click **Face + Voice Attack** (or toggle individual effects).
   - Tester sees: `SIMULATED DEEPFAKE ATTACK ACTIVE` with live watermark.
   - Look at the **User's screen**:
     - The user immediately sees the tester's face transformed with the synthetic persona avatar mask!
     - The user hears the tester's voice transformed into a robotic synthetic vocoder!
     - This stream modification is executed via `RTCRtpSender.replaceTrack()`.

7. **Step 7: Reset & Restore**
   - On the Tester's screen, click **Reset Attack**.
   - Outgoing video and audio seamlessly revert back to the original camera and microphone feeds.

---

## 🔒 Security Model & Validation

DeepTrace enforces role separation across multiple layers:

1. **Client-Side Rendering Guard**: Attack simulation components and control buttons are conditionally rendered only when `user.role === 'tester'`.
2. **Session Verification**: The backend issues signed JWT tokens containing the authenticated user's role.
3. **Socket Handshake Auth**: The Socket.IO signaling connection inspects and decodes the JWT token on connection.
4. **Server-Side Event Filtering**: When an `attack-simulation-update` event is dispatched, the server checks `socket.data.user.role === 'tester'`. If a non-tester emits this event, it is immediately rejected with:
   ```
   Security Policy Violation: Only authenticated TESTER accounts are authorized to execute attack simulations.
   ```
5. **Strict Capacity Limit**: The signaling room manager limits rooms to exactly 2 participants. Additional connection attempts receive `"This room is full."`.

---

## 🔮 Future DeepTrace Detection Architecture

In Phase 2, the `SecurityPanel` will integrate with the DeepTrace real-time analysis pipeline:

```
WebRTC Media (Audio + Video)
      │
      ▼
Audio & Video Frame Capture
      │
      ├───────────────────────────────┬───────────────────────────────┐
      ▼                               ▼                               ▼
Voice Analysis Engine           Face Analysis Engine            Lip-Sync Engine
(Spectral flux, vocoder         (Biometric landmark jitter,     (Phoneme-viseme temporal
harmonic artifact detection)    neural boundary blending)       audio-visual correlation)
      │                               │                               │
      └───────────────────────────────┼───────────────────────────────┘
                                      ▼
                             DeepTrace Risk Engine
                        (Aggregated Confidence Score)
                                      ▼
                          DeepTrace Monitoring UI
                       (Real-Time Risk Alerts & HUD)
```

The placeholder `SecurityPanel` component is already integrated with incoming stream telemetry hooks to display real-time analysis once the detection model inference worker is attached.

---

## 🛠️ Verification & Test Suite

Run the automated integration test suite to verify authentication, role enforcement, room limits, and WebRTC signaling:

```bash
node test_runner.mjs
```

**Verified Test Cases:**
- ✓ Tester Login & Role Validation
- ✓ User Login & Role Validation
- ✓ Tester Socket.IO connection & Room Creation
- ✓ User Socket.IO connection & Peer-to-Peer Joining
- ✓ WebRTC SDP Offer & Answer Signaling Exchange
- ✓ Tester Attack Simulation Activation (`combined` mode)
- ✓ Server-Side Rejection of Unauthorized User Attack Simulations
- ✓ Rejection of 3rd Participant with `"This room is full."`
- ✓ Participant Disconnect & Room Cleanup
# Research release setup

For the complete frontend/backend/Phase 6G/live face-swap/anomaly-detection stack, follow [GETTING_STARTED.md](GETTING_STARTED.md). Downloaded models have separate [research restrictions and notices](MODEL_NOTICES.md).
