import { authService } from '../services/auth';

export class WebRtcVideoPreview {
  public ready = false;
  public faceDetected = false;
  public faceChanged = false;
  public fps = 0;
  public pipelineMs = 0;
  private peer: RTCPeerConnection | null = null;
  private camera: MediaStreamTrack | null = null;
  private output: MediaStreamTrack | null = null;
  private session: string | null = null;
  private abort: AbortController | null = null;
  private epoch = 0;
  public onFailure: (() => void) | null = null;

  private headers() {
    const token = authService.getToken();
    if (!token) throw new Error('Sign in as tester to stream a face swap.');
    return { Authorization: `Bearer ${token}` };
  }

  private async release(session: string | null) {
    if (session) await fetch(`/api/ml/video/sessions/${session}`, {
      method: 'DELETE', headers: this.headers(), keepalive: true, signal: AbortSignal.timeout(3000),
    }).catch(() => {});
  }

  public async start(track: MediaStreamTrack, source: Blob): Promise<MediaStreamTrack> {
    this.stop();
    const epoch = this.epoch;
    this.abort = new AbortController();
    const signal = AbortSignal.any([this.abort.signal, AbortSignal.timeout(20000)]);
    const form = new FormData(); form.append('source', source, 'source.jpg');
    try {
      let response: Response | undefined;
      for (let attempt = 0; attempt < 4; attempt++) {
        response = await fetch('/api/ml/video/sessions', { method: 'POST', headers: this.headers(), body: form, signal });
        if (response.status !== 429) break;
        await new Promise(resolve => setTimeout(resolve, 150 * (attempt + 1)));
      }
      if (!response?.ok) throw new Error('Could not create WebRTC inference session.');
      const session = (await response.json()).session_id;
      if (epoch !== this.epoch) { await this.release(session); throw new Error('Streaming startup cancelled.'); }
      this.session = session;
      const turnUrls = String(import.meta.env.VITE_TURN_URLS || '').split(',').map(value => value.trim()).filter(Boolean);
      const pc = new RTCPeerConnection({ iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
        ...(turnUrls.length ? [{ urls: turnUrls, username: import.meta.env.VITE_TURN_USERNAME, credential: import.meta.env.VITE_TURN_CREDENTIAL }] : []),
      ] });
      this.peer = pc;
      const channel = pc.createDataChannel('swap-telemetry', { ordered: false, maxRetransmits: 0 });
      channel.onmessage = event => {
        if (epoch !== this.epoch) return;
        try {
          const value = JSON.parse(event.data);
          this.faceDetected = value.face_detected === true; this.faceChanged = value.face_changed === true;
          this.fps = Number.isFinite(value.fps) ? value.fps : 0;
          this.pipelineMs = Number.isFinite(value.pipeline_ms) ? value.pipeline_ms : 0;
        } catch { /* Invalid telemetry never becomes a measurement. */ }
      };
      this.camera = track.clone();
      const sender = pc.addTrack(this.camera, new MediaStream([this.camera]));
      const parameters = sender.getParameters();
      if (!parameters.encodings?.length) parameters.encodings = [{}];
      parameters.encodings[0].maxFramerate = 15;
      parameters.encodings[0].maxBitrate = 2000000;
      parameters.encodings[0].scaleResolutionDownBy = Math.max(1, (track.getSettings().width || 640) / 640);
      await sender.setParameters(parameters);
      let resolveTrack: (value: MediaStreamTrack) => void = () => {};
      let rejectTrack: (error: Error) => void = () => {};
      const received = new Promise<MediaStreamTrack>((resolve, reject) => { resolveTrack = resolve; rejectTrack = reject; });
      // Avoid an unhandled rejection if signaling fails before this is awaited.
      void received.catch(() => {});
      let remote: MediaStreamTrack | null = null;
      const connected = () => {
        if (pc.connectionState === 'connected' && remote && epoch === this.epoch) resolveTrack(remote);
      };
      pc.ontrack = event => { if (event.track.kind === 'video') { remote = event.track; connected(); } };
      pc.onconnectionstatechange = () => {
        connected();
        if (pc.connectionState === 'failed' || pc.connectionState === 'closed') {
          this.ready = false; this.faceChanged = false; this.faceDetected = false;
          rejectTrack(new Error('WebRTC media connection failed.'));
          if (this.output && epoch === this.epoch) this.onFailure?.();
        }
      };
      const onAbort = () => rejectTrack(new Error('WebRTC connection timed out or cancelled.'));
      signal.addEventListener('abort', onAbort, { once: true });
      try {
        await pc.setLocalDescription(await pc.createOffer());
        await new Promise<void>((resolve, reject) => {
          if (pc.iceGatheringState === 'complete') { resolve(); return; }
          const timer = setTimeout(() => { cleanup(); resolve(); }, 2500);
          const update = () => { if (pc.iceGatheringState === 'complete') { cleanup(); resolve(); } };
          const cancel = () => { cleanup(); reject(new Error('ICE gathering cancelled.')); };
          const cleanup = () => { clearTimeout(timer); pc.removeEventListener('icegatheringstatechange', update); signal.removeEventListener('abort', cancel); };
          pc.addEventListener('icegatheringstatechange', update); signal.addEventListener('abort', cancel, { once: true });
        });
        const answerResponse = await fetch(`/api/ml/video/sessions/${session}/offer`, {
          method: 'POST', headers: { ...this.headers(), 'Content-Type': 'application/json' },
          body: JSON.stringify(pc.localDescription), signal,
        });
        if (!answerResponse.ok) throw new Error('WebRTC signaling failed.');
        await pc.setRemoteDescription(await answerResponse.json());
        const result = await received;
        if (epoch !== this.epoch) throw new Error('Streaming startup cancelled.');
        this.output = result; this.ready = true;
        return result;
      } finally { signal.removeEventListener('abort', onAbort); }
    } catch (error) {
      if (epoch === this.epoch) this.stop();
      throw error;
    }
  }

  public stop() {
    this.epoch++;
    this.abort?.abort(); this.abort = null;
    const peer = this.peer; this.peer = null;
    if (peer) { peer.onconnectionstatechange = null; peer.ontrack = null; peer.close(); }
    this.camera?.stop(); this.camera = null;
    this.output?.stop(); this.output = null;
    const session = this.session; this.session = null;
    this.ready = false; this.faceChanged = false; this.faceDetected = false; this.fps = 0;
    void this.release(session);
  }
}
