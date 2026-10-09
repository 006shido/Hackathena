import { authService } from '../services/auth';
import { WebRtcVideoPreview } from './webRtcVideoPreview';

/** One outstanding frame, per-call source cache, and explicit cleanup. */
class HttpNeuralVideoPreview {
  private canvas = document.createElement('canvas');
  private input = document.createElement('canvas');
  private video = document.createElement('video');
  private stream: MediaStream | null = null;
  private session: string | null = null;
  private abort: AbortController | null = null;
  private generation = 0;
  private loopEpoch = 0;
  private timer: number | null = null;
  private running = false;
  private frames = 0;
  private started = 0;
  public faceDetected = false;
  public faceChanged = false;
  public fps = 0;
  public ready = false;

  constructor() {
    this.video.muted = true;
    this.video.playsInline = true;
  }

  private headers(): HeadersInit {
    const token = authService.getToken();
    if (!token) throw new Error('Sign in as a tester to use neural video preview.');
    return { Authorization: `Bearer ${token}` };
  }

  private async release(session: string | null) {
    const token = authService.getToken();
    if (!token) return;
    if (session) await fetch(`/api/ml/video/sessions/${session}`, {
      method: 'DELETE', headers: { Authorization: `Bearer ${token}` }, keepalive: true,
      signal: AbortSignal.timeout(2000),
    }).catch(() => {});
  }

  public async setSource(source: Blob) {
    const generation = ++this.generation;
    this.abort?.abort();
    this.abort = new AbortController();
    const old = this.session;
    this.session = null;
    this.ready = false;
    this.faceDetected = false;
    this.faceChanged = false;
    await this.release(old);
    const form = new FormData();
    form.append('source', source, 'source.jpg');
    let response: Response | undefined;
    for (let attempt = 0; attempt < 4; attempt++) {
      if (generation !== this.generation) return;
      response = await fetch('/api/ml/video/sessions', {
        method: 'POST', headers: this.headers(), body: form,
        signal: AbortSignal.any([this.abort.signal, AbortSignal.timeout(10000)]),
      });
      if (response.status !== 429) break;
      await new Promise(resolve => window.setTimeout(resolve, 100 * (attempt + 1)));
    }
    if (!response) return;
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || data.error || 'Could not initialize neural video.');
    if (generation !== this.generation) {
      await this.release(data.session_id);
      return;
    }
    this.session = data.session_id;
    this.ready = true;
  }

  public async start(track: MediaStreamTrack, source: Blob): Promise<MediaStreamTrack> {
    this.stop();
    const startupEpoch = this.loopEpoch;
    this.running = true;
    const settings = track.getSettings();
    const width = Math.min(settings.width || 640, 640);
    const height = Math.round(width * (settings.height || 480) / (settings.width || 640));
    this.canvas.width = this.input.width = width;
    this.canvas.height = this.input.height = height;
    this.video.srcObject = new MediaStream([track]);
    try {
      await this.video.play();
      if (!this.running || startupEpoch !== this.loopEpoch) throw new Error('Neural preview startup was cancelled.');
      await this.setSource(source);
      if (!this.running || startupEpoch !== this.loopEpoch) throw new Error('Neural preview startup was cancelled.');
      this.canvas.getContext('2d')!.drawImage(this.video, 0, 0, width, height);
      this.stream = this.canvas.captureStream(15);
      this.started = performance.now();
      this.frames = 0;
      void this.frame(startupEpoch);
      return this.stream.getVideoTracks()[0];
    } catch (error) {
      if (startupEpoch === this.loopEpoch) this.stop();
      throw error;
    }
  }

  private async frame(epoch: number) {
    if (!this.running || epoch !== this.loopEpoch) return;
    const started = performance.now();
    const generation = this.generation;
    const session = this.session;
    try {
      const context = this.input.getContext('2d')!;
      context.drawImage(this.video, 0, 0, this.input.width, this.input.height);
      if (!session) {
        this.canvas.getContext('2d')!.drawImage(this.input, 0, 0);
      } else {
        const timestamp = performance.now();
        const blob = await new Promise<Blob | null>(resolve => this.input.toBlob(resolve, 'image/jpeg', 0.9));
        if (!blob || !this.running || generation !== this.generation) return;
        const form = new FormData();
        form.append('frame', blob, 'frame.jpg');
        form.append('timestamp_ms', String(timestamp));
        const response = await fetch(`/api/ml/video/sessions/${session}/frame`, {
          method: 'POST', headers: this.headers(), body: form,
          signal: AbortSignal.any([this.abort!.signal, AbortSignal.timeout(2000)]),
        });
        if (response.ok) {
          const bitmap = await createImageBitmap(await response.blob());
          try {
            if (this.running && generation === this.generation) {
              this.canvas.getContext('2d')!.drawImage(bitmap, 0, 0);
              this.ready = true;
              this.faceDetected = response.headers.get('x-face-detected') === 'true';
              this.faceChanged = response.headers.get('x-face-changed') === 'true';
              this.frames++;
              this.fps = this.frames * 1000 / Math.max(1, performance.now() - this.started);
            }
          } finally { bitmap.close(); }
        } else if (response.status !== 429 && this.running && generation === this.generation) {
          this.ready = false;
          this.faceDetected = false;
          this.faceChanged = false;
          this.canvas.getContext('2d')!.drawImage(this.input, 0, 0);
        }
      }
    } catch {
      if (this.running && generation === this.generation) {
        this.ready = false;
        this.faceDetected = false;
        this.faceChanged = false;
        this.canvas.getContext('2d')!.drawImage(this.input, 0, 0);
      }
    } finally {
      if (this.running && epoch === this.loopEpoch) {
        this.timer = window.setTimeout(() => void this.frame(epoch), Math.max(0, 67 - (performance.now() - started)));
      }
    }
  }

  public stop() {
    this.running = false;
    this.loopEpoch++;
    this.generation++;
    this.abort?.abort();
    if (this.timer !== null) window.clearTimeout(this.timer);
    this.timer = null;
    this.stream?.getTracks().forEach(track => track.stop());
    this.stream = null;
    this.video.pause();
    this.video.srcObject = null;
    const session = this.session;
    this.session = null;
    this.ready = false;
    this.faceDetected = false;
    this.faceChanged = false;
    void this.release(session);
  }
}

/** Stable outgoing track: WebRTC inference first, HTTP fallback if unavailable. */
export class NeuralVideoPreview {
  private http = new HttpNeuralVideoPreview();
  private rtc = new WebRtcVideoPreview();
  private usingRtc = false;
  private streaming = false;
  private epoch = 0;
  private source: Blob | null = null;
  private original: MediaStreamTrack | null = null;
  private canvas = document.createElement('canvas');
  private video = document.createElement('video');
  private output: MediaStream | null = null;
  private frameRequest: number | null = null;
  private fallbackBusy = false;
  private sourceGeneration = 0;
  private sourceStarting = false;
  private lastFrame = 0;
  public transport = 'http';
  public get ready() { return this.usingRtc ? this.rtc.ready : this.http.ready; }
  public get faceDetected() { return this.usingRtc ? this.rtc.faceDetected : this.http.faceDetected; }
  public get faceChanged() { return this.usingRtc ? this.rtc.faceChanged : this.http.faceChanged; }
  public get fps() { return this.usingRtc ? this.rtc.fps : this.http.fps; }

  constructor() { this.video.muted = true; this.video.playsInline = true; }

  private pump(epoch: number) {
    if (!this.streaming || epoch !== this.epoch) return;
    if (this.video.readyState >= 2) {
      this.canvas.getContext('2d')!.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
      (this.output?.getVideoTracks()[0] as CanvasCaptureMediaStreamTrack | undefined)?.requestFrame?.();
      this.lastFrame = performance.now();
    }
    if (typeof this.video.requestVideoFrameCallback === 'function') {
      this.frameRequest = this.video.requestVideoFrameCallback(() => this.pump(epoch));
    } else this.frameRequest = window.setTimeout(() => this.pump(epoch), 67);
  }

  private async bind(track: MediaStreamTrack, epoch: number) {
    if (!this.streaming || epoch !== this.epoch) return;
    if (this.frameRequest !== null) {
      if (typeof this.video.cancelVideoFrameCallback === 'function') this.video.cancelVideoFrameCallback(this.frameRequest);
      else window.clearTimeout(this.frameRequest);
    }
    this.video.srcObject = new MediaStream([track]);
    await this.video.play();
    this.pump(epoch);
  }

  private async fallback(epoch: number) {
    if (this.fallbackBusy || !this.streaming || epoch !== this.epoch || !this.original || !this.source) return;
    this.fallbackBusy = true;
    this.rtc.stop(); this.usingRtc = false; this.transport = 'http-fallback';
    try {
      const track = await this.http.start(this.original, this.source);
      if (epoch === this.epoch && this.streaming) await this.bind(track, epoch);
    } catch { /* The panel exposes not-ready status; original call audio remains intact. */ }
    finally { this.fallbackBusy = false; }
  }

  public async start(track: MediaStreamTrack, source: Blob): Promise<MediaStreamTrack> {
    this.stop();
    if (import.meta.env?.VITE_ENABLE_WEBRTC_VIDEO !== 'true' || typeof RTCPeerConnection === 'undefined') {
      this.transport = 'http';
      return this.http.start(track, source);
    }
    const epoch = this.epoch;
    this.streaming = true; this.original = track; this.source = source;
    const settings = track.getSettings();
    this.canvas.width = Math.min(640, settings.width || 640);
    this.canvas.height = Math.round(this.canvas.width * (settings.height || 480) / (settings.width || 640));
    try {
      let transformed: MediaStreamTrack;
      try {
        transformed = await this.rtc.start(track, source);
        this.usingRtc = true; this.transport = 'webrtc';
      } catch (error) {
        if (!this.streaming || epoch !== this.epoch) throw error;
        transformed = await this.http.start(track, source); this.transport = 'http-fallback';
      }
      if (!this.streaming || epoch !== this.epoch) throw new Error('Preview startup cancelled.');
      this.output = this.canvas.captureStream(0);
      if (typeof (this.output.getVideoTracks()[0] as CanvasCaptureMediaStreamTrack).requestFrame !== 'function') {
        this.output.getTracks().forEach(value => value.stop()); this.output = this.canvas.captureStream(15);
      }
      await this.bind(transformed, epoch);
      this.rtc.onFailure = () => { void this.fallback(epoch); };
      // A silent stalled connection must also recover without replacing the call track.
      this.watch(epoch);
      return this.output.getVideoTracks()[0];
    } catch (error) { if (epoch === this.epoch) this.stop(); throw error; }
  }

  private watchdog: number | null = null;
  private watch(epoch: number) {
    this.lastFrame = performance.now();
    this.watchdog = window.setInterval(() => {
      if (this.usingRtc && !this.sourceStarting && epoch === this.epoch && performance.now() - this.lastFrame > 5000) void this.fallback(epoch);
    }, 1000);
  }

  public async setSource(source: Blob) {
    const generation = ++this.sourceGeneration;
    if (!this.streaming || !this.original) { await this.http.setSource(source); return; }
    this.source = source;
    if (!this.usingRtc) { await this.http.setSource(source); return; }
    const epoch = this.epoch;
    this.sourceStarting = true;
    this.canvas.getContext('2d')!.clearRect(0, 0, this.canvas.width, this.canvas.height);
    (this.output?.getVideoTracks()[0] as CanvasCaptureMediaStreamTrack | undefined)?.requestFrame?.();
    try {
      const track = await this.rtc.start(this.original, source);
      if (epoch === this.epoch && generation === this.sourceGeneration) { this.lastFrame = performance.now(); await this.bind(track, epoch); }
    } catch { if (generation === this.sourceGeneration) await this.fallback(epoch); }
    finally { if (generation === this.sourceGeneration) this.sourceStarting = false; }
  }

  public stop() {
    this.streaming = false; this.epoch++;
    this.sourceGeneration++;
    this.sourceStarting = false;
    if (this.watchdog !== null) window.clearInterval(this.watchdog); this.watchdog = null;
    if (this.frameRequest !== null) {
      if (typeof this.video.cancelVideoFrameCallback === 'function') this.video.cancelVideoFrameCallback(this.frameRequest);
      else window.clearTimeout(this.frameRequest);
    }
    this.frameRequest = null;
    this.rtc.onFailure = null; this.rtc.stop(); this.http.stop();
    this.output?.getTracks().forEach(track => track.stop()); this.output = null;
    this.video.pause(); this.video.srcObject = null;
    this.usingRtc = false; this.original = null; this.source = null;
  }
}
