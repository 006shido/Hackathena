export interface MediaAccessResult {
  stream: MediaStream | null;
  hasVideo: boolean;
  hasAudio: boolean;
  error?: string;
  cameraDenied?: boolean;
  micDenied?: boolean;
}

// Global registry of all active media tracks and streams to guarantee zero hardware camera/mic leaks
const activeTracks = new Set<MediaStreamTrack>();
const activeStreams = new Set<MediaStream>();
let mediaSessionToken = 0;

function registerStream(stream: MediaStream | null): void {
  if (!stream) return;
  activeStreams.add(stream);
  stream.getTracks().forEach((track) => {
    activeTracks.add(track);
    const onEnded = () => {
      activeTracks.delete(track);
      track.removeEventListener('ended', onEnded);
    };
    track.addEventListener('ended', onEnded);
  });
}

export const mediaService = {
  /**
   * Invalidate the current media acquisition session. Any in-flight getUserMedia
   * calls will immediately terminate their tracks upon resolving, and all existing
   * tracks will be stopped.
   */
  invalidateMediaSession(): void {
    mediaSessionToken++;
    this.stopAllMedia();
  },

  /**
   * Create an animated synthetic video stream for testing or when hardware cameras
   * are absent or blocked (e.g. desktop PCs with no webcam or insecure LAN HTTP).
   */
  createSyntheticStream(username = 'Desktop User'): MediaStream {
    if (typeof document === 'undefined') {
      return new MediaStream();
    }
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext('2d');

    let frame = 0;
    const draw = () => {
      if (!ctx) return;
      frame++;
      // Background gradient
      const grad = ctx.createLinearGradient(0, 0, 640, 480);
      grad.addColorStop(0, '#0f172a');
      grad.addColorStop(1, '#1e293b');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, 640, 480);

      // Avatar circle
      ctx.fillStyle = '#2563eb';
      ctx.beginPath();
      ctx.arc(320, 210, 65, 0, Math.PI * 2);
      ctx.fill();

      // Initials
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 40px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      const initials = username.split(' ').map((n) => n[0]).join('').toUpperCase().slice(0, 2) || 'U';
      ctx.fillText(initials, 320, 210);

      // Username text
      ctx.fillStyle = '#f8fafc';
      ctx.font = '600 18px sans-serif';
      ctx.fillText(username, 320, 305);

      // Subtitle
      ctx.fillStyle = '#94a3b8';
      ctx.font = '13px sans-serif';
      ctx.fillText('Webcam Offline / Simulated Video', 320, 335);

      // Pulsing green live indicator
      const radius = 5 + Math.sin(frame * 0.08) * 2;
      ctx.fillStyle = '#22c55e';
      ctx.beginPath();
      ctx.arc(40, 40, radius, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = '#86efac';
      ctx.font = '11px monospace';
      ctx.textAlign = 'left';
      ctx.fillText('LIVE STREAM', 55, 43);
    };

    draw();
    const interval = setInterval(draw, 66); // ~15 fps
    const canvasStream = canvas.captureStream(15);
    const videoTrack = canvasStream.getVideoTracks()[0];

    // Silent audio track via AudioContext destination
    let audioTrack: MediaStreamTrack | null = null;
    try {
      const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      if (AudioCtx) {
        const audioCtx = new AudioCtx();
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        gain.gain.value = 0; // silent
        const dst = audioCtx.createMediaStreamDestination();
        osc.connect(gain);
        gain.connect(dst);
        osc.start();
        audioTrack = dst.stream.getAudioTracks()[0] || null;
      }
    } catch (_) {}

    const tracks: MediaStreamTrack[] = [];
    if (videoTrack) tracks.push(videoTrack);
    if (audioTrack) tracks.push(audioTrack);
    const stream = new MediaStream(tracks);

    videoTrack?.addEventListener('ended', () => {
      clearInterval(interval);
    });

    return stream;
  },

  /**
   * Request user media with graceful fallback and strict session lifecycle checks.
   */
  async getUserMedia(video = true, audio = true): Promise<MediaAccessResult> {
    const currentToken = mediaSessionToken;

    const checkCancelled = (stream: MediaStream | null): boolean => {
      if (mediaSessionToken !== currentToken) {
        if (stream) {
          stream.getTracks().forEach((track) => {
            try {
              track.enabled = false;
              track.stop();
            } catch (e) {}
          });
        }
        return true;
      }
      return false;
    };

    // Safe check for browser mediaDevices support (handles insecure HTTP on LAN)
    if (typeof navigator === 'undefined' || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      const isHttp = typeof window !== 'undefined' &&
        window.location.protocol === 'http:' &&
        window.location.hostname !== 'localhost' &&
        window.location.hostname !== '127.0.0.1';

      const errorMsg = isHttp
        ? `Camera access requires HTTPS or localhost. Browsers block webcam on non-localhost HTTP (${window.location.hostname}). Using simulated video stream so call can proceed.`
        : 'Webcam API not supported in this browser. Using simulated video stream.';

      console.warn('[mediaService]', errorMsg);
      const fallbackStream = this.createSyntheticStream('Desktop User');
      registerStream(fallbackStream);
      return {
        stream: fallbackStream,
        hasVideo: true,
        hasAudio: true,
        error: errorMsg,
      };
    }

    // 1. Try high-definition video if video requested
    if (video) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            frameRate: { ideal: 30 },
          },
          audio: audio ? {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          } : false,
        });

        if (checkCancelled(stream)) {
          return { stream: null, hasVideo: false, hasAudio: false };
        }

        registerStream(stream);
        return {
          stream,
          hasVideo: stream.getVideoTracks().length > 0,
          hasAudio: stream.getAudioTracks().length > 0,
        };
      } catch (hdErr) {
        console.warn('[mediaService] Ideal 720p constraints failed, attempting basic video+audio...', hdErr);
      }

      // 2. Try standard unconstrained video
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: audio ? {
            echoCancellation: true,
            noiseSuppression: true,
          } : false,
        });

        if (checkCancelled(stream)) {
          return { stream: null, hasVideo: false, hasAudio: false };
        }

        registerStream(stream);
        return {
          stream,
          hasVideo: stream.getVideoTracks().length > 0,
          hasAudio: stream.getAudioTracks().length > 0,
        };
      } catch (basicErr) {
        console.warn('[mediaService] Basic video+audio failed, checking error type...', basicErr);
        const err = basicErr as Error;
        const isPermissionDenied = err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError';
        const isNotFound = err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError';

        // 3. If audio was also requested, try video-only before giving up on video
        if (audio) {
          try {
            const videoOnlyStream = await navigator.mediaDevices.getUserMedia({ video: true });
            if (checkCancelled(videoOnlyStream)) {
              return { stream: null, hasVideo: false, hasAudio: false };
            }
            registerStream(videoOnlyStream);
            return {
              stream: videoOnlyStream,
              hasVideo: true,
              hasAudio: false,
              micDenied: true,
              error: 'Microphone unavailable. Video-only mode active.',
            };
          } catch (videoOnlyErr) {
            console.warn('[mediaService] Video-only acquisition failed:', videoOnlyErr);
          }

          // 4. If video is genuinely denied or unavailable, fallback to real audio + simulated video
          try {
            const audioOnlyStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            if (checkCancelled(audioOnlyStream)) {
              return { stream: null, hasVideo: false, hasAudio: false };
            }
            const synStream = this.createSyntheticStream('Desktop User');
            const synVideo = synStream.getVideoTracks()[0];
            const realAudio = audioOnlyStream.getAudioTracks()[0];
            const combinedStream = new MediaStream([synVideo, realAudio]);
            registerStream(combinedStream);
            return {
              stream: combinedStream,
              hasVideo: true,
              hasAudio: true,
              cameraDenied: isPermissionDenied,
              error: isPermissionDenied
                ? 'Camera permission denied. Audio active with simulated video stream.'
                : 'Webcam device not found. Audio active with simulated video stream.',
            };
          } catch (audioErr) {
            console.warn('[mediaService] Audio acquisition failed:', audioErr);
          }
        }

        // 5. Final fallback when hardware is unavailable: simulated stream so WebRTC never fails
        const synOnly = this.createSyntheticStream('Desktop User');
        registerStream(synOnly);
        return {
          stream: synOnly,
          hasVideo: true,
          hasAudio: false,
          cameraDenied: isPermissionDenied,
          error: isPermissionDenied
            ? 'Camera access denied. Using simulated video stream.'
            : (isNotFound ? 'Webcam device not found. Using simulated video stream.' : 'Webcam error: ' + err.message + '. Using simulated stream.'),
        };
      }
    }

    // Audio-only requested
    if (audio) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: false,
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
          },
        });
        if (checkCancelled(stream)) {
          return { stream: null, hasVideo: false, hasAudio: false };
        }
        registerStream(stream);
        return {
          stream,
          hasVideo: false,
          hasAudio: stream.getAudioTracks().length > 0,
        };
      } catch (audioErr) {
        const aErr = audioErr as Error;
        const micDenied = aErr.name === 'NotAllowedError' || aErr.name === 'PermissionDeniedError';
        return {
          stream: null,
          hasVideo: false,
          hasAudio: false,
          micDenied,
          error: micDenied
            ? 'Microphone access denied. Please allow microphone in browser.'
            : 'Microphone not found or in use.',
        };
      }
    }

    return {
      stream: null,
      hasVideo: false,
      hasAudio: false,
    };
  },

  async getScreenMedia(): Promise<MediaStream | null> {
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: true,
        audio: true,
      });
      registerStream(stream);
      return stream;
    } catch (err) {
      console.warn('Screen share cancelled or failed:', err);
      return null;
    }
  },

  stopStream(stream: MediaStream | null): void {
    if (!stream) return;
    stream.getTracks().forEach((track) => {
      try {
        track.enabled = false;
        track.stop();
      } catch (err) {
        console.error('Error stopping track', err);
      }
      activeTracks.delete(track);
    });
    activeStreams.delete(stream);
  },

  /**
   * Forcibly terminate and release ALL camera, screen, and audio hardware tracks
   * across the entire application and flush all video element references.
   */
  stopAllMedia(): void {
    // 1. Release video elements across DOM to unlock WebKit/Safari AVFoundation capture session
    if (typeof document !== 'undefined') {
      try {
        const videoElements = document.querySelectorAll('video');
        videoElements.forEach((vid) => {
          try {
            if (vid.srcObject instanceof MediaStream) {
              vid.srcObject.getTracks().forEach((track) => {
                try {
                  track.enabled = false;
                  track.stop();
                } catch (e) {}
              });
            }
            vid.pause();
            vid.srcObject = null;
            vid.load();
          } catch (e) {}
        });
      } catch (e) {}
    }

    // 2. Stop all globally registered individual tracks
    activeTracks.forEach((track) => {
      try {
        track.enabled = false;
        track.stop();
      } catch (err) {
        console.error('[mediaService] Error stopping registered track:', err);
      }
    });
    activeTracks.clear();

    // 3. Stop all registered streams
    activeStreams.forEach((stream) => {
      try {
        stream.getTracks().forEach((track) => {
          try {
            track.enabled = false;
            track.stop();
          } catch (e) {}
        });
      } catch (e) {}
    });
    activeStreams.clear();
  },
};
