export interface MediaAccessResult {
  stream: MediaStream | null;
  hasVideo: boolean;
  hasAudio: boolean;
  error?: string;
  cameraDenied?: boolean;
  micDenied?: boolean;
}

export const mediaService = {
  /**
   * Request user media with graceful fallback:
   * 1. Try audio + video
   * 2. If video fails (e.g. permission or no webcam), try audio-only
   */
  async getUserMedia(video = true, audio = true): Promise<MediaAccessResult> {
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

          // 4. If video is genuinely denied or unavailable, fallback to audio-only
          try {
            const audioOnlyStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            return {
              stream: audioOnlyStream,
              hasVideo: false,
              hasAudio: true,
              cameraDenied: isPermissionDenied,
              error: isPermissionDenied
                ? 'Camera permission denied. Please allow camera in browser address bar.'
                : 'Webcam device not found. Continuing with audio only.',
            };
          } catch (audioErr) {
            const aErr = audioErr as Error;
            const micDenied = aErr.name === 'NotAllowedError' || aErr.name === 'PermissionDeniedError';
            return {
              stream: null,
              hasVideo: false,
              hasAudio: false,
              cameraDenied: isPermissionDenied,
              micDenied,
              error: 'Unable to access camera or microphone. Please check system permissions.',
            };
          }
        }

        return {
          stream: null,
          hasVideo: false,
          hasAudio: false,
          cameraDenied: isPermissionDenied,
          error: isPermissionDenied
            ? 'Camera access denied. Please allow camera in browser address bar.'
            : (isNotFound ? 'Webcam device not found.' : 'Webcam error: ' + err.message),
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
    });
  },
};
