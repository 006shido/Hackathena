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
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: video ? {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30 },
          facingMode: 'user',
        } : false,
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
    } catch (err: unknown) {
      console.warn('Initial getUserMedia failed, attempting fallback...', err);
      const error = err as Error;

      const isPermissionDenied = error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError';
      const isNotFound = error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError';

      // If both were requested and it failed, try audio only if video wasn't permitted or found
      if (video && audio) {
        try {
          const audioOnlyStream = await navigator.mediaDevices.getUserMedia({ audio: true });
          return {
            stream: audioOnlyStream,
            hasVideo: false,
            hasAudio: true,
            cameraDenied: isPermissionDenied,
            error: isPermissionDenied
              ? 'Camera access is required for video calling. Audio-only mode enabled.'
              : 'Webcam not found. Continuing with audio only.',
          };
        } catch (audioErr: unknown) {
          const aErr = audioErr as Error;
          const micDenied = aErr.name === 'NotAllowedError' || aErr.name === 'PermissionDeniedError';
          return {
            stream: null,
            hasVideo: false,
            hasAudio: false,
            cameraDenied: isPermissionDenied,
            micDenied,
            error: micDenied
              ? 'Microphone access is required for audio calling.'
              : 'Unable to access camera or microphone. Please check system permissions.',
          };
        }
      }

      return {
        stream: null,
        hasVideo: false,
        hasAudio: false,
        cameraDenied: video && isPermissionDenied,
        micDenied: audio && isPermissionDenied,
        error: isPermissionDenied
          ? (video ? 'Camera access is required for video calling.' : 'Microphone access is required for audio calling.')
          : (isNotFound ? 'Requested media device not found.' : 'Failed to acquire media stream: ' + error.message),
      };
    }
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
        track.stop();
      } catch (err) {
        console.error('Error stopping track', err);
      }
    });
  },
};
