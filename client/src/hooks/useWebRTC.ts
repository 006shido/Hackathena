import { useRef, useCallback, useState } from 'react';
import { signalingService } from '../services/signaling';

const RTC_CONFIG: RTCConfiguration = {
  iceServers: [
    { urls: 'stun:stun.l.google.com:19302' },
    { urls: 'stun:stun1.l.google.com:19302' },
    { urls: 'stun:stun2.l.google.com:19302' },
    { urls: 'stun:stun3.l.google.com:19302' },
    { urls: 'stun:stun4.l.google.com:19302' },
    { urls: 'stun:stun.cloudflare.com:3478' },
    { urls: 'stun:stun.nextcloud.com:443' },
    { urls: 'stun:stun.nextcloud.com:3478' },
    { urls: 'stun:global.stun.twilio.com:3478' },
  ],
  iceCandidatePoolSize: 10,
};

export type WebRTCConnectionState =
  | 'new'
  | 'connecting'
  | 'connected'
  | 'disconnected'
  | 'failed'
  | 'closed';

export function useWebRTC() {
  const pcRef = useRef<RTCPeerConnection | null>(null);
  const [remoteStream, setRemoteStream] = useState<MediaStream | null>(null);
  const remoteStreamInstanceRef = useRef<MediaStream>(new MediaStream());
  const [connectionState, setConnectionState] = useState<WebRTCConnectionState>('new');
  const [iceState, setIceState] = useState<RTCIceConnectionState>('new');
  const queuedCandidates = useRef<RTCIceCandidateInit[]>([]);
  const isInitiator = useRef<boolean>(false);

  const createPeerConnection = useCallback((
    roomId: string,
    localStream: MediaStream | null,
    forceRecreate = false
  ): RTCPeerConnection => {
    // If peer connection already exists and is active, reuse it and ensure local tracks are attached
    if (pcRef.current && !forceRecreate && pcRef.current.signalingState !== 'closed') {
      console.log('[WebRTC] Reusing active RTCPeerConnection, signalingState:', pcRef.current.signalingState);
      if (localStream) {
        const senders = pcRef.current.getSenders();
        localStream.getTracks().forEach((track) => {
          const matchingSender = senders.find((s) => s.track?.kind === track.kind);
          if (matchingSender) {
            if (matchingSender.track !== track) {
              matchingSender.replaceTrack(track).catch(() => {});
            }
          } else {
            try {
              pcRef.current?.addTrack(track, localStream);
              console.log(`[WebRTC] Attached missing local track: ${track.kind} (${track.id})`);
            } catch (err) {
              console.warn('[WebRTC] Could not add track to existing RTCPeerConnection', err);
            }
          }
        });
      }
      return pcRef.current;
    }

    // Clean up previous connection if closing
    if (pcRef.current) {
      console.log('[WebRTC] Closing previous RTCPeerConnection');
      try {
        pcRef.current.close();
      } catch {
        // Ignored
      }
      pcRef.current = null;
    }

    console.log('[WebRTC] Initializing new RTCPeerConnection with high-availability STUN servers...');
    const pc = new RTCPeerConnection(RTC_CONFIG);
    pcRef.current = pc;
    queuedCandidates.current = [];
    remoteStreamInstanceRef.current = new MediaStream();

    // Attach local tracks
    if (localStream) {
      const tracks = localStream.getTracks();
      console.log(`[WebRTC] Attaching ${tracks.length} local tracks to new connection`);
      tracks.forEach((track) => {
        try {
          pc.addTrack(track, localStream);
          console.log(`[WebRTC] Attached track: ${track.kind} (${track.id})`);
        } catch (err) {
          console.warn('[WebRTC] Could not add track to RTCPeerConnection', err);
        }
      });
    }

    // Handle remote track arrivals
    pc.ontrack = (event) => {
      console.log('[WebRTC] Received remote track:', event.track.kind, event.track.id);
      
      const targetStream = (event.streams && event.streams[0])
        ? event.streams[0]
        : remoteStreamInstanceRef.current;

      if (!targetStream.getTrackById(event.track.id)) {
        targetStream.addTrack(event.track);
      }
      if (!remoteStreamInstanceRef.current.getTrackById(event.track.id)) {
        remoteStreamInstanceRef.current.addTrack(event.track);
      }

      // Create a fresh MediaStream instance so React state change triggers component re-render
      const freshStream = new MediaStream(targetStream.getTracks());
      setRemoteStream(freshStream);
    };

    // Handle ICE candidates generated locally
    pc.onicecandidate = (event) => {
      if (event.candidate) {
        console.log('[WebRTC] Generated ICE candidate:', event.candidate.type, event.candidate.protocol);
        signalingService.sendIceCandidate(roomId, event.candidate.toJSON());
      }
    };

    // Monitor connection states
    pc.onconnectionstatechange = () => {
      console.log('[WebRTC] Connection state change:', pc.connectionState);
      setConnectionState(pc.connectionState);
    };

    pc.oniceconnectionstatechange = () => {
      console.log('[WebRTC] ICE connection state change:', pc.iceConnectionState);
      setIceState(pc.iceConnectionState);
    };

    return pc;
  }, []);

  const makeOffer = useCallback(async (roomId: string, localStream?: MediaStream | null): Promise<void> => {
    const pc = pcRef.current;
    if (!pc) {
      console.warn('[WebRTC] Cannot make offer: No RTCPeerConnection');
      return;
    }

    try {
      isInitiator.current = true;
      if (localStream) {
        const senders = pc.getSenders();
        localStream.getTracks().forEach((track) => {
          const matchingSender = senders.find((s) => s.track?.kind === track.kind);
          if (matchingSender) {
            if (matchingSender.track !== track) {
              matchingSender.replaceTrack(track).catch(() => {});
            }
          } else {
            try {
              pc.addTrack(track, localStream);
            } catch (err) {
              console.warn('[WebRTC] makeOffer addTrack error:', err);
            }
          }
        });
      }

      console.log('[WebRTC] Creating offer...');
      const offer = await pc.createOffer({
        offerToReceiveAudio: true,
        offerToReceiveVideo: true,
      });
      await pc.setLocalDescription(offer);
      signalingService.sendOffer(roomId, offer);
      console.log('[WebRTC] Sent offer to peer');
    } catch (err) {
      console.error('[WebRTC] Error creating offer:', err);
    }
  }, []);

  const handleOffer = useCallback(async (
    roomId: string,
    sdp: RTCSessionDescriptionInit,
    localStream?: MediaStream | null
  ): Promise<void> => {
    let pc = pcRef.current;
    if (!pc || pc.signalingState === 'closed') {
      pc = createPeerConnection(roomId, localStream || null);
    }

    try {
      isInitiator.current = false;
      console.log('[WebRTC] Handling offer in state:', pc.signalingState);

      // Ensure local tracks are attached before answering!
      if (localStream) {
        const senders = pc.getSenders();
        localStream.getTracks().forEach((track) => {
          const matchingSender = senders.find((s) => s.track?.kind === track.kind);
          if (matchingSender) {
            if (matchingSender.track !== track) {
              matchingSender.replaceTrack(track).catch(() => {});
            }
          } else {
            try {
              pc?.addTrack(track, localStream);
              console.log(`[WebRTC] handleOffer attached local track: ${track.kind}`);
            } catch (err) {
              console.warn('[WebRTC] handleOffer addTrack error:', err);
            }
          }
        });
      }

      // Handle offer collision / glare with sequential rollback
      if (pc.signalingState !== 'stable') {
        console.warn('[WebRTC] Offer received while in non-stable state, rolling back local description');
        await pc.setLocalDescription({ type: 'rollback' });
      }

      await pc.setRemoteDescription(new RTCSessionDescription(sdp));

      // Process any queued candidates
      while (queuedCandidates.current.length > 0) {
        const cand = queuedCandidates.current.shift();
        if (cand && cand.candidate) {
          try {
            await pc.addIceCandidate(cand);
          } catch (e) {
            console.warn('[WebRTC] Failed to add queued ICE candidate', e);
          }
        }
      }

      console.log('[WebRTC] Creating answer...');
      const answer = await pc.createAnswer();
      await pc.setLocalDescription(answer);
      signalingService.sendAnswer(roomId, answer);
      console.log('[WebRTC] Sent answer to peer');
    } catch (err) {
      console.error('[WebRTC] Error handling offer and sending answer:', err);
    }
  }, [createPeerConnection]);

  const handleAnswer = useCallback(async (sdp: RTCSessionDescriptionInit): Promise<void> => {
    const pc = pcRef.current;
    if (!pc) {
      console.warn('[WebRTC] Cannot handle answer: No RTCPeerConnection');
      return;
    }

    try {
      console.log('[WebRTC] Setting remote description from answer');
      await pc.setRemoteDescription(new RTCSessionDescription(sdp));
      console.log('[WebRTC] Remote description successfully set from answer');

      // Process queued candidates
      while (queuedCandidates.current.length > 0) {
        const cand = queuedCandidates.current.shift();
        if (cand && cand.candidate) {
          try {
            await pc.addIceCandidate(cand);
          } catch (e) {
            console.warn('[WebRTC] Failed to add queued ICE candidate', e);
          }
        }
      }
    } catch (err) {
      console.error('[WebRTC] Error handling answer:', err);
    }
  }, []);

  const handleIceCandidate = useCallback(async (candidate: RTCIceCandidateInit): Promise<void> => {
    const pc = pcRef.current;
    if (!pc || !candidate || !candidate.candidate) return;

    try {
      if (pc.remoteDescription && pc.remoteDescription.type) {
        await pc.addIceCandidate(candidate);
      } else {
        queuedCandidates.current.push(candidate);
      }
    } catch (err) {
      console.error('[WebRTC] Error adding ICE candidate:', err);
    }
  }, []);

  const replaceVideoTrack = useCallback(async (newTrack: MediaStreamTrack | null): Promise<boolean> => {
    const pc = pcRef.current;
    if (!pc) return false;

    const sender = pc.getSenders().find((s) => s.track?.kind === 'video');
    if (sender) {
      try {
        await sender.replaceTrack(newTrack);
        console.log('[WebRTC] Replaced video track with:', newTrack ? newTrack.id : 'null');
        return true;
      } catch (err) {
        console.error('[WebRTC] Failed to replace video track:', err);
        return false;
      }
    }
    return false;
  }, []);

  const replaceAudioTrack = useCallback(async (newTrack: MediaStreamTrack | null): Promise<boolean> => {
    const pc = pcRef.current;
    if (!pc) return false;

    const sender = pc.getSenders().find((s) => s.track?.kind === 'audio');
    if (sender) {
      try {
        await sender.replaceTrack(newTrack);
        console.log('[WebRTC] Replaced audio track with:', newTrack ? newTrack.id : 'null');
        return true;
      } catch (err) {
        console.error('[WebRTC] Failed to replace audio track:', err);
        return false;
      }
    }
    return false;
  }, []);

  const cleanupPeerConnection = useCallback(() => {
    if (pcRef.current) {
      try {
        const senders = pcRef.current.getSenders();
        senders.forEach((sender) => {
          if (sender.track) {
            try {
              sender.track.enabled = false;
              sender.track.stop();
            } catch (err) {
              console.error('Error stopping sender track', err);
            }
          }
        });
      } catch (err) {
        // Ignored
      }

      try {
        pcRef.current.close();
      } catch {
        // Ignored
      }
      pcRef.current = null;
    }

    if (remoteStreamInstanceRef.current) {
      remoteStreamInstanceRef.current.getTracks().forEach((track) => {
        try {
          track.enabled = false;
          track.stop();
        } catch (e) {}
      });
      remoteStreamInstanceRef.current = new MediaStream();
    }

    setRemoteStream(null);
    setConnectionState('closed');
    queuedCandidates.current = [];
  }, []);

  return {
    pcRef,
    remoteStream,
    connectionState,
    iceState,
    createPeerConnection,
    makeOffer,
    handleOffer,
    handleAnswer,
    handleIceCandidate,
    replaceVideoTrack,
    replaceAudioTrack,
    cleanupPeerConnection,
  };
}
