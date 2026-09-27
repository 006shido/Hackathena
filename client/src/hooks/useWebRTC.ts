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
  const queuedCandidates = useRef<RTCIceCandidateInit[]>([]);
  const isInitiator = useRef<boolean>(false);

  const createPeerConnection = useCallback((
    roomId: string,
    localStream: MediaStream | null,
    onRemoteStreamReceived?: (stream: MediaStream) => void
  ): RTCPeerConnection => {
    // If peer connection exists, close it first
    if (pcRef.current) {
      console.log('[WebRTC] Closing previous PeerConnection');
      try {
        pcRef.current.close();
      } catch {
        // Ignored
      }
      pcRef.current = null;
    }

    console.log('[WebRTC] Creating new RTCPeerConnection for room:', roomId);
    const pc = new RTCPeerConnection(RTC_CONFIG);
    pcRef.current = pc;
    queuedCandidates.current = [];
    remoteStreamInstanceRef.current = new MediaStream();

    // Attach local stream tracks to RTCPeerConnection
    if (localStream) {
      const tracks = localStream.getTracks();
      console.log(`[WebRTC] Attaching ${tracks.length} local tracks to connection`);
      tracks.forEach((track) => {
        try {
          pc.addTrack(track, localStream);
          console.log(`[WebRTC] Attached track: ${track.kind} (${track.id})`);
        } catch (err) {
          console.warn('Could not add track to RTCPeerConnection', err);
        }
      });
    } else {
      console.warn('[WebRTC] createPeerConnection called with null localStream');
    }

    // Handle incoming remote tracks
    pc.ontrack = (event) => {
      console.log('[WebRTC] Received remote track:', event.track.kind, event.track.id);
      
      // Add track to accumulating MediaStream
      if (!remoteStreamInstanceRef.current.getTrackById(event.track.id)) {
        remoteStreamInstanceRef.current.addTrack(event.track);
      }

      // Clone MediaStream with latest tracks to ensure React state update
      const updatedStream = new MediaStream(remoteStreamInstanceRef.current.getTracks());
      setRemoteStream(updatedStream);
      onRemoteStreamReceived?.(updatedStream);
    };

    // Handle ICE candidates generated locally
    pc.onicecandidate = (event) => {
      if (event.candidate) {
        console.log('[WebRTC] Generated ICE candidate:', event.candidate.candidate.substring(0, 30));
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
    };

    pc.onicegatheringstatechange = () => {
      console.log('[WebRTC] ICE gathering state change:', pc.iceGatheringState);
    };

    return pc;
  }, []);

  const makeOffer = useCallback(async (roomId: string): Promise<void> => {
    const pc = pcRef.current;
    if (!pc) {
      console.warn('[WebRTC] Cannot make offer: No RTCPeerConnection');
      return;
    }

    try {
      isInitiator.current = true;
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
    sdp: RTCSessionDescriptionInit
  ): Promise<void> => {
    const pc = pcRef.current;
    if (!pc) {
      console.warn('[WebRTC] Cannot handle offer: No RTCPeerConnection');
      return;
    }

    try {
      isInitiator.current = false;
      console.log('[WebRTC] Setting remote description from offer');
      await pc.setRemoteDescription(new RTCSessionDescription(sdp));

      // Process queued candidates
      while (queuedCandidates.current.length > 0) {
        const cand = queuedCandidates.current.shift();
        if (cand) {
          try {
            await pc.addIceCandidate(new RTCIceCandidate(cand));
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
  }, []);

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
        if (cand) {
          try {
            await pc.addIceCandidate(new RTCIceCandidate(cand));
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
    if (!pc) return;

    try {
      if (pc.remoteDescription && pc.remoteDescription.type) {
        await pc.addIceCandidate(new RTCIceCandidate(candidate));
      } else {
        queuedCandidates.current.push(candidate);
      }
    } catch (err) {
      console.error('[WebRTC] Error adding ICE candidate:', err);
    }
  }, []);

  /**
   * Replace outgoing video track on the fly (for Face Swap simulation)
   */
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

  /**
   * Replace outgoing audio track on the fly (for Voice Transformation simulation)
   */
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
        pcRef.current.close();
      } catch {
        // Ignored
      }
      pcRef.current = null;
    }
    remoteStreamInstanceRef.current = new MediaStream();
    setRemoteStream(null);
    setConnectionState('closed');
    queuedCandidates.current = [];
  }, []);

  return {
    pc: pcRef.current,
    remoteStream,
    connectionState,
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
