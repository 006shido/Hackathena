import { useRef, useCallback, useState } from 'react';
import { signalingService } from '../services/signaling';

const RTC_CONFIG: RTCConfiguration = {
  iceServers: [
    { urls: 'stun:stun.l.google.com:19302' },
    { urls: 'stun:stun1.l.google.com:19302' },
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
      pcRef.current.close();
      pcRef.current = null;
    }

    const pc = new RTCPeerConnection(RTC_CONFIG);
    pcRef.current = pc;
    queuedCandidates.current = [];

    // Attach local stream tracks to RTCPeerConnection
    if (localStream) {
      localStream.getTracks().forEach((track) => {
        try {
          pc.addTrack(track, localStream);
        } catch (err) {
          console.warn('Could not add track to RTCPeerConnection', err);
        }
      });
    }

    // Handle remote track
    pc.ontrack = (event) => {
      console.log('[WebRTC] Received remote track:', event.track.kind);
      const incomingStream = event.streams[0] || new MediaStream([event.track]);
      setRemoteStream(incomingStream);
      onRemoteStreamReceived?.(incomingStream);
    };

    // Handle ICE candidates generated locally
    pc.onicecandidate = (event) => {
      if (event.candidate) {
        signalingService.sendIceCandidate(roomId, event.candidate.toJSON());
      }
    };

    // Monitor connection state
    pc.onconnectionstatechange = () => {
      console.log('[WebRTC] Connection state change:', pc.connectionState);
      setConnectionState(pc.connectionState);
    };

    pc.oniceconnectionstatechange = () => {
      console.log('[WebRTC] ICE connection state change:', pc.iceConnectionState);
    };

    return pc;
  }, []);

  const makeOffer = useCallback(async (roomId: string): Promise<void> => {
    const pc = pcRef.current;
    if (!pc) return;

    try {
      isInitiator.current = true;
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
    if (!pc) return;

    try {
      isInitiator.current = false;
      await pc.setRemoteDescription(new RTCSessionDescription(sdp));

      // Process queued candidates
      while (queuedCandidates.current.length > 0) {
        const cand = queuedCandidates.current.shift();
        if (cand) {
          await pc.addIceCandidate(new RTCIceCandidate(cand));
        }
      }

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
    if (!pc) return;

    try {
      await pc.setRemoteDescription(new RTCSessionDescription(sdp));
      console.log('[WebRTC] Remote description set from answer');

      // Process queued candidates
      while (queuedCandidates.current.length > 0) {
        const cand = queuedCandidates.current.shift();
        if (cand) {
          await pc.addIceCandidate(new RTCIceCandidate(cand));
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
      pcRef.current.close();
      pcRef.current = null;
    }
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
