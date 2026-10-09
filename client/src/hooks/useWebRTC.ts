import { useCallback,useRef,useState } from 'react';
import { signalingService } from '../services/signaling';

const turnUrls = String(import.meta.env.VITE_TURN_URLS || '')
  .split(',').map((url) => url.trim()).filter(Boolean);
if (turnUrls.some((url) => !/^turns?:[^\s]+$/i.test(url))) {
  throw new Error('VITE_TURN_URLS must contain comma-separated TURN URLs.');
}
const turnUsername = String(import.meta.env.VITE_TURN_USERNAME || '');
const turnCredential = String(import.meta.env.VITE_TURN_CREDENTIAL || '');
if (turnUrls.length && (!turnUsername || !turnCredential)) {
  throw new Error('Configured TURN relay requires a username and credential.');
}
const relayOnly = import.meta.env.VITE_WEBRTC_RELAY_ONLY === 'true';
if (relayOnly && !turnUrls.length) {
  throw new Error('Relay-only calls require a configured TURN server.');
}

const RTC_CONFIG: RTCConfiguration = {
  iceServers: [
    { urls: 'stun:stun.l.google.com:19302' },
    { urls: 'stun:stun1.l.google.com:19302' },
    { urls: 'stun:stun2.l.google.com:19302' },
    { urls: 'stun:stun3.l.google.com:19302' },
    { urls: 'stun:stun4.l.google.com:19302' },
    { urls: 'stun:stun.cloudflare.com:3478' },
    ...(turnUrls.length ? [{ urls: turnUrls, username: turnUsername, credential: turnCredential }] : []),
  ],
  iceTransportPolicy: relayOnly ? 'relay' : 'all',
  iceCandidatePoolSize: 0,
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

  const attachLocalTracks = (pc: RTCPeerConnection, stream: MediaStream) => {
    const senders = pc.getSenders();
    stream.getTracks().forEach((track) => {
      const matchingSender = senders.find((s) => s.track?.kind === track.kind);
      if (matchingSender) {
        if (matchingSender.track !== track) {
          matchingSender.replaceTrack(track).catch((err) => {
            console.warn(`[WebRTC] replaceTrack error for ${track.kind}:`, err);
          });
        }
      } else {
        try {
          pc.addTrack(track, stream);
        } catch (err) {
          console.warn(`[WebRTC] Could not add track ${track.kind}:`, err);
        }
      }
    });
  };

  const drainQueuedCandidates = async (pc: RTCPeerConnection) => {
    if (queuedCandidates.current.length === 0) return;
    while (queuedCandidates.current.length > 0) {
      const cand = queuedCandidates.current.shift();
      if (cand && cand.candidate) {
        try {
          await pc.addIceCandidate(new RTCIceCandidate(cand));
        } catch (e) {
          console.warn('[WebRTC] Failed to add queued ICE candidate', e);
        }
      }
    }
  };

  const createPeerConnection = useCallback((
    roomId: string,
    localStream: MediaStream | null,
    forceRecreate = false
  ): RTCPeerConnection => {
    // If peer connection already exists and is active, reuse it unless forceRecreate is specified
    if (pcRef.current && !forceRecreate && pcRef.current.signalingState !== 'closed') {
      if (localStream) {
        attachLocalTracks(pcRef.current, localStream);
      }
      return pcRef.current;
    }

    if (pcRef.current) {
      try {
        pcRef.current.close();
      } catch {}
      pcRef.current = null;
    }
    const pc = new RTCPeerConnection(RTC_CONFIG);
    pcRef.current = pc;
    remoteStreamInstanceRef.current = new MediaStream();

    // Attach local tracks only for initiator (receiver attaches after setRemoteDescription)
    if (localStream && isInitiator.current) {
      attachLocalTracks(pc, localStream);
    }

    // Handle remote track arrivals
    pc.ontrack = (event) => {

      if (!remoteStreamInstanceRef.current.getTrackById(event.track.id)) {
        remoteStreamInstanceRef.current.addTrack(event.track);
      }

      if (event.streams && event.streams[0]) {
        event.streams[0].getTracks().forEach((track) => {
          if (!remoteStreamInstanceRef.current.getTrackById(track.id)) {
            remoteStreamInstanceRef.current.addTrack(track);
          }
        });
      }

      event.track.onunmute = () => {
        const tracks = remoteStreamInstanceRef.current.getTracks();
        setRemoteStream(new MediaStream(tracks));
      };

      const allTracks = remoteStreamInstanceRef.current.getTracks();
      setRemoteStream(new MediaStream(allTracks));
    };

    // Handle ICE candidates generated locally
    pc.onicecandidate = (event) => {
      if (event.candidate && event.candidate.candidate) {
        signalingService.sendIceCandidate(roomId, event.candidate.toJSON());
      }
    };

    // Monitor connection states
    pc.onconnectionstatechange = () => {
      setConnectionState(pc.connectionState);
    };

    pc.oniceconnectionstatechange = () => {
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
        attachLocalTracks(pc, localStream);
      }

      // Ensure transceivers exist for both audio and video
      const kinds = pc.getSenders().map((s) => s.track?.kind);
      if (!kinds.includes('audio')) {
        pc.addTransceiver('audio', { direction: 'sendrecv' });
      }
      if (!kinds.includes('video')) {
        pc.addTransceiver('video', { direction: 'sendrecv' });
      }
      const offer = await pc.createOffer({
        offerToReceiveAudio: true,
        offerToReceiveVideo: true,
      });
      await pc.setLocalDescription(offer);
      signalingService.sendOffer(roomId, offer);
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
      pc = createPeerConnection(roomId, localStream || null, true);
    }

    try {
      isInitiator.current = false;

      // Handle offer collision / glare
      if (pc.signalingState !== 'stable') {
        console.warn('[WebRTC] Offer received while in non-stable state, rolling back local description');
        await pc.setLocalDescription({ type: 'rollback' });
      }

      // 1. Set Remote Description FIRST from offer
      await pc.setRemoteDescription(new RTCSessionDescription(sdp));

      // 2. Attach local tracks to match offer's transceivers
      if (localStream) {
        attachLocalTracks(pc, localStream);
      }

      // 3. Process queued ICE candidates
      await drainQueuedCandidates(pc);

      // 4. Create and send answer
      const answer = await pc.createAnswer();
      await pc.setLocalDescription(answer);
      signalingService.sendAnswer(roomId, answer);
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
      if (pc.signalingState === 'have-local-offer') {
        await pc.setRemoteDescription(new RTCSessionDescription(sdp));
        await drainQueuedCandidates(pc);
      } else {
        console.warn('[WebRTC] Skipping handleAnswer as signalingState is', pc.signalingState);
      }
    } catch (err) {
      console.error('[WebRTC] Error handling answer:', err);
    }
  }, []);

  const handleIceCandidate = useCallback(async (candidate: RTCIceCandidateInit): Promise<void> => {
    if (!candidate || !candidate.candidate) return;
    const pc = pcRef.current;

    if (!pc || !pc.remoteDescription || !pc.remoteDescription.type) {
      queuedCandidates.current.push(candidate);
      return;
    }

    try {
      await pc.addIceCandidate(new RTCIceCandidate(candidate));
    } catch (err) {
      console.error('[WebRTC] Error adding ICE candidate:', err);
    }
  }, []);

  const replaceVideoTrack = useCallback(async (newTrack: MediaStreamTrack | null): Promise<boolean> => {
    const pc = pcRef.current;
    if (!pc) return false;

    const sender = pc.getSenders().find((s) => s.track?.kind === 'video') ||
      pc.getTransceivers().find((t) => t.receiver.track?.kind === 'video')?.sender;

    if (sender) {
      try {
        await sender.replaceTrack(newTrack);
        return true;
      } catch (err) {
        console.error('[WebRTC] Failed to replace video track:', err);
        return false;
      }
    } else if (newTrack) {
      try {
        pc.addTrack(newTrack);
        return true;
      } catch (err) {
        console.error('[WebRTC] Failed to add video track:', err);
        return false;
      }
    }
    return false;
  }, []);

  const replaceAudioTrack = useCallback(async (newTrack: MediaStreamTrack | null): Promise<boolean> => {
    const pc = pcRef.current;
    if (!pc) return false;

    const sender = pc.getSenders().find((s) => s.track?.kind === 'audio') ||
      pc.getTransceivers().find((t) => t.receiver.track?.kind === 'audio')?.sender;

    if (sender) {
      try {
        await sender.replaceTrack(newTrack);
        return true;
      } catch (err) {
        console.error('[WebRTC] Failed to replace audio track:', err);
        return false;
      }
    } else if (newTrack) {
      try {
        pc.addTrack(newTrack);
        return true;
      } catch (err) {
        console.error('[WebRTC] Failed to add audio track:', err);
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

    if (remoteStreamInstanceRef.current) {
      remoteStreamInstanceRef.current.getTracks().forEach((track) => {
        try {
          track.enabled = false;
          track.stop();
        } catch {}
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
