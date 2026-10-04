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
  ],
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
          console.log(`[WebRTC] Attached local track: ${track.kind} (${track.id})`);
        } catch (err) {
          console.warn(`[WebRTC] Could not add track ${track.kind}:`, err);
        }
      }
    });
  };

  const drainQueuedCandidates = async (pc: RTCPeerConnection) => {
    if (queuedCandidates.current.length === 0) return;
    console.log(`[WebRTC] Draining ${queuedCandidates.current.length} queued ICE candidate(s)`);
    while (queuedCandidates.current.length > 0) {
      const cand = queuedCandidates.current.shift();
      if (cand && cand.candidate) {
        try {
          await pc.addIceCandidate(new RTCIceCandidate(cand));
          console.log('[WebRTC] Added queued ICE candidate');
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
      console.log('[WebRTC] Reusing active RTCPeerConnection, signalingState:', pcRef.current.signalingState);
      if (localStream) {
        attachLocalTracks(pcRef.current, localStream);
      }
      return pcRef.current;
    }

    if (pcRef.current) {
      console.log('[WebRTC] Closing previous RTCPeerConnection for recreation');
      try {
        pcRef.current.close();
      } catch {}
      pcRef.current = null;
    }

    console.log('[WebRTC] Initializing fresh RTCPeerConnection with STUN & TURN...');
    const pc = new RTCPeerConnection(RTC_CONFIG);
    pcRef.current = pc;
    remoteStreamInstanceRef.current = new MediaStream();

    // Attach local tracks only for initiator (receiver attaches after setRemoteDescription)
    if (localStream && isInitiator.current) {
      attachLocalTracks(pc, localStream);
    }

    // Handle remote track arrivals
    pc.ontrack = (event) => {
      console.log('[WebRTC] Received remote track:', event.track.kind, event.track.id, 'readyState:', event.track.readyState);

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
        console.log('[WebRTC] Remote track unmuted:', event.track.kind);
        const tracks = remoteStreamInstanceRef.current.getTracks();
        setRemoteStream(new MediaStream(tracks));
      };

      const allTracks = remoteStreamInstanceRef.current.getTracks();
      console.log(`[WebRTC] Remote stream updated with ${allTracks.length} tracks:`, allTracks.map((t) => `${t.kind}(${t.enabled})`));
      setRemoteStream(new MediaStream(allTracks));
    };

    // Handle ICE candidates generated locally
    pc.onicecandidate = (event) => {
      if (event.candidate && event.candidate.candidate) {
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
      pc = createPeerConnection(roomId, localStream || null, true);
    }

    try {
      isInitiator.current = false;
      console.log('[WebRTC] Handling offer in state:', pc.signalingState);

      // Handle offer collision / glare
      if (pc.signalingState !== 'stable') {
        console.warn('[WebRTC] Offer received while in non-stable state, rolling back local description');
        await pc.setLocalDescription({ type: 'rollback' });
      }

      // 1. Set Remote Description FIRST from offer
      await pc.setRemoteDescription(new RTCSessionDescription(sdp));
      console.log('[WebRTC] Remote description successfully set from offer');

      // 2. Attach local tracks to match offer's transceivers
      if (localStream) {
        attachLocalTracks(pc, localStream);
      }

      // 3. Process queued ICE candidates
      await drainQueuedCandidates(pc);

      // 4. Create and send answer
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
      console.log('[WebRTC] Setting remote description from answer, state:', pc.signalingState);
      if (pc.signalingState === 'have-local-offer') {
        await pc.setRemoteDescription(new RTCSessionDescription(sdp));
        console.log('[WebRTC] Remote description successfully set from answer');
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
      console.log('[WebRTC] Remote description not set yet, queuing ICE candidate');
      queuedCandidates.current.push(candidate);
      return;
    }

    try {
      await pc.addIceCandidate(new RTCIceCandidate(candidate));
      console.log('[WebRTC] Added ICE candidate');
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
        console.log('[WebRTC] Replaced video track with:', newTrack ? newTrack.id : 'null');
        return true;
      } catch (err) {
        console.error('[WebRTC] Failed to replace video track:', err);
        return false;
      }
    } else if (newTrack) {
      try {
        pc.addTrack(newTrack);
        console.log('[WebRTC] Added new video track to connection:', newTrack.id);
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
        console.log('[WebRTC] Replaced audio track with:', newTrack ? newTrack.id : 'null');
        return true;
      } catch (err) {
        console.error('[WebRTC] Failed to replace audio track:', err);
        return false;
      }
    } else if (newTrack) {
      try {
        pc.addTrack(newTrack);
        console.log('[WebRTC] Added new audio track to connection:', newTrack.id);
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
