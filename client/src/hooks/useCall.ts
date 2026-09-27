import { useState, useRef, useCallback, useEffect } from 'react';
import { User } from '../types/auth';
import { CallStatus, ParticipantInfo } from '../types/call';
import { AttackMode, AttackState } from '../types/attack';
import { signalingService } from '../services/signaling';
import { mediaService, MediaAccessResult } from '../services/media';
import { useWebRTC } from './useWebRTC';
import { FaceSimulationPipeline, FacePreset } from '../attack/faceSimulation';
import { VoiceTransformationPipeline, VoicePreset } from '../attack/voiceTransformation';

export function useCall(user: User | null, token: string | null) {
  const [roomId, setRoomId] = useState<string>('');
  const [callStatus, setCallStatus] = useState<CallStatus>('idle');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [peerInfo, setPeerInfo] = useState<ParticipantInfo | null>(null);

  // Local media state
  const [localStream, setLocalStream] = useState<MediaStream | null>(null);
  const [isMicMuted, setIsMicMuted] = useState<boolean>(false);
  const [isCameraOff, setIsCameraOff] = useState<boolean>(false);
  const [isScreenSharing, setIsScreenSharing] = useState<boolean>(false);
  const [mediaError, setMediaError] = useState<string | null>(null);

  // Attack simulator state (strictly initialized & managed for tester role)
  const isTester = user?.role === 'tester';
  const [attackState, setAttackState] = useState<AttackState>({
    faceSwap: false,
    voiceTransform: false,
    mode: 'none',
    facePreset: 'neural-clone',
    voicePreset: 'robotic-vocoder',
  });

  // Peer's attack simulation state (received by DeepTrace monitoring placeholder)
  const [peerAttackState, setPeerAttackState] = useState<{
    active: boolean;
    mode: AttackMode;
    faceSwap: boolean;
    voiceTransform: boolean;
  }>({
    active: false,
    mode: 'none',
    faceSwap: false,
    voiceTransform: false,
  });

  // Pipelines
  const facePipelineRef = useRef<FaceSimulationPipeline | null>(null);
  const voicePipelineRef = useRef<VoiceTransformationPipeline | null>(null);

  // Keep references to original tracks & stream
  const localStreamRef = useRef<MediaStream | null>(null);
  const originalVideoTrackRef = useRef<MediaStreamTrack | null>(null);
  const originalAudioTrackRef = useRef<MediaStreamTrack | null>(null);
  const screenStreamRef = useRef<MediaStream | null>(null);

  const {
    pcRef,
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
  } = useWebRTC();

  // Initialize media devices for preview or call
  const initLocalMedia = useCallback(async (video = true, audio = true): Promise<MediaAccessResult> => {
    setCallStatus('checking-devices');
    const result = await mediaService.getUserMedia(video, audio);

    if (result.stream) {
      setLocalStream(result.stream);
      localStreamRef.current = result.stream;
      originalVideoTrackRef.current = result.stream.getVideoTracks()[0] || null;
      originalAudioTrackRef.current = result.stream.getAudioTracks()[0] || null;
      setIsCameraOff(!result.hasVideo);
      setIsMicMuted(!result.hasAudio);
      setCallStatus('ready');
    } else {
      setCallStatus('idle');
    }

    if (result.error) {
      setMediaError(result.error);
    } else {
      setMediaError(null);
    }

    return result;
  }, []);

  // Join a room and initialize signaling
  const joinRoom = useCallback(async (targetRoomId: string) => {
    if (!targetRoomId.trim()) {
      setErrorMessage('Room ID is required.');
      return;
    }
    if (!token) {
      setErrorMessage('User session missing.');
      return;
    }

    const cleanRoomId = targetRoomId.toUpperCase().trim();
    setRoomId(cleanRoomId);
    setErrorMessage(null);
    setCallStatus('joining');

    // Connect to signaling server with auth token
    signalingService.connect(token, {
      onRoomJoined: (data) => {
        console.log('[useCall] Joined room:', data.roomId, 'Peers already in room:', data.peers.length);
        const activeLocalStream = localStreamRef.current || localStream;
        
        if (data.peers.length > 0) {
          // A peer is already in room: initialize connection and WAIT for their offer
          setPeerInfo(data.peers[0]);
          setCallStatus('connected');
          createPeerConnection(data.roomId, activeLocalStream);
          console.log('[useCall] Initialized receiver peer connection, awaiting offer from host...');
        } else {
          // First in room: waiting for someone to join
          setCallStatus('waiting');
          setPeerInfo(null);
          createPeerConnection(data.roomId, activeLocalStream);
        }
      },

      onPeerJoined: (peer) => {
        console.log('[useCall] Peer joined room, initiating WebRTC offer:', peer);
        const activeLocalStream = localStreamRef.current || localStream;
        setPeerInfo(peer);
        setCallStatus('connected');
        
        // Host initiates the offer to the newly joined peer
        createPeerConnection(cleanRoomId, activeLocalStream);
        makeOffer(cleanRoomId);
      },

      onPeerLeft: (data) => {
        console.log('[useCall] Peer left:', data);
        setPeerInfo(null);
        setCallStatus('waiting');
        cleanupPeerConnection();
        // Reset peer attack monitor
        setPeerAttackState({
          active: false,
          mode: 'none',
          faceSwap: false,
          voiceTransform: false,
        });
      },

      onOffer: async ({ sdp }) => {
        console.log('[useCall] Received WebRTC offer from peer');
        setCallStatus('connected');
        const activeLocalStream = localStreamRef.current || localStream;
        await handleOffer(cleanRoomId, sdp, activeLocalStream);
      },

      onAnswer: async ({ sdp }) => {
        console.log('[useCall] Received WebRTC answer from peer');
        await handleAnswer(sdp);
      },

      onIceCandidate: async ({ candidate }) => {
        await handleIceCandidate(candidate);
      },

      onRoomError: ({ message }) => {
        console.error('[useCall] Room error:', message);
        setErrorMessage(message);
        setCallStatus('failed');
      },

      onPeerAttackState: (data) => {
        console.log('[useCall] Remote peer attack status updated:', data);
        setPeerAttackState({
          active: data.attackMode !== 'none',
          mode: data.attackMode,
          faceSwap: data.faceSwap,
          voiceTransform: data.voiceTransform,
        });
      },

      onAttackError: (data) => {
        setErrorMessage(data.message);
      },

      onDisconnect: (reason) => {
        if (callStatus !== 'ended') {
          console.warn('[useCall] Signaling disconnected:', reason);
        }
      },
    });

    signalingService.joinRoom(cleanRoomId);
  }, [token, localStream, createPeerConnection, makeOffer, handleOffer, handleAnswer, handleIceCandidate, cleanupPeerConnection, callStatus]);

  // Controls: Toggle Microphone
  const toggleMic = useCallback(() => {
    if (!localStream) return;
    const audioTrack = localStream.getAudioTracks()[0];
    if (audioTrack) {
      audioTrack.enabled = !audioTrack.enabled;
      setIsMicMuted(!audioTrack.enabled);
    }
  }, [localStream]);

  // Controls: Toggle Camera
  const toggleCamera = useCallback(() => {
    if (!localStream) return;
    const videoTrack = localStream.getVideoTracks()[0];
    if (videoTrack) {
      videoTrack.enabled = !videoTrack.enabled;
      setIsCameraOff(!videoTrack.enabled);
    }
  }, [localStream]);

  // Controls: Screen Share
  const toggleScreenShare = useCallback(async () => {
    if (isScreenSharing) {
      // Revert back to original video track
      if (screenStreamRef.current) {
        mediaService.stopStream(screenStreamRef.current);
        screenStreamRef.current = null;
      }
      setIsScreenSharing(false);

      const activeVideoTrack = attackState.faceSwap && facePipelineRef.current
        ? facePipelineRef.current.start(originalVideoTrackRef.current!)
        : originalVideoTrackRef.current;

      if (activeVideoTrack) {
        await replaceVideoTrack(activeVideoTrack);
      }
    } else {
      const screenStream = await mediaService.getScreenMedia();
      if (screenStream) {
        screenStreamRef.current = screenStream;
        const screenTrack = screenStream.getVideoTracks()[0];
        
        screenTrack.onended = () => {
          setIsScreenSharing(false);
          if (originalVideoTrackRef.current) {
            replaceVideoTrack(originalVideoTrackRef.current);
          }
        };

        setIsScreenSharing(true);
        await replaceVideoTrack(screenTrack);
      }
    }
  }, [isScreenSharing, attackState.faceSwap, replaceVideoTrack]);

  // ==========================================
  // TESTER ONLY ATTACK SIMULATION METHODS
  // ==========================================

  const activateFaceSwap = useCallback(async (enable?: boolean) => {
    if (!isTester) {
      console.error('[Security] Non-tester account attempted to activate face swap.');
      setErrorMessage('Access Denied: Only TESTER accounts can access Attack Simulation.');
      return;
    }

    const nextState = enable !== undefined ? enable : !attackState.faceSwap;
    const currentVideoTrack = originalVideoTrackRef.current;

    if (nextState) {
      if (!currentVideoTrack) {
        setErrorMessage('Cannot activate Face Simulation: Camera track not available.');
        return;
      }

      if (!facePipelineRef.current) {
        facePipelineRef.current = new FaceSimulationPipeline();
      }

      const syntheticTrack = facePipelineRef.current.start(currentVideoTrack, attackState.facePreset as FacePreset);
      await replaceVideoTrack(syntheticTrack);

      const newMode = attackState.voiceTransform ? 'combined' : 'face';
      setAttackState((prev) => ({
        ...prev,
        faceSwap: true,
        mode: newMode,
      }));

      signalingService.sendAttackUpdate(roomId, true, attackState.voiceTransform, newMode);
    } else {
      // Disable face swap
      if (facePipelineRef.current) {
        facePipelineRef.current.stop();
      }

      if (currentVideoTrack) {
        await replaceVideoTrack(currentVideoTrack);
      }

      const newMode = attackState.voiceTransform ? 'voice' : 'none';
      setAttackState((prev) => ({
        ...prev,
        faceSwap: false,
        mode: newMode,
      }));

      signalingService.sendAttackUpdate(roomId, false, attackState.voiceTransform, newMode);
    }
  }, [isTester, attackState.faceSwap, attackState.voiceTransform, attackState.facePreset, roomId, replaceVideoTrack]);

  const activateVoiceTransform = useCallback(async (enable?: boolean) => {
    if (!isTester) {
      console.error('[Security] Non-tester account attempted to activate voice transformation.');
      setErrorMessage('Access Denied: Only TESTER accounts can access Attack Simulation.');
      return;
    }

    const nextState = enable !== undefined ? enable : !attackState.voiceTransform;
    const currentAudioTrack = originalAudioTrackRef.current;

    if (nextState) {
      if (!currentAudioTrack) {
        setErrorMessage('Cannot activate Voice Simulation: Microphone track not available.');
        return;
      }

      if (!voicePipelineRef.current) {
        voicePipelineRef.current = new VoiceTransformationPipeline();
      }

      const transformedTrack = voicePipelineRef.current.start(currentAudioTrack, attackState.voicePreset as VoicePreset);
      await replaceAudioTrack(transformedTrack);

      const newMode = attackState.faceSwap ? 'combined' : 'voice';
      setAttackState((prev) => ({
        ...prev,
        voiceTransform: true,
        mode: newMode,
      }));

      signalingService.sendAttackUpdate(roomId, attackState.faceSwap, true, newMode);
    } else {
      // Disable voice transformation
      if (voicePipelineRef.current) {
        voicePipelineRef.current.stop();
      }

      if (currentAudioTrack) {
        await replaceAudioTrack(currentAudioTrack);
      }

      const newMode = attackState.faceSwap ? 'face' : 'none';
      setAttackState((prev) => ({
        ...prev,
        voiceTransform: false,
        mode: newMode,
      }));

      signalingService.sendAttackUpdate(roomId, attackState.faceSwap, false, newMode);
    }
  }, [isTester, attackState.faceSwap, attackState.voiceTransform, attackState.voicePreset, roomId, replaceAudioTrack]);

  const activateCombinedAttack = useCallback(async () => {
    if (!isTester) {
      setErrorMessage('Access Denied: Only TESTER accounts can access Attack Simulation.');
      return;
    }

    // Turn both ON
    const currentVideoTrack = originalVideoTrackRef.current;
    const currentAudioTrack = originalAudioTrackRef.current;

    if (currentVideoTrack) {
      if (!facePipelineRef.current) {
        facePipelineRef.current = new FaceSimulationPipeline();
      }
      const syntheticTrack = facePipelineRef.current.start(currentVideoTrack, attackState.facePreset as FacePreset);
      await replaceVideoTrack(syntheticTrack);
    }

    if (currentAudioTrack) {
      if (!voicePipelineRef.current) {
        voicePipelineRef.current = new VoiceTransformationPipeline();
      }
      const transformedTrack = voicePipelineRef.current.start(currentAudioTrack, attackState.voicePreset as VoicePreset);
      await replaceAudioTrack(transformedTrack);
    }

    setAttackState((prev) => ({
      ...prev,
      faceSwap: true,
      voiceTransform: true,
      mode: 'combined',
    }));

    signalingService.sendAttackUpdate(roomId, true, true, 'combined');
  }, [isTester, attackState.facePreset, attackState.voicePreset, roomId, replaceVideoTrack, replaceAudioTrack]);

  const resetAttack = useCallback(async () => {
    if (!isTester) return;

    if (facePipelineRef.current) {
      facePipelineRef.current.stop();
    }
    if (voicePipelineRef.current) {
      voicePipelineRef.current.stop();
    }

    if (originalVideoTrackRef.current) {
      await replaceVideoTrack(originalVideoTrackRef.current);
    }
    if (originalAudioTrackRef.current) {
      await replaceAudioTrack(originalAudioTrackRef.current);
    }

    setAttackState((prev) => ({
      ...prev,
      faceSwap: false,
      voiceTransform: false,
      mode: 'none',
    }));

    signalingService.sendAttackUpdate(roomId, false, false, 'none');
  }, [isTester, roomId, replaceVideoTrack, replaceAudioTrack]);

  const setFacePreset = useCallback((preset: 'neural-clone' | 'biometric-mask' | 'synthetic-executive') => {
    setAttackState((prev) => ({ ...prev, facePreset: preset }));
    if (facePipelineRef.current && attackState.faceSwap) {
      facePipelineRef.current.setPreset(preset);
    }
  }, [attackState.faceSwap]);

  const setVoicePreset = useCallback((preset: 'robotic-vocoder' | 'deep-pitch-neural' | 'synthetic-clone') => {
    setAttackState((prev) => ({ ...prev, voicePreset: preset }));
    if (voicePipelineRef.current && attackState.voiceTransform) {
      voicePipelineRef.current.setPreset(preset);
    }
  }, [attackState.voiceTransform]);

  // End Call & Cleanup
  const endCall = useCallback(() => {
    if (roomId) {
      signalingService.leaveRoom(roomId);
    }
    resetAttack();
    cleanupPeerConnection();

    if (screenStreamRef.current) {
      mediaService.stopStream(screenStreamRef.current);
      screenStreamRef.current = null;
    }
    if (localStream) {
      mediaService.stopStream(localStream);
      setLocalStream(null);
    }

    signalingService.disconnect();
    setCallStatus('ended');
    setPeerInfo(null);
  }, [roomId, resetAttack, cleanupPeerConnection, localStream]);

  // Clean up on component unmount
  useEffect(() => {
    return () => {
      if (facePipelineRef.current) {
        facePipelineRef.current.stop();
      }
      if (voicePipelineRef.current) {
        voicePipelineRef.current.stop();
      }
      if (screenStreamRef.current) {
        mediaService.stopStream(screenStreamRef.current);
      }
      if (localStream) {
        mediaService.stopStream(localStream);
      }
      cleanupPeerConnection();
      signalingService.disconnect();
    };
  }, [cleanupPeerConnection, localStream]);

  return {
    roomId,
    callStatus,
    errorMessage,
    mediaError,
    peerInfo,
    localStream,
    remoteStream,
    connectionState,
    isMicMuted,
    isCameraOff,
    isScreenSharing,
    isTester,
    attackState,
    peerAttackState,
    initLocalMedia,
    joinRoom,
    toggleMic,
    toggleCamera,
    toggleScreenShare,
    activateFaceSwap,
    activateVoiceTransform,
    activateCombinedAttack,
    resetAttack,
    setFacePreset,
    setVoicePreset,
    endCall,
    clearError: () => setErrorMessage(null),
  };
}
