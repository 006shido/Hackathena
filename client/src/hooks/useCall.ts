import { useCallback,useEffect,useLayoutEffect,useRef,useState } from 'react';
import {
FacePreset,
FaceSimulationPipeline,
FaceSwapTelemetry,
GalleryFaceValidationResult,
} from '../attack/faceSimulation';
import { VoicePreset,VoiceTransformationPipeline } from '../attack/voiceTransformation';
import { MediaAccessResult,mediaService } from '../services/media';
import { signalingService } from '../services/signaling';
import { AttackMode,AttackState,DEFAULT_FACE_BLEND_CONFIG,FaceBlendConfig } from '../types/attack';
import { User } from '../types/auth';
import { CallStatus,ParticipantInfo } from '../types/call';
import { useWebRTC } from './useWebRTC';

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

  // Attack simulator & AI Face Swap state (strictly initialized & managed for tester role)
  const isTester = user?.role === 'tester';
  const [attackState, setAttackState] = useState<AttackState>({
    faceSwap: false,
    voiceTransform: false,
    mode: 'none',
    facePreset: 'neural-clone',
    voicePreset: 'robotic-vocoder',
  });

  const [selectedFacePreview, setSelectedFacePreview] = useState<string | null>('/synthetic_face_avatar.jpg');
  const attackStateRef = useRef(attackState);
  useLayoutEffect(() => { attackStateRef.current = attackState; }, [attackState]);
  const screenSharingRef = useRef(isScreenSharing);
  useLayoutEffect(() => { screenSharingRef.current = isScreenSharing; }, [isScreenSharing]);
  const [selectedFaceName, setSelectedFaceName] = useState<string>('Marcus (Neural Clone)');
  const [faceSwapTelemetry, setFaceSwapTelemetry] = useState<FaceSwapTelemetry | undefined>(undefined);
  const [previousFaceSwap, setPreviousFaceSwap] = useState(attackState.faceSwap);
  if (previousFaceSwap !== attackState.faceSwap) {
    setPreviousFaceSwap(attackState.faceSwap);
    setFaceSwapTelemetry(undefined);
  }
  const [faceBlendConfig, setFaceBlendConfig] = useState<FaceBlendConfig>({
    ...DEFAULT_FACE_BLEND_CONFIG,
  });

  // Pipelines
  const facePipelineRef = useRef<FaceSimulationPipeline | null>(null);
  const facePresetRef = useRef<FacePreset>('neural-clone');
  const voicePipelineRef = useRef<VoiceTransformationPipeline | null>(null);
  const [isVoiceMonitoring, setIsVoiceMonitoring] = useState<boolean>(false);

  const updateFaceBlendConfig = useCallback((newConfig: Partial<FaceBlendConfig>) => {
    setFaceBlendConfig((prev) => {
      const merged = { ...prev, ...newConfig };
      if (facePipelineRef.current) {
        facePipelineRef.current.setBlendConfig(merged);
      }
      return merged;
    });
  }, []);

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


  // Keep references to original tracks & stream
  const localStreamRef = useRef<MediaStream | null>(null);
  const originalVideoTrackRef = useRef<MediaStreamTrack | null>(null);
  const originalAudioTrackRef = useRef<MediaStreamTrack | null>(null);
  const screenStreamRef = useRef<MediaStream | null>(null);
  const isMountedRef = useRef<boolean>(true);
  const isCallEndedRef = useRef<boolean>(false);

  const {
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
  } = useWebRTC();

  // Initialize media devices for preview or call
  const initLocalMedia = useCallback(async (video = true, audio = true): Promise<MediaAccessResult> => {
    isCallEndedRef.current = false;

    // If existing media tracks are open, stop them first to prevent hardware leak
    if (localStreamRef.current) {
      mediaService.stopStream(localStreamRef.current);
      localStreamRef.current = null;
    }
    if (originalVideoTrackRef.current) {
      try {
        originalVideoTrackRef.current.enabled = false;
        originalVideoTrackRef.current.stop();
      } catch {}
      originalVideoTrackRef.current = null;
    }
    if (originalAudioTrackRef.current) {
      try {
        originalAudioTrackRef.current.enabled = false;
        originalAudioTrackRef.current.stop();
      } catch {}
      originalAudioTrackRef.current = null;
    }

    setCallStatus('checking-devices');
    const result = await mediaService.getUserMedia(video, audio);

    // If call was ended or component unmounted while waiting for camera, immediately kill the acquired stream
    if (!isMountedRef.current || isCallEndedRef.current) {
      if (result.stream) {
        mediaService.stopStream(result.stream);
      }
      return { stream: null, hasVideo: false, hasAudio: false };
    }

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
    setPeerAttackState({ active: false, mode: 'none', faceSwap: false, voiceTransform: false });
    setErrorMessage(null);
    setCallStatus('joining');

    // Connect to signaling server with auth token
    signalingService.connect(token, {
      onRoomJoined: (data) => {
        if (user?.role === 'tester') {
          const current = attackStateRef.current;
          signalingService.sendAttackUpdate(data.roomId, current.faceSwap && !screenSharingRef.current, current.voiceTransform, current.mode);
        }
        const activeLocalStream = localStreamRef.current || localStream;
        
        if (data.peers.length > 0) {
          // A peer is already in room: initialize connection and WAIT for their offer
          setPeerInfo(data.peers[0]);
          setCallStatus('connected');
          createPeerConnection(data.roomId, activeLocalStream, true);
        } else {
          // First in room: waiting for someone to join. Do not pre-gather ICE candidates into the void
          setCallStatus('waiting');
          setPeerInfo(null);
        }
      },

      onPeerJoined: (peer) => {
        const activeLocalStream = localStreamRef.current || localStream;
        setPeerInfo(peer);
        setCallStatus('connected');
        
        // Host initializes fresh peer connection and sends offer to the newly joined peer
        createPeerConnection(cleanRoomId, activeLocalStream, true);
        makeOffer(cleanRoomId, activeLocalStream);
      },

      onPeerLeft: () => {
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
        setCallStatus('connected');
        const activeLocalStream = localStreamRef.current || localStream;
        await handleOffer(cleanRoomId, sdp, activeLocalStream);
      },

      onAnswer: async ({ sdp }) => {
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
        setPeerAttackState({ active: false, mode: 'none', faceSwap: false, voiceTransform: false });
        if (callStatus !== 'ended') {
          console.warn('[useCall] Signaling disconnected:', reason);
        }
      },
    });

    signalingService.joinRoom(cleanRoomId);
  }, [token, user?.role, localStream, createPeerConnection, makeOffer, handleOffer, handleAnswer, handleIceCandidate, cleanupPeerConnection, callStatus]);

  // Controls: Toggle Microphone
  const toggleMic = useCallback(async () => {
    const stream = localStreamRef.current || localStream;
    if (!stream || stream.getAudioTracks().length === 0) {
      try {
        const result = await mediaService.getUserMedia(!isCameraOff, true);
        if (result.stream) {
          const newAudioTrack = result.stream.getAudioTracks()[0];
          if (newAudioTrack) {
            if (stream) {
              stream.addTrack(newAudioTrack);
              originalAudioTrackRef.current = newAudioTrack;
              const refreshed = new MediaStream(stream.getTracks());
              localStreamRef.current = refreshed;
              setLocalStream(refreshed);
              setIsMicMuted(false);
              await replaceAudioTrack(newAudioTrack);
            } else {
              setLocalStream(result.stream);
              localStreamRef.current = result.stream;
              originalAudioTrackRef.current = newAudioTrack;
              originalVideoTrackRef.current = result.stream.getVideoTracks()[0] || null;
              setIsMicMuted(false);
            }
          }
        }
      } catch (err) {
        console.warn('Could not acquire microphone:', err);
      }
      return;
    }

    const audioTrack = stream.getAudioTracks()[0];
    if (audioTrack) {
      const nextEnabled = !audioTrack.enabled;
      audioTrack.enabled = nextEnabled;
      setIsMicMuted(!nextEnabled);
      const refreshed = new MediaStream(stream.getTracks());
      localStreamRef.current = refreshed;
      setLocalStream(refreshed);
    }
  }, [localStream, isCameraOff, replaceAudioTrack]);

  // Controls: Toggle Camera
  const toggleCamera = useCallback(async () => {
    const stream = localStreamRef.current || localStream;
    if (!stream || stream.getVideoTracks().length === 0) {
      try {
        const result = await mediaService.getUserMedia(true, !isMicMuted);
        if (result.stream) {
          const newVideoTrack = result.stream.getVideoTracks()[0];
          if (newVideoTrack) {
            if (stream) {
              stream.addTrack(newVideoTrack);
              originalVideoTrackRef.current = newVideoTrack;
              const refreshed = new MediaStream(stream.getTracks());
              localStreamRef.current = refreshed;
              setLocalStream(refreshed);
              setIsCameraOff(false);
              await replaceVideoTrack(newVideoTrack);
            } else {
              setLocalStream(result.stream);
              localStreamRef.current = result.stream;
              originalVideoTrackRef.current = newVideoTrack;
              originalAudioTrackRef.current = result.stream.getAudioTracks()[0] || null;
              setIsCameraOff(false);
            }
          }
        }
      } catch (err) {
        console.warn('Could not acquire camera on toggle:', err);
      }
      return;
    }

    const videoTrack = stream.getVideoTracks()[0];
    if (videoTrack) {
      const nextEnabled = !videoTrack.enabled;
      videoTrack.enabled = nextEnabled;
      setIsCameraOff(!nextEnabled);
      const refreshed = new MediaStream(stream.getTracks());
      localStreamRef.current = refreshed;
      setLocalStream(refreshed);
    }
  }, [localStream, isMicMuted, replaceVideoTrack]);

  // Controls: Screen Share
  const toggleScreenShare = useCallback(async () => {
    const publishVideo = async (track: MediaStreamTrack) => {
      await replaceVideoTrack(track);
      if (localStreamRef.current) {
        const updated = new MediaStream([...localStreamRef.current.getAudioTracks(), track]);
        localStreamRef.current = updated;
        setLocalStream(updated);
      }
    };
    const restoreCamera = async () => {
      if (isCallEndedRef.current) return;
      if (screenStreamRef.current) {
        mediaService.stopStream(screenStreamRef.current);
        screenStreamRef.current = null;
      }
      setIsScreenSharing(false);
      screenSharingRef.current = false;
      const current = attackStateRef.current;
      let activeVideoTrack = originalVideoTrackRef.current;
      let faceRestored = false;
      if (current.faceSwap && activeVideoTrack) {
        if (!facePipelineRef.current) facePipelineRef.current = new FaceSimulationPipeline();
        try {
          activeVideoTrack = await facePipelineRef.current.start(activeVideoTrack, facePresetRef.current);
          faceRestored = true;
        } catch (error) {
          facePipelineRef.current.stop();
          const mode = current.voiceTransform ? 'voice' : 'none';
          setAttackState((previous) => ({ ...previous, faceSwap: false, mode }));
          setErrorMessage(error instanceof Error ? error.message : 'Could not restore face simulation.');
        }
      }
      if (isCallEndedRef.current) { facePipelineRef.current?.stop(); return; }
      if (activeVideoTrack) await publishVideo(activeVideoTrack);
      if (isTester) {
        const mode = faceRestored ? (current.voiceTransform ? 'combined' : 'face') : (current.voiceTransform ? 'voice' : 'none');
        signalingService.sendAttackUpdate(roomId, faceRestored, current.voiceTransform, mode);
      }
    };
    if (isScreenSharing) {
      await restoreCamera();
    } else {
      const screenStream = await mediaService.getScreenMedia();
      if (screenStream) {
        screenStreamRef.current = screenStream;
        const screenTrack = screenStream.getVideoTracks()[0];
        screenTrack.onended = () => {
          void restoreCamera().catch(error => setErrorMessage(error instanceof Error ? error.message : 'Could not restore camera.'));
        };
        await publishVideo(screenTrack);
        facePipelineRef.current?.stop();
        setIsScreenSharing(true);
        screenSharingRef.current = true;
        if (isTester) {
          const current = attackStateRef.current;
          signalingService.sendAttackUpdate(roomId, false, current.voiceTransform, current.voiceTransform ? 'voice' : 'none');
        }
      }
    }
  }, [isScreenSharing, isTester, roomId, replaceVideoTrack]);

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

    if (isScreenSharing) {
      setAttackState(previous => ({ ...previous, faceSwap: nextState,
        mode: nextState ? (previous.voiceTransform ? 'combined' : 'face') : (previous.voiceTransform ? 'voice' : 'none') }));
      signalingService.sendAttackUpdate(roomId, false, attackState.voiceTransform, attackState.voiceTransform ? 'voice' : 'none');
      return;
    }

    if (nextState) {
      if (!currentVideoTrack) {
        setErrorMessage('Cannot activate Face Simulation: Camera track not available.');
        return;
      }

      if (!facePipelineRef.current) {
        facePipelineRef.current = new FaceSimulationPipeline();
      }
      facePipelineRef.current.setBlendConfig(faceBlendConfig);

      const currentPreset = facePresetRef.current;
      let syntheticTrack: MediaStreamTrack;
      try {
        syntheticTrack = await facePipelineRef.current.start(currentVideoTrack, currentPreset);
      } catch (error) {
        setErrorMessage(error instanceof Error ? error.message : 'Could not start face swap.');
        return;
      }
      await replaceVideoTrack(syntheticTrack);

      if (localStreamRef.current) {
        const audioTracks = localStreamRef.current.getAudioTracks();
        const updatedStream = new MediaStream([...audioTracks, syntheticTrack]);
        localStreamRef.current = updatedStream;
        setLocalStream(updatedStream);
      }

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
        if (localStreamRef.current) {
          const audioTracks = localStreamRef.current.getAudioTracks();
          const restoredStream = new MediaStream([...audioTracks, currentVideoTrack]);
          localStreamRef.current = restoredStream;
          setLocalStream(restoredStream);
        }
      }

      const newMode = attackState.voiceTransform ? 'voice' : 'none';
      setAttackState((prev) => ({
        ...prev,
        faceSwap: false,
        mode: newMode,
      }));

      signalingService.sendAttackUpdate(roomId, false, attackState.voiceTransform, newMode);
    }
  }, [isTester, isScreenSharing, attackState.faceSwap, attackState.voiceTransform, faceBlendConfig, roomId, replaceVideoTrack]);

  const toggleVoiceMonitor = useCallback((enable?: boolean) => {
    setIsVoiceMonitoring((prev) => {
      const next = enable !== undefined ? enable : !prev;
      if (voicePipelineRef.current) {
        voicePipelineRef.current.setMonitoring(next);
      }
      return next;
    });
  }, []);

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

      const transformedTrack = await voicePipelineRef.current.start(currentAudioTrack, attackState.voicePreset as VoicePreset);
      await replaceAudioTrack(transformedTrack);

      // Sync active transformed audio track into localStreamRef and state
      if (localStreamRef.current) {
        const videoTracks = localStreamRef.current.getVideoTracks();
        const updatedStream = new MediaStream([transformedTrack, ...videoTracks]);
        localStreamRef.current = updatedStream;
        setLocalStream(updatedStream);
      }

      const newMode = attackState.faceSwap ? 'combined' : 'voice';
      setAttackState((prev) => ({
        ...prev,
        voiceTransform: true,
        mode: newMode,
      }));

      signalingService.sendAttackUpdate(roomId, attackState.faceSwap && !isScreenSharing, true, newMode);
    } else {
      // Disable voice transformation
      if (voicePipelineRef.current) {
        voicePipelineRef.current.stop();
      }

      if (currentAudioTrack) {
        await replaceAudioTrack(currentAudioTrack);
        if (localStreamRef.current) {
          const videoTracks = localStreamRef.current.getVideoTracks();
          const restoredStream = new MediaStream([currentAudioTrack, ...videoTracks]);
          localStreamRef.current = restoredStream;
          setLocalStream(restoredStream);
        }
      }

      const newMode = attackState.faceSwap ? 'face' : 'none';
      setAttackState((prev) => ({
        ...prev,
        voiceTransform: false,
        mode: newMode,
      }));

      signalingService.sendAttackUpdate(roomId, attackState.faceSwap && !isScreenSharing, false, newMode);
    }
  }, [isTester, isScreenSharing, attackState.faceSwap, attackState.voiceTransform, attackState.voicePreset, roomId, replaceAudioTrack]);

  const activateCombinedAttack = useCallback(async () => {
    if (!isTester) {
      setErrorMessage('Access Denied: Only TESTER accounts can access Attack Simulation.');
      return;
    }

    // Turn both ON
    const currentVideoTrack = originalVideoTrackRef.current;
    const currentAudioTrack = originalAudioTrackRef.current;

    let activeVideo = isScreenSharing ? screenStreamRef.current?.getVideoTracks()[0] : currentVideoTrack;
    if (currentVideoTrack && !isScreenSharing) {
      if (!facePipelineRef.current) {
        facePipelineRef.current = new FaceSimulationPipeline();
      }
      const currentPreset = facePresetRef.current;
      let syntheticTrack: MediaStreamTrack;
      try {
        syntheticTrack = await facePipelineRef.current.start(currentVideoTrack, currentPreset);
      } catch (error) {
        setErrorMessage(error instanceof Error ? error.message : 'Could not start face swap.');
        return;
      }
      await replaceVideoTrack(syntheticTrack);
      activeVideo = syntheticTrack;
    }

    let activeAudio = currentAudioTrack;
    if (currentAudioTrack) {
      if (!voicePipelineRef.current) {
        voicePipelineRef.current = new VoiceTransformationPipeline();
      }
      const transformedTrack = await voicePipelineRef.current.start(currentAudioTrack, attackState.voicePreset as VoicePreset);
      await replaceAudioTrack(transformedTrack);
      activeAudio = transformedTrack;
    }

    if (localStreamRef.current && activeVideo && activeAudio) {
      const updatedStream = new MediaStream([activeAudio, activeVideo]);
      localStreamRef.current = updatedStream;
      setLocalStream(updatedStream);
    }

    setAttackState((prev) => ({
      ...prev,
      faceSwap: true,
      voiceTransform: true,
      mode: 'combined',
    }));

    signalingService.sendAttackUpdate(roomId, !isScreenSharing, true, isScreenSharing ? 'voice' : 'combined');
  }, [isTester, isScreenSharing, attackState.voicePreset, roomId, replaceVideoTrack, replaceAudioTrack]);

  const resetAttack = useCallback(async () => {
    if (!isTester) return;

    if (facePipelineRef.current) {
      facePipelineRef.current.stop();
    }
    if (voicePipelineRef.current) {
      voicePipelineRef.current.stop();
    }

    const restoredVideo = isScreenSharing ? screenStreamRef.current?.getVideoTracks()[0] : originalVideoTrackRef.current;
    if (restoredVideo) {
      await replaceVideoTrack(restoredVideo);
    }
    if (originalAudioTrackRef.current) {
      await replaceAudioTrack(originalAudioTrackRef.current);
    }

    if (originalAudioTrackRef.current && restoredVideo) {
      const restoredStream = new MediaStream([originalAudioTrackRef.current, restoredVideo]);
      localStreamRef.current = restoredStream;
      setLocalStream(restoredStream);
    }

    setAttackState((prev) => ({
      ...prev,
      faceSwap: false,
      voiceTransform: false,
      mode: 'none',
    }));

    signalingService.sendAttackUpdate(roomId, false, false, 'none');
  }, [isTester, isScreenSharing, roomId, replaceVideoTrack, replaceAudioTrack]);

  const setFacePreset = useCallback((preset: FacePreset) => {
    facePresetRef.current = preset;
    setAttackState((prev) => ({ ...prev, facePreset: preset }));

    const presetNames: Record<string, { src: string; name: string }> = {
      'neural-clone': { src: '/synthetic_face_avatar.jpg', name: 'Marcus (Neural Clone)' },
      'mona-lisa': { src: '/mona_lisa.jpg', name: 'Emma (Studio Headshot)' },
      'cyber-agent': { src: '/cyber_agent.jpg', name: 'Alex (Clean Headshot)' },
      'astronaut': { src: '/astronaut.jpg', name: 'Sophia (Natural Portrait)' },
      'synthetic-executive': { src: '/synthetic_executive.jpg', name: 'David (Corporate Exec)' },
    };
    if (presetNames[preset]) {
      setSelectedFacePreview(presetNames[preset].src);
      setSelectedFaceName(presetNames[preset].name);
    }

    if (facePipelineRef.current) {
      void facePipelineRef.current.setPreset(preset).catch((error: unknown) => {
        setErrorMessage(error instanceof Error ? error.message : 'Could not load the selected face.');
      });
    }
  }, []);

  const uploadGalleryFace = useCallback(async (file: File): Promise<GalleryFaceValidationResult> => {
    if (!isTester) {
      return { success: false, error: 'Unauthorized: Only Tester accounts can upload faces.' };
    }

    if (!facePipelineRef.current) {
      facePipelineRef.current = new FaceSimulationPipeline();
    }

    const result = await facePipelineRef.current.loadGalleryImage(file);
    if (result.success) {
      facePresetRef.current = 'custom-upload';
      setAttackState((prev) => ({ ...prev, facePreset: 'custom-upload' }));
      setSelectedFacePreview(result.previewUrl || null);
      setSelectedFaceName(file.name);
    }
    return result;
  }, [isTester]);

  const resetFaceSwapFace = useCallback(async () => {
    if (!isTester) return;
    if (facePipelineRef.current) {
      await facePipelineRef.current.resetFace();
    }
    facePresetRef.current = 'neural-clone';
    setAttackState((prev) => ({ ...prev, facePreset: 'neural-clone' }));
    setSelectedFacePreview('/synthetic_face_avatar.jpg');
    setSelectedFaceName('Marcus (Neural Clone)');
  }, [isTester]);

  // Periodic telemetry poll when face swap is active
  useEffect(() => {
    if (!attackState.faceSwap) {
      return;
    }
    const interval = setInterval(() => {
      if (facePipelineRef.current) {
        setFaceSwapTelemetry(facePipelineRef.current.getTelemetry());
      }
    }, 1000);
    return () => clearInterval(interval);
  }, [attackState.faceSwap]);

const setVoicePreset = useCallback((preset: 'robotic-vocoder' | 'deep-pitch-neural' | 'synthetic-clone') => {
    setAttackState((prev) => ({ ...prev, voicePreset: preset }));
    if (voicePipelineRef.current && attackState.voiceTransform) {
      voicePipelineRef.current.setPreset(preset);
    }
  }, [attackState.voiceTransform]);

  // Stop all media tracks helper
  const stopAllMedia = useCallback(() => {
    isCallEndedRef.current = true;
    mediaService.invalidateMediaSession();

    // 1. Screen sharing stream
    if (screenStreamRef.current) {
      mediaService.stopStream(screenStreamRef.current);
      screenStreamRef.current = null;
    }

    // 2. Original individual tracks
    if (originalVideoTrackRef.current) {
      try {
        originalVideoTrackRef.current.enabled = false;
        originalVideoTrackRef.current.stop();
      } catch {}
      originalVideoTrackRef.current = null;
    }
    if (originalAudioTrackRef.current) {
      try {
        originalAudioTrackRef.current.enabled = false;
        originalAudioTrackRef.current.stop();
      } catch {}
      originalAudioTrackRef.current = null;
    }

    // 3. Local stream ref & state
    if (localStreamRef.current) {
      mediaService.stopStream(localStreamRef.current);
      localStreamRef.current = null;
    }
    setLocalStream(null);

    // 4. Attack simulation pipelines
    if (facePipelineRef.current) {
      facePipelineRef.current.destroy();
      facePipelineRef.current = null;
    }
    if (voicePipelineRef.current) {
      voicePipelineRef.current.stop();
      voicePipelineRef.current = null;
    }

    // 5. Global kill-switch: guarantees all tracks and video elements are detached
    mediaService.stopAllMedia();
  }, []);

  // End Call & Cleanup
  const endCall = useCallback(() => {
    isCallEndedRef.current = true;
    if (roomId) {
      signalingService.leaveRoom(roomId);
    }

    // Stop attack simulation pipelines
    if (facePipelineRef.current) {
      facePipelineRef.current.destroy();
      facePipelineRef.current = null;
    }
    if (voicePipelineRef.current) {
      voicePipelineRef.current.stop();
      voicePipelineRef.current = null;
    }

    setAttackState({
      faceSwap: false,
      voiceTransform: false,
      mode: 'none',
      facePreset: 'neural-clone',
      voicePreset: 'robotic-vocoder',
    });

    // Cleanup WebRTC connection and sender tracks
    cleanupPeerConnection();

    // Release all camera, screen, and audio hardware tracks
    stopAllMedia();

    signalingService.disconnect();
    setCallStatus('ended');
    setPeerInfo(null);
  }, [roomId, cleanupPeerConnection, stopAllMedia]);

  // Clean up on component unmount (ONLY runs once when component unmounts)
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      isCallEndedRef.current = true;
      if (screenStreamRef.current) {
        mediaService.stopStream(screenStreamRef.current);
        screenStreamRef.current = null;
      }
      if (originalVideoTrackRef.current) {
        try {
          originalVideoTrackRef.current.enabled = false;
          originalVideoTrackRef.current.stop();
        } catch {}
        originalVideoTrackRef.current = null;
      }
      if (originalAudioTrackRef.current) {
        try {
          originalAudioTrackRef.current.enabled = false;
          originalAudioTrackRef.current.stop();
        } catch {}
        originalAudioTrackRef.current = null;
      }
      if (localStreamRef.current) {
        mediaService.stopStream(localStreamRef.current);
        localStreamRef.current = null;
      }
      if (facePipelineRef.current) {
        facePipelineRef.current.destroy();
        facePipelineRef.current = null;
      }
      if (voicePipelineRef.current) {
        voicePipelineRef.current.stop();
        voicePipelineRef.current = null;
      }
      cleanupPeerConnection();
      signalingService.disconnect();
      mediaService.stopAllMedia();
    };
  }, [cleanupPeerConnection]);


  return {
    roomId,
    callStatus,
    errorMessage,
    mediaError,
    peerInfo,
    localStream,
    remoteStream,
    connectionState,
    iceState,
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
    uploadGalleryFace,
    resetFaceSwapFace,
    selectedFacePreview,
    selectedFaceName,
    faceSwapTelemetry: attackState.faceSwap ? faceSwapTelemetry : undefined,
    faceBlendConfig,
    updateFaceBlendConfig,
    isVoiceMonitoring,
    toggleVoiceMonitor,
    endCall,
    clearError: () => setErrorMessage(null),
  };
}

