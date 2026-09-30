import React, { useState, useEffect, useRef } from 'react';
import { Users, AlertCircle, ArrowLeft, AlertTriangle } from 'lucide-react';
import { User as UserType } from '../types/auth';
import { useCall } from '../hooks/useCall';
import { useVoiceDetection } from '../hooks/useVoiceDetection';
import { RoomHeader } from '../components/RoomHeader';
import { VideoTile } from '../components/VideoTile';
import { CallControls } from '../components/CallControls';
import { SecurityPanel } from '../components/SecurityPanel';
import { AttackSimulator } from '../components/AttackSimulator';
import { DevicePreview } from '../components/DevicePreview';

interface CallPageProps {
  roomId: string;
  user: UserType;
  token: string;
  onExit: () => void;
}

export const CallPage: React.FC<CallPageProps> = ({
  roomId,
  user,
  token,
  onExit,
}) => {
  const isTester = user.role === 'tester';
  const [showDevicePreview, setShowDevicePreview] = useState(true);
  
  // On desktop open by default for tester, on mobile keep closed so video takes full screen
  const [isAttackDrawerOpen, setIsAttackDrawerOpen] = useState(
    () => isTester && typeof window !== 'undefined' && window.innerWidth >= 1024
  );
  const [isSecurityPanelOpen, setIsSecurityPanelOpen] = useState(false);

  const {
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
  } = useCall(user, token);

  // Autonomous real-time acoustic deepfake analysis on incoming peer audio stream (DSP & ML feature classifier)
  const voiceDetection = useVoiceDetection(
    remoteStream,
    !showDevicePreview && Boolean(remoteStream)
  );

  // Deepfake alert triggered purely by autonomous acoustic classification
  const isDeepfakeAlert = voiceDetection.status === 'deepfake';

  // Initialize camera and mic for preview on mount
  useEffect(() => {
    initLocalMedia(true, true);
  }, [initLocalMedia]);

  // Keep a stable ref to endCall so beforeunload / unmount handlers don't re-trigger cleanup during re-renders
  const endCallRef = useRef(endCall);
  useEffect(() => {
    endCallRef.current = endCall;
  }, [endCall]);

  // Clean up all media and connections on unmount and beforeunload
  useEffect(() => {
    const handleBeforeUnload = () => {
      endCallRef.current();
    };
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
      endCallRef.current();
    };
  }, []);

  const handleConfirmJoin = () => {
    setShowDevicePreview(false);
    joinRoom(roomId);
  };

  const handleEndCall = () => {
    endCall();
    onExit();
  };

  const handleCancelPreview = () => {
    endCall();
    onExit();
  };

  // Device preview
  if (showDevicePreview) {
    return (
      <div className="min-h-screen w-full bg-[#202124] flex items-center justify-center p-3 sm:p-4 overflow-y-auto">
        <DevicePreview
          stream={localStream}
          isMicMuted={isMicMuted}
          isCameraOff={isCameraOff}
          mediaError={mediaError}
          username={user.name}
          role={user.role}
          roomId={roomId}
          onToggleMic={toggleMic}
          onToggleCamera={toggleCamera}
          onJoin={handleConfirmJoin}
          onCancel={handleCancelPreview}
        />
      </div>
    );
  }

  // Error
  if (callStatus === 'failed' || errorMessage) {
    return (
      <div className="min-h-screen w-full bg-[#202124] flex items-center justify-center p-4">
        <div className="w-full max-w-md p-6 rounded-2xl bg-[#28292c] border border-[#ea4335]/30 text-center shadow-xl">
          <div className="h-12 w-12 rounded-full bg-[#ea4335]/15 text-[#ea4335] flex items-center justify-center mx-auto mb-4">
            <AlertCircle className="h-6 w-6" />
          </div>
          <h2 className="text-xl font-medium text-[#e8eaed] mb-2">Connection Error</h2>
          <p className="text-sm text-[#ea4335] mb-6 bg-[#ea4335]/10 p-3 rounded-xl">
            {errorMessage || 'Failed to join room. Please check room details and try again.'}
          </p>
          <button
            onClick={onExit}
            className="w-full py-2.5 px-4 rounded-full bg-[#3c4043] hover:bg-[#4a4d51] text-[#e8eaed] text-sm font-medium flex items-center justify-center gap-2 cursor-pointer transition-colors"
          >
            <ArrowLeft className="h-4 w-4" />
            <span>Return to Dashboard</span>
          </button>
        </div>
      </div>
    );
  }

  const participantCount = peerInfo ? 2 : 1;

  return (
    <div className="h-screen w-screen flex flex-col bg-[#202124] text-[#e8eaed] overflow-hidden">
      {/* Header */}
      <RoomHeader
        roomId={roomId}
        username={user.name}
        role={user.role}
        participantCount={participantCount}
        connectionState={connectionState}
        iceState={iceState}
      />

      {/* Main Layout */}
      <div className="flex-1 flex overflow-hidden relative">
        {/* Call Stage */}
        <div className="flex-1 flex flex-col p-2 sm:p-4 overflow-hidden relative">
          {/* Main Video Stage */}
          <div className="flex-1 w-full h-full relative rounded-2xl overflow-hidden bg-[#3c4043] shadow-md flex items-center justify-center">
            {peerInfo && remoteStream ? (
              <VideoTile
                stream={remoteStream}
                username={peerInfo.username}
                role={peerInfo.role}
                isLocal={false}
                subtitle="Remote Participant"
                className="w-full h-full"
                isDeepfakeAlert={isDeepfakeAlert}
                deepfakeScore={voiceDetection.anomalyScore}
              />
            ) : (
              <div className="relative w-full h-full flex items-center justify-center">
                <VideoTile
                  stream={localStream}
                  username={user.name}
                  role={user.role}
                  isLocal={true}
                  isMuted={isMicMuted}
                  isVideoOff={isCameraOff}
                  isFloating={false}
                  className="w-full h-full"
                  subtitle={
                    isTester && attackState.mode !== 'none'
                      ? `[${attackState.mode.toUpperCase()}]`
                      : 'You'
                  }
                />

                {/* Status chip informing user they are alone */}
                <div className="absolute top-4 left-4 z-20 flex items-center gap-2 px-3 py-1.5 rounded-full bg-[#202124]/85 backdrop-blur-md border border-[#3c4043] text-xs text-[#e8eaed] shadow-lg">
                  <span className="h-2 w-2 rounded-full bg-[#8ab4f8] animate-pulse" />
                  <span>Waiting for others to join</span>
                  <span className="text-[#9aa0a6]">•</span>
                  <span className="text-[#8ab4f8] font-mono font-medium">{roomId}</span>
                </div>
              </div>
            )}

            {/* When a peer IS present, show the local user video in the floating picture-in-picture tile */}
            {peerInfo && (
              <VideoTile
                stream={localStream}
                username={user.name}
                role={user.role}
                isLocal={true}
                isMuted={isMicMuted}
                isVideoOff={isCameraOff}
                isFloating={true}
                subtitle={
                  isTester && attackState.mode !== 'none'
                    ? `[${attackState.mode.toUpperCase()}]`
                    : 'You'
                }
              />
            )}

            {/* Attack Active */}
            {isTester && attackState.mode !== 'none' && (
              <div className="absolute top-3 left-3 sm:top-4 sm:left-4 z-30 flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-[#ea4335] text-white text-xs font-medium shadow-md">
                <span className="h-2 w-2 rounded-full bg-white animate-ping" />
                <span>
                  {attackState.mode === 'combined'
                    ? 'Full Attack Active'
                    : attackState.mode === 'face'
                    ? 'Face Swap Active'
                    : 'Voice Transform Active'}
                </span>
              </div>
            )}

            {/* Deepfake Alert */}
            {!isTester && isDeepfakeAlert && (
              <div className="absolute top-3 left-3 right-3 sm:top-4 sm:left-4 sm:right-4 z-40 flex items-center justify-between p-3.5 rounded-2xl bg-[#ea4335] text-white shadow-xl backdrop-blur-md">
                <div className="flex items-center gap-2.5">
                  <div className="h-9 w-9 rounded-full bg-white/20 flex items-center justify-center shrink-0">
                    <AlertTriangle className="h-5 w-5" />
                  </div>
                  <div>
                    <h4 className="text-sm font-semibold">Deepfake Detected!</h4>
                    <p className="text-xs text-white/90">
                      Score: {voiceDetection.anomalyScore}% — Voice anomalies found
                    </p>
                  </div>
                </div>

                <button
                  onClick={() => setIsSecurityPanelOpen(true)}
                  className="px-3.5 py-1.5 rounded-full bg-white/20 hover:bg-white/30 text-white text-xs font-medium transition-colors shrink-0 cursor-pointer"
                >
                  Details
                </button>
              </div>
            )}
          </div>

          {/* Bottom Controls */}
          <div className="pt-2 sm:pt-3 flex justify-center z-20">
            <CallControls
              isMicMuted={isMicMuted}
              isCameraOff={isCameraOff}
              isScreenSharing={isScreenSharing}
              isTester={isTester}
              isAttackDrawerOpen={isAttackDrawerOpen}
              isSecurityPanelOpen={isSecurityPanelOpen}
              attackMode={attackState.mode}
              onToggleMic={toggleMic}
              onToggleCamera={toggleCamera}
              onToggleScreenShare={toggleScreenShare}
              onEndCall={handleEndCall}
              onToggleAttackDrawer={() => {
                setIsAttackDrawerOpen(!isAttackDrawerOpen);
                if (!isAttackDrawerOpen) setIsSecurityPanelOpen(false);
              }}
              onToggleSecurityPanel={() => {
                setIsSecurityPanelOpen(!isSecurityPanelOpen);
                if (!isSecurityPanelOpen) setIsAttackDrawerOpen(false);
              }}
            />
          </div>
        </div>

        {/* Desktop Side Panels */}
        <div className="hidden lg:flex h-full shrink-0">
          <SecurityPanel
            peerAttackState={peerAttackState}
            voiceDetection={voiceDetection}
            isTester={isTester}
          />

          {isTester && isAttackDrawerOpen && (
            <AttackSimulator
              attackState={attackState}
              onToggleFaceSwap={activateFaceSwap}
              onToggleVoiceTransform={activateVoiceTransform}
              onActivateCombined={activateCombinedAttack}
              onReset={resetAttack}
              onSelectFacePreset={setFacePreset}
              onSelectVoicePreset={setVoicePreset}
              onClose={() => setIsAttackDrawerOpen(false)}
            />
          )}
        </div>
      </div>

      {/* Mobile: Security Panel */}
      {isSecurityPanelOpen && (
        <div className="fixed inset-0 z-50 lg:hidden flex flex-col justify-end bg-black/60 backdrop-blur-sm">
          <div className="w-full max-h-[85vh] h-[520px] rounded-t-2xl overflow-hidden shadow-2xl border-t border-[#3c4043] flex flex-col">
            <SecurityPanel
              peerAttackState={peerAttackState}
              voiceDetection={voiceDetection}
              isTester={isTester}
              onClose={() => setIsSecurityPanelOpen(false)}
            />
          </div>
        </div>
      )}

      {/* Mobile: Attack Simulator */}
      {isTester && isAttackDrawerOpen && (
        <div className="fixed inset-0 z-50 lg:hidden flex flex-col justify-end bg-black/60 backdrop-blur-sm">
          <div className="w-full max-h-[85vh] h-[560px] rounded-t-2xl overflow-hidden shadow-2xl border-t border-[#3c4043] flex flex-col">
            <AttackSimulator
              attackState={attackState}
              onToggleFaceSwap={activateFaceSwap}
              onToggleVoiceTransform={activateVoiceTransform}
              onActivateCombined={activateCombinedAttack}
              onReset={resetAttack}
              onSelectFacePreset={setFacePreset}
              onSelectVoicePreset={setVoicePreset}
              onClose={() => setIsAttackDrawerOpen(false)}
            />
          </div>
        </div>
      )}
    </div>
  );
};
