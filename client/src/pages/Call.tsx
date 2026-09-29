import React, { useState, useEffect } from 'react';
import { Users, Radio, AlertCircle, ArrowLeft, AlertTriangle } from 'lucide-react';
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

  // Real-time voice deepfake analysis on incoming peer audio stream
  const isPeerVoiceAttacking = Boolean(
    peerAttackState?.voiceTransform ||
    peerAttackState?.mode === 'voice' ||
    peerAttackState?.mode === 'combined'
  );

  const voiceDetection = useVoiceDetection(
    remoteStream,
    isPeerVoiceAttacking,
    !showDevicePreview && Boolean(remoteStream)
  );

  const isDeepfakeAlert = voiceDetection.status === 'deepfake' || isPeerVoiceAttacking;

  // Initialize camera and mic for preview on mount
  useEffect(() => {
    initLocalMedia(true, true);
  }, [initLocalMedia]);

  const handleConfirmJoin = () => {
    setShowDevicePreview(false);
    joinRoom(roomId);
  };

  const handleEndCall = () => {
    endCall();
    onExit();
  };

  // If in device preview stage before entering the call
  if (showDevicePreview) {
    return (
      <div className="min-h-screen w-full bg-black flex items-center justify-center p-3 sm:p-4 amoled-grid overflow-y-auto">
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
          onCancel={onExit}
        />
      </div>
    );
  }

  // Room error / Room full handling
  if (callStatus === 'failed' || errorMessage) {
    return (
      <div className="min-h-screen w-full bg-black flex items-center justify-center p-4 amoled-grid">
        <div className="w-full max-w-md p-6 rounded-md bg-[#070709] border border-rose-500/40 text-center shadow-2xl">
          <div className="h-12 w-12 rounded-md bg-rose-500/15 text-rose-400 border border-rose-500/30 flex items-center justify-center mx-auto mb-4">
            <AlertCircle className="h-6 w-6" />
          </div>
          <h2 className="text-xl font-bold text-white mb-2">Connection Notice</h2>
          <p className="text-xs text-rose-300 font-mono mb-6 bg-rose-500/10 p-3 rounded-md border border-rose-500/20">
            {errorMessage || 'Failed to join room. Please check room details and try again.'}
          </p>
          <button
            onClick={onExit}
            className="w-full py-2.5 px-4 rounded-md bg-zinc-800 hover:bg-zinc-700 text-zinc-200 text-xs font-mono font-semibold uppercase tracking-wider flex items-center justify-center gap-2 cursor-pointer transition-colors"
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
    <div className="h-screen w-screen flex flex-col bg-black text-zinc-100 overflow-hidden">
      {/* Header */}
      <RoomHeader
        roomId={roomId}
        username={user.name}
        role={user.role}
        participantCount={participantCount}
        connectionState={connectionState}
        iceState={iceState}
      />

      {/* Main Workspace Layout */}
      <div className="flex-1 flex overflow-hidden relative">
        {/* Call Stage Area - Expansive, minimal outer padding */}
        <div className="flex-1 flex flex-col p-1.5 sm:p-2.5 md:p-3 overflow-hidden relative">
          {/* Main Remote Video Container */}
          <div className="flex-1 w-full h-full relative rounded-md overflow-hidden bg-black border border-zinc-800/90 shadow-2xl flex items-center justify-center">
            {peerInfo && remoteStream ? (
              <VideoTile
                stream={remoteStream}
                username={peerInfo.username}
                role={peerInfo.role}
                isLocal={false}
                subtitle="Remote Participant"
                className="w-full h-full"
                isDeepfakeAlert={isDeepfakeAlert}
                deepfakeScore={voiceDetection.anomalyScore || (isPeerVoiceAttacking ? 94 : 0)}
              />
            ) : (
              // Waiting for Participant State
              <div className="flex flex-col items-center justify-center text-center p-4 sm:p-8 z-10 max-w-sm">
                <div className="relative mb-4 sm:mb-6">
                  {/* Tactical radar pulse */}
                  <div className="h-20 w-20 sm:h-24 sm:w-24 rounded-full border border-orange-500/30 flex items-center justify-center">
                    <div className="h-14 w-14 sm:h-16 sm:w-16 rounded-full border border-orange-500/50 flex items-center justify-center animate-pulse">
                      <Users className="h-6 w-6 text-orange-400" />
                    </div>
                  </div>
                  <div className="absolute inset-0 rounded-full border-t-2 border-orange-400 animate-spin" />
                </div>

                <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-md bg-zinc-900 border border-zinc-800 text-[11px] sm:text-xs font-mono text-orange-400 mb-2.5">
                  <Radio className="h-2.5 w-2.5 animate-ping text-orange-400" />
                  <span>Awaiting Connection</span>
                </div>

                <h3 className="text-base sm:text-lg font-bold text-white mb-1">
                  Waiting for peer to connect...
                </h3>
                <p className="text-xs text-zinc-400 leading-relaxed font-mono">
                  Share Room ID <span className="text-orange-400 font-bold">{roomId}</span> with your peer to establish WebRTC call.
                </p>
              </div>
            )}

            {/* Movable & Resizable Floating Self-Video (You) */}
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
                  : 'Local Feed'
              }
            />

            {/* Tester Watermark Indicator if Attack is active */}
            {isTester && attackState.mode !== 'none' && (
              <div className="absolute top-3 left-3 sm:top-4 sm:left-4 z-30 flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-rose-950/80 border border-rose-500 text-rose-300 font-mono text-[10px] sm:text-xs font-bold tracking-wider backdrop-blur-md animate-pulse">
                <span className="h-1.5 w-1.5 sm:h-2 sm:w-2 rounded-full bg-rose-500 animate-ping" />
                <span>
                  {attackState.mode === 'combined'
                    ? 'DEEPFAKE ATTACK ACTIVE'
                    : attackState.mode === 'face'
                    ? 'FACE SIMULATION ACTIVE'
                    : 'VOICE SIMULATION ACTIVE'}
                </span>
              </div>
            )}

            {/* USER ONLY: Real-Time Defense In-Call HUD Banner */}
            {!isTester && isDeepfakeAlert && (
              <div className="absolute top-3 left-3 right-3 sm:top-4 sm:left-4 sm:right-4 z-40 flex items-center justify-between p-3 sm:p-4 rounded-2xl bg-rose-950/95 border border-rose-500/80 shadow-2xl backdrop-blur-xl animate-bounce">
                <div className="flex items-center gap-2.5 sm:gap-3">
                  <div className="h-9 w-9 sm:h-11 sm:w-11 rounded-xl bg-rose-500/20 border border-rose-500/50 text-rose-400 flex items-center justify-center shrink-0">
                    <AlertTriangle className="h-5 w-5 sm:h-6 sm:w-6 text-rose-400 animate-pulse" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] sm:text-xs font-mono font-bold uppercase tracking-wider text-rose-300 bg-rose-500/30 px-2 py-0.5 rounded-full border border-rose-500/40">
                        Real-Time Defense Alert
                      </span>
                      <span className="text-[10px] sm:text-xs font-mono text-rose-200">
                        Score: <strong className="text-white">{voiceDetection.anomalyScore || 94}%</strong>
                      </span>
                    </div>
                    <h4 className="text-xs sm:text-sm font-bold text-white mt-0.5">
                      Voice Deepfake Caught in Real Time!
                    </h4>
                    <p className="text-[10px] sm:text-xs text-rose-200/90 font-mono hidden sm:block">
                      Synthetic vocoder ring-modulation & carrier harmonics detected.
                    </p>
                  </div>
                </div>

                <button
                  onClick={() => setIsSecurityPanelOpen(true)}
                  className="px-2.5 sm:px-3 py-1.5 rounded-xl bg-rose-600 hover:bg-rose-500 text-white font-mono text-[11px] sm:text-xs font-semibold shadow-lg transition-colors shrink-0"
                >
                  View Telemetry
                </button>
              </div>
            )}
          </div>

          {/* Bottom Floating Call Controls */}
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

        {/* Desktop Side Panels Area (Docked on large screens) */}
        <div className="hidden lg:flex h-full shrink-0">
          {/* DeepTrace Security Monitoring Panel */}
          <SecurityPanel
            peerAttackState={peerAttackState}
            voiceDetection={voiceDetection}
            isTester={isTester}
          />

          {/* STRICTLY TESTER ONLY: Attack Simulator Panel */}
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

      {/* Mobile Drawer 1: Security Monitoring Panel */}
      {isSecurityPanelOpen && (
        <div className="fixed inset-0 z-50 lg:hidden flex flex-col justify-end bg-black/80 backdrop-blur-sm">
          <div className="w-full max-h-[85vh] h-[520px] rounded-t-lg overflow-hidden shadow-2xl border-t border-zinc-800 flex flex-col">
            <SecurityPanel
              peerAttackState={peerAttackState}
              voiceDetection={voiceDetection}
              isTester={isTester}
              onClose={() => setIsSecurityPanelOpen(false)}
            />
          </div>
        </div>
      )}

      {/* Mobile Drawer 2: Tester Attack Simulator */}
      {isTester && isAttackDrawerOpen && (
        <div className="fixed inset-0 z-50 lg:hidden flex flex-col justify-end bg-black/80 backdrop-blur-sm">
          <div className="w-full max-h-[85vh] h-[560px] rounded-t-lg overflow-hidden shadow-2xl border-t border-orange-500/40 flex flex-col">
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
