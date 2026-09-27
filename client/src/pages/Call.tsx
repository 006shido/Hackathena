import React, { useState, useEffect } from 'react';
import { Users, Radio, AlertCircle, ArrowLeft } from 'lucide-react';
import { User as UserType } from '../types/auth';
import { useCall } from '../hooks/useCall';
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
      <div className="min-h-screen w-full bg-[#08090d] flex items-center justify-center p-3 sm:p-4 cyber-grid overflow-y-auto">
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
      <div className="min-h-screen w-full bg-[#08090d] flex items-center justify-center p-4 cyber-grid">
        <div className="w-full max-w-md p-6 rounded-3xl bg-slate-900/95 border border-rose-500/40 text-center shadow-2xl">
          <div className="h-14 w-14 rounded-2xl bg-rose-500/20 text-rose-400 border border-rose-500/40 flex items-center justify-center mx-auto mb-4">
            <AlertCircle className="h-7 w-7" />
          </div>
          <h2 className="text-xl font-bold text-white mb-2">Connection Notice</h2>
          <p className="text-xs text-rose-300 font-mono mb-6 bg-rose-500/10 p-3 rounded-xl border border-rose-500/20">
            {errorMessage || 'Failed to join room. Please check room details and try again.'}
          </p>
          <button
            onClick={onExit}
            className="w-full py-2.5 px-4 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-mono font-semibold uppercase tracking-wider flex items-center justify-center gap-2"
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
    <div className="h-screen w-screen flex flex-col bg-[#08090d] text-slate-100 overflow-hidden">
      {/* Header */}
      <RoomHeader
        roomId={roomId}
        username={user.name}
        role={user.role}
        participantCount={participantCount}
      />

      {/* Main Workspace Layout */}
      <div className="flex-1 flex overflow-hidden relative">
        {/* Call Stage Area */}
        <div className="flex-1 flex flex-col p-2 sm:p-4 md:p-6 overflow-hidden relative">
          {/* Main Remote Video Container */}
          <div className="flex-1 w-full h-full relative rounded-2xl overflow-hidden bg-slate-950 border border-slate-800/90 shadow-2xl flex items-center justify-center">
            {peerInfo && remoteStream ? (
              <VideoTile
                stream={remoteStream}
                username={peerInfo.username}
                role={peerInfo.role}
                isLocal={false}
                subtitle="Remote Participant"
                className="w-full h-full"
              />
            ) : (
              // Waiting for Participant State (Do NOT fabricate participant!)
              <div className="flex flex-col items-center justify-center text-center p-4 sm:p-8 z-10 max-w-sm">
                <div className="relative mb-4 sm:mb-6">
                  {/* Cyber scanning ring */}
                  <div className="h-20 w-20 sm:h-28 sm:w-28 rounded-full border-2 border-cyan-500/20 flex items-center justify-center">
                    <div className="h-14 w-14 sm:h-20 sm:w-20 rounded-full border border-cyan-500/40 flex items-center justify-center animate-pulse">
                      <Users className="h-6 w-6 sm:h-8 sm:w-8 text-cyan-400" />
                    </div>
                  </div>
                  <div className="absolute inset-0 rounded-full border-t-2 border-cyan-400 animate-spin" />
                </div>

                <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 sm:px-3 sm:py-1 rounded-full bg-slate-900 border border-slate-800 text-[11px] sm:text-xs font-mono text-cyan-400 mb-2">
                  <Radio className="h-2.5 w-2.5 sm:h-3 sm:w-3 animate-ping" />
                  <span>Awaiting Connection</span>
                </div>

                <h3 className="text-base sm:text-lg font-bold text-white mb-1">
                  Waiting for another participant...
                </h3>
                <p className="text-[11px] sm:text-xs text-slate-400 leading-relaxed font-mono">
                  Share Room ID <span className="text-cyan-300 font-bold">{roomId}</span> with your peer to establish the WebRTC call.
                </p>
              </div>
            )}

            {/* Small Floating Local Video Preview */}
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

            {/* Tester Watermark Indicator if Attack is active */}
            {isTester && attackState.mode !== 'none' && (
              <div className="absolute top-2 left-2 sm:top-4 sm:left-4 z-30 flex items-center gap-1.5 px-2.5 py-1 sm:px-3 sm:py-1.5 rounded-lg sm:rounded-xl bg-rose-600/30 border border-rose-500 text-rose-300 font-mono text-[10px] sm:text-xs font-bold tracking-wider backdrop-blur-md animate-pulse">
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
          </div>

          {/* Bottom Floating Call Controls */}
          <div className="pt-2 sm:pt-4 flex justify-center z-20">
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
        <div className="fixed inset-0 z-50 lg:hidden flex flex-col justify-end bg-black/70 backdrop-blur-sm animate-fadeIn">
          <div className="w-full max-h-[85vh] h-[520px] rounded-t-3xl overflow-hidden shadow-2xl border-t border-cyan-500/30 flex flex-col">
            <SecurityPanel
              peerAttackState={peerAttackState}
              isTester={isTester}
              onClose={() => setIsSecurityPanelOpen(false)}
            />
          </div>
        </div>
      )}

      {/* Mobile Drawer 2: Tester Attack Simulator */}
      {isTester && isAttackDrawerOpen && (
        <div className="fixed inset-0 z-50 lg:hidden flex flex-col justify-end bg-black/70 backdrop-blur-sm animate-fadeIn">
          <div className="w-full max-h-[85vh] h-[560px] rounded-t-3xl overflow-hidden shadow-2xl border-t border-amber-500/40 flex flex-col">
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
