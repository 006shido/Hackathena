import { AlertCircle,AlertTriangle,ArrowLeft } from 'lucide-react';
import React,{ useEffect,useRef,useState } from 'react';
import { AIFaceSwapPanel } from '../components/AIFaceSwapPanel';
import { CallControls } from '../components/CallControls';
import { CallFeedbackModal } from '../components/CallFeedbackModal';
import { DevicePreview } from '../components/DevicePreview';
import { RoomHeader } from '../components/RoomHeader';
import { SecurityPanel } from '../components/SecurityPanel';
import { VideoTile } from '../components/VideoTile';
import { useCall } from '../hooks/useCall';
import { useMediaDetection } from '../hooks/useMediaDetection';
import { useVoiceDetection } from '../hooks/useVoiceDetection';
import { mediaService } from '../services/media';
import { User as UserType } from '../types/auth';

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
  const [showFeedbackModal, setShowFeedbackModal] = useState(false);

  // Desktop default: Panels closed so video takes full screen
  const [isSecurityPanelOpen, setIsSecurityPanelOpen] = useState(false);
  const [isTesterDrawerOpen, setIsTesterDrawerOpen] = useState(false);
  const [showEndCallSlider, setShowEndCallSlider] = useState(false);

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
    uploadGalleryFace,
    resetFaceSwapFace,
    selectedFacePreview,
    selectedFaceName,
    faceSwapTelemetry,
    faceBlendConfig,
    updateFaceBlendConfig,
    isVoiceMonitoring,
    toggleVoiceMonitor,
    endCall,
  } = useCall(user, token);

  // Experimental acoustic diagnostics on incoming peer audio, separate from learned models.
  const voiceDetection = useVoiceDetection(
    remoteStream,
    !showDevicePreview && Boolean(remoteStream)
  );

  // Audio anomaly alert triggered by the browser diagnostic heuristic.
  const isDeepfakeAlert = voiceDetection.status === 'deepfake';
  const mediaDetection = useMediaDetection(remoteStream, token, !showDevicePreview && Boolean(remoteStream));

  useEffect(() => {
    initLocalMedia(true, true);
  }, [initLocalMedia]);

  const endCallRef = useRef(endCall);
  useEffect(() => {
    endCallRef.current = endCall;
  }, [endCall]);

  useEffect(() => {
    const handleBeforeUnload = () => {
      endCallRef.current();
      mediaService.stopAllMedia();
    };
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
      endCallRef.current();
      mediaService.stopAllMedia();
    };
  }, []);

  const handleConfirmJoin = () => {
    setShowDevicePreview(false);
    joinRoom(roomId);
  };

  const handleRequestEndCall = () => {
    setShowEndCallSlider(true);
  };

  const handleConfirmEndCall = () => {
    setShowEndCallSlider(false);
    endCall();
    mediaService.stopAllMedia();
    setShowFeedbackModal(true);
  };

  const handleCancelPreview = () => {
    endCall();
    mediaService.stopAllMedia();
    onExit();
  };

  const handleFeedbackSubmit = (feedback: { rating: number }) => {
    console.log('[Feedback] User rated experience:', feedback);
    setShowFeedbackModal(false);
    onExit();
  };

  const handleFeedbackSkip = () => {
    setShowFeedbackModal(false);
    onExit();
  };

  // Feedback Rating Screen (After call is cut / ended)
  if (showFeedbackModal) {
    return (
      <div className="h-screen w-screen flex items-center justify-center bg-[#f8fafc] p-4">
        <CallFeedbackModal
          isOpen={true}
          roomId={roomId}
          onSubmit={handleFeedbackSubmit}
          onSkip={handleFeedbackSkip}
        />
      </div>
    );
  }

  // Device Preview Screen
  if (showDevicePreview) {
    return (
      <div className="min-h-screen w-full bg-[#f8fafc] text-slate-900 flex items-center justify-center p-3 sm:p-4 overflow-y-auto animate-fade-in">
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

  // Error Screen
  if (callStatus === 'failed' || errorMessage) {
    return (
      <div className="min-h-screen w-full bg-[#0f1115] flex items-center justify-center p-4">
        <div className="w-full max-w-md p-6 rounded-3xl bg-[#16181f] border border-red-500/30 text-center shadow-2xl">
          <div className="h-12 w-12 rounded-full bg-red-500/15 text-red-500 flex items-center justify-center mx-auto mb-4">
            <AlertCircle className="h-6 w-6" />
          </div>
          <h2 className="text-xl font-bold text-white mb-2">Connection Error</h2>
          <p className="text-xs sm:text-sm text-red-400 mb-6 bg-red-500/10 p-3 rounded-2xl border border-red-500/20">
            {errorMessage || 'Failed to join room. Please check room details and try again.'}
          </p>
          <button
            onClick={onExit}
            className="w-full py-3 px-4 rounded-full bg-white/10 hover:bg-white/20 text-white text-sm font-medium flex items-center justify-center gap-2 cursor-pointer transition-colors"
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
    <div className="h-[100dvh] w-full flex flex-col bg-[#0f1115] text-slate-100 overflow-hidden font-sans">
      {/* Top Header (Matching Screenshot 1 Right) */}
      <RoomHeader
        roomId={roomId}
        username={user.name}
        role={user.role}
        participantCount={participantCount}
        connectionState={connectionState}
        iceState={iceState}
        onExit={handleRequestEndCall}
        onToggleSecurityPanel={() => setIsSecurityPanelOpen(!isSecurityPanelOpen)}
        isSecurityPanelOpen={isSecurityPanelOpen}
      />

      {/* Main Layout Area */}
      <div className="flex-1 flex overflow-hidden relative">
        {/* Call Stage */}
        <div className="flex-1 flex flex-col p-1.5 sm:p-3 lg:p-4 overflow-hidden relative">
          {/* Video Container (Framed with smooth rounded corners) */}
          <div className="flex-1 w-full h-full relative rounded-2xl sm:rounded-3xl overflow-hidden bg-[#16181d] border border-white/5 shadow-2xl flex items-center justify-center">
            {peerInfo && remoteStream ? (
              <VideoTile
                stream={remoteStream}
                username={peerInfo.username}
                role={peerInfo.role}
                isLocal={false}
                subtitle="Peer"
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
                      : 'Your outgoing camera'
                  }
                />
              </div>
            )}

            {/* Top-Left Stage HUD: Waiting status and Active Attack mode */}
            <div className="absolute top-3 left-3 sm:top-4 sm:left-4 z-30 flex items-center gap-2 pointer-events-none">
              {!peerInfo && (
                <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-black/65 backdrop-blur-md border border-white/10 text-[11px] text-white/95 shadow-md pointer-events-auto">
                  <span className="h-1.5 w-1.5 rounded-full bg-blue-500 animate-pulse" />
                  <span>Waiting for peer to join</span>
                  <span className="text-slate-500">•</span>
                  <span className="text-blue-400 font-mono font-medium">{roomId}</span>
                </div>
              )}

              {isTester && attackState.mode !== 'none' && (
                <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-red-600 text-white text-[11px] font-medium shadow-md shadow-red-600/30 pointer-events-auto">
                  <span className="h-1.5 w-1.5 rounded-full bg-white animate-ping" />
                  <span>
                    {attackState.mode === 'combined'
                      ? 'Full Attack Active'
                      : attackState.mode === 'face'
                        ? 'Face Swap Active'
                        : 'Voice Transform Active'}
                  </span>
                </div>
              )}
            </div>

            {/* When Peer is present: Floating PiP Local User Video (Matching Screenshot 1 Right) */}
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
                    : 'Your outgoing camera'
                }
              />
            )}

            {/* Autonomous Deepfake Alert Floating Banner */}
            {!isTester && isDeepfakeAlert && (
              <div className="absolute top-3 inset-x-3 sm:top-4 sm:inset-x-6 z-40 flex items-center justify-between p-3.5 rounded-2xl bg-red-600/90 text-white shadow-2xl backdrop-blur-md border border-red-500">
                <div className="flex items-center gap-2.5">
                  <div className="h-9 w-9 rounded-full bg-white/20 flex items-center justify-center shrink-0">
                    <AlertTriangle className="h-5 w-5" />
                  </div>
                  <div>
                    <h4 className="text-xs sm:text-sm font-bold">Audio Anomaly Detected</h4>
                    <p className="text-[11px] text-white/90">
                      Anomaly Score: {voiceDetection.anomalyScore}% — Synthetic audio harmonics detected
                    </p>
                  </div>
                </div>

                <button
                  onClick={() => setIsSecurityPanelOpen(true)}
                  className="px-3.5 py-1.5 rounded-full bg-white text-red-700 text-xs font-semibold hover:bg-white/90 transition-all shrink-0 cursor-pointer shadow-sm"
                >
                  View Details
                </button>
              </div>
            )}
          </div>

          {/* Floating Bottom Controls (Matching Screenshot 1 Right) */}
          <div className="pt-3 sm:pt-4 flex justify-center z-20">
            <CallControls
              isMicMuted={isMicMuted}
              isCameraOff={isCameraOff}
              isScreenSharing={isScreenSharing}
              isTester={isTester}
              isTesterDrawerOpen={isTesterDrawerOpen}
              isSecurityPanelOpen={isSecurityPanelOpen}
              attackMode={attackState.mode}
              faceSwapActive={attackState.faceSwap}
              onToggleMic={toggleMic}
              onToggleCamera={toggleCamera}
              onToggleScreenShare={toggleScreenShare}
              isSliderActive={showEndCallSlider}
              onToggleSlider={setShowEndCallSlider}
              onEndCall={handleConfirmEndCall}
              onToggleTesterDrawer={() => {
                setIsTesterDrawerOpen(!isTesterDrawerOpen);
                if (!isTesterDrawerOpen) {
                  setIsSecurityPanelOpen(false);
                }
              }}
              onToggleSecurityPanel={() => {
                setIsSecurityPanelOpen(!isSecurityPanelOpen);
                if (!isSecurityPanelOpen) {
                  setIsTesterDrawerOpen(false);
                }
              }}
            />
          </div>
        </div>

        {/* Desktop Side Panels (Slide in seamlessly) */}
        <div className="hidden lg:flex h-full shrink-0">
          {isSecurityPanelOpen && (
            <SecurityPanel
              mediaDetection={mediaDetection}
              hasIncomingPeer={Boolean(peerInfo)}
              peerAttackState={peerAttackState}
              voiceDetection={voiceDetection}
              isTester={isTester}
              onClose={() => setIsSecurityPanelOpen(false)}
            />
          )}

          {isTester && isTesterDrawerOpen && (
            <AIFaceSwapPanel
              faceSwapActive={attackState.faceSwap}
              onToggleFaceSwap={activateFaceSwap}
              onUploadGalleryFace={uploadGalleryFace}
              onResetFace={resetFaceSwapFace}
              onSelectPreset={setFacePreset}
              currentPreset={attackState.facePreset}
              telemetry={faceSwapTelemetry}
              blendConfig={faceBlendConfig}
              onUpdateBlendConfig={updateFaceBlendConfig}
              selectedFacePreview={selectedFacePreview}
              selectedFaceName={selectedFaceName}
              processedStream={localStream}
              voiceTransformActive={attackState.voiceTransform}
              onToggleVoiceTransform={activateVoiceTransform}
              onActivateCombined={activateCombinedAttack}
              onResetAll={resetAttack}
              onSelectVoicePreset={setVoicePreset}
              voicePreset={attackState.voicePreset}
              attackMode={attackState.mode}
              isVoiceMonitoring={isVoiceMonitoring}
              onToggleVoiceMonitor={toggleVoiceMonitor}
              onClose={() => setIsTesterDrawerOpen(false)}
            />
          )}
        </div>
      </div>

      {/* Mobile Bottom Sheets: Security Panel */}
      {isSecurityPanelOpen && (
        <div className="fixed inset-0 z-50 lg:hidden flex flex-col justify-end bg-black/60 backdrop-blur-xs">
          <div className="w-full max-h-[85dvh] h-[520px] rounded-t-3xl overflow-hidden shadow-2xl border-t border-white/10 flex flex-col bg-[#16181f]">
            {/* Grab handle */}
            <div className="w-12 h-1 bg-white/20 rounded-full mx-auto my-2.5 shrink-0" />
            <div className="flex-1 overflow-y-auto min-h-0">
              <SecurityPanel
                mediaDetection={mediaDetection}
                hasIncomingPeer={Boolean(peerInfo)}
                peerAttackState={peerAttackState}
                voiceDetection={voiceDetection}
                isTester={isTester}
                onClose={() => setIsSecurityPanelOpen(false)}
              />
            </div>
          </div>
        </div>
      )}

      {/* Mobile Bottom Sheets: Integrated AI Face Swap & Tester Panel */}
      {isTester && isTesterDrawerOpen && (
        <div className="fixed inset-0 z-50 lg:hidden flex flex-col justify-end bg-black/60 backdrop-blur-xs">
          <div className="w-full max-h-[85dvh] h-[580px] rounded-t-3xl overflow-hidden shadow-2xl border-t border-white/10 flex flex-col bg-[#16181f]">
            {/* Grab handle */}
            <div className="w-12 h-1 bg-white/20 rounded-full mx-auto my-2.5 shrink-0" />
            <div className="flex-1 overflow-y-auto min-h-0">
              <AIFaceSwapPanel
                faceSwapActive={attackState.faceSwap}
                onToggleFaceSwap={activateFaceSwap}
                onUploadGalleryFace={uploadGalleryFace}
                onResetFace={resetFaceSwapFace}
                onSelectPreset={setFacePreset}
                currentPreset={attackState.facePreset}
                telemetry={faceSwapTelemetry}
                blendConfig={faceBlendConfig}
                onUpdateBlendConfig={updateFaceBlendConfig}
                selectedFacePreview={selectedFacePreview}
                selectedFaceName={selectedFaceName}
                processedStream={localStream}
                voiceTransformActive={attackState.voiceTransform}
                onToggleVoiceTransform={activateVoiceTransform}
                onActivateCombined={activateCombinedAttack}
                onResetAll={resetAttack}
                onSelectVoicePreset={setVoicePreset}
                voicePreset={attackState.voicePreset}
                attackMode={attackState.mode}
                isVoiceMonitoring={isVoiceMonitoring}
                onToggleVoiceMonitor={toggleVoiceMonitor}
                onClose={() => setIsTesterDrawerOpen(false)}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
