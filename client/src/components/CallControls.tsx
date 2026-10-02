import React from 'react';
import { Mic, MicOff, Video, VideoOff, ScreenShare, PhoneOff, ShieldAlert, Activity } from 'lucide-react';

interface CallControlsProps {
  isMicMuted: boolean;
  isCameraOff: boolean;
  isScreenSharing: boolean;
  isTester: boolean;
  isAttackDrawerOpen: boolean;
  isSecurityPanelOpen?: boolean;
  attackMode: string;
  onToggleMic: () => void;
  onToggleCamera: () => void;
  onToggleScreenShare: () => void;
  onEndCall: () => void;
  onToggleAttackDrawer?: () => void;
  onToggleSecurityPanel?: () => void;
}

export const CallControls: React.FC<CallControlsProps> = ({
  isMicMuted,
  isCameraOff,
  isScreenSharing,
  isTester,
  isAttackDrawerOpen,
  isSecurityPanelOpen,
  attackMode,
  onToggleMic,
  onToggleCamera,
  onToggleScreenShare,
  onEndCall,
  onToggleAttackDrawer,
  onToggleSecurityPanel,
}) => {
  const isAttackActive = attackMode !== 'none';

  return (
    <div className="flex items-center justify-center gap-2.5 sm:gap-3 py-2 px-3 sm:px-4 rounded-full bg-[#1c1f26]/90 backdrop-blur-md border border-white/10 shadow-2xl max-w-full">
      {/* Microphone */}
      <button
        onClick={onToggleMic}
        title={isMicMuted ? 'Turn on microphone' : 'Turn off microphone'}
        className={`flex h-11 w-11 items-center justify-center rounded-full transition-all shrink-0 cursor-pointer ${
          isMicMuted
            ? 'bg-red-500/90 hover:bg-red-600 text-white shadow-md shadow-red-500/20'
            : 'bg-white/10 hover:bg-white/20 text-white'
        }`}
      >
        {isMicMuted ? <MicOff className="h-5 w-5" /> : <Mic className="h-5 w-5" />}
      </button>

      {/* Camera */}
      <button
        onClick={onToggleCamera}
        title={isCameraOff ? 'Turn on camera' : 'Turn off camera'}
        className={`flex h-11 w-11 items-center justify-center rounded-full transition-all shrink-0 cursor-pointer ${
          isCameraOff
            ? 'bg-red-500/90 hover:bg-red-600 text-white shadow-md shadow-red-500/20'
            : 'bg-white/10 hover:bg-white/20 text-white'
        }`}
      >
        {isCameraOff ? <VideoOff className="h-5 w-5" /> : <Video className="h-5 w-5" />}
      </button>

      {/* Screen Share */}
      <button
        onClick={onToggleScreenShare}
        title={isScreenSharing ? 'Stop presenting' : 'Present now'}
        className={`hidden sm:flex h-11 w-11 items-center justify-center rounded-full transition-all shrink-0 cursor-pointer ${
          isScreenSharing
            ? 'bg-blue-600 text-white shadow-md shadow-blue-500/25'
            : 'bg-white/10 hover:bg-white/20 text-white'
        }`}
      >
        <ScreenShare className="h-5 w-5" />
      </button>

      {/* Security Monitor */}
      {onToggleSecurityPanel && (
        <button
          onClick={onToggleSecurityPanel}
          title="Security Monitor & Visualizer"
          className={`flex h-11 w-11 items-center justify-center rounded-full transition-all shrink-0 cursor-pointer ${
            isSecurityPanelOpen
              ? 'bg-blue-600 text-white shadow-md shadow-blue-500/25'
              : 'bg-white/10 hover:bg-white/20 text-white'
          }`}
        >
          <Activity className="h-5 w-5" />
        </button>
      )}

      {/* Tester: Attack Simulator */}
      {isTester && onToggleAttackDrawer && (
        <button
          onClick={onToggleAttackDrawer}
          title="Attack Simulator"
          className={`flex h-11 w-11 items-center justify-center rounded-full transition-all shrink-0 cursor-pointer relative ${
            isAttackActive
              ? 'bg-red-500 text-white shadow-md shadow-red-500/30'
              : isAttackDrawerOpen
              ? 'bg-blue-600 text-white'
              : 'bg-white/10 hover:bg-white/20 text-white'
          }`}
        >
          <ShieldAlert className="h-5 w-5" />
          {isAttackActive && (
            <span className="absolute top-1 right-1 h-2 w-2 rounded-full bg-white animate-ping" />
          )}
        </button>
      )}

      {/* End Call (Red Pill/Circle) */}
      <button
        onClick={onEndCall}
        title="Leave call"
        className="flex h-11 px-5 items-center justify-center rounded-full bg-red-600 hover:bg-red-700 text-white transition-all active:scale-95 shrink-0 cursor-pointer shadow-md shadow-red-600/30"
      >
        <PhoneOff className="h-5 w-5" />
      </button>
    </div>
  );
};
