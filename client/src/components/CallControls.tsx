import React from 'react';
import { Mic, MicOff, Video, VideoOff, ScreenShare, PhoneOff, ShieldAlert, Activity, Sparkles } from 'lucide-react';

interface CallControlsProps {
  isMicMuted: boolean;
  isCameraOff: boolean;
  isScreenSharing: boolean;
  isTester: boolean;
  isTesterDrawerOpen: boolean;
  isSecurityPanelOpen?: boolean;
  attackMode: string;
  faceSwapActive?: boolean;
  onToggleMic: () => void;
  onToggleCamera: () => void;
  onToggleScreenShare: () => void;
  onEndCall: () => void;
  onToggleTesterDrawer?: () => void;
  onToggleSecurityPanel?: () => void;
}

export const CallControls: React.FC<CallControlsProps> = ({
  isMicMuted,
  isCameraOff,
  isScreenSharing,
  isTester,
  isTesterDrawerOpen,
  isSecurityPanelOpen,
  attackMode,
  faceSwapActive,
  onToggleMic,
  onToggleCamera,
  onToggleScreenShare,
  onEndCall,
  onToggleTesterDrawer,
  onToggleSecurityPanel,
}) => {
  const isAttackActive = attackMode !== 'none';

  return (
    <div className="flex items-center justify-center gap-2 sm:gap-3 py-1.5 sm:py-2 px-2.5 sm:px-4 rounded-full bg-[#1c1f26]/90 backdrop-blur-md border border-white/10 shadow-2xl max-w-full">
      {/* Microphone */}
      <button
        onClick={onToggleMic}
        title={isMicMuted ? 'Turn on microphone' : 'Turn off microphone'}
        className={`flex h-10 w-10 sm:h-11 sm:w-11 items-center justify-center rounded-full transition-all shrink-0 cursor-pointer ${isMicMuted
            ? 'bg-red-500/90 hover:bg-red-600 text-white shadow-md shadow-red-500/20'
            : 'bg-white/10 hover:bg-white/20 text-white'
          }`}
      >
        {isMicMuted ? <MicOff className="h-4 w-4 sm:h-5 sm:w-5" /> : <Mic className="h-4 w-4 sm:h-5 sm:w-5" />}
      </button>

      {/* Camera */}
      <button
        onClick={onToggleCamera}
        title={isCameraOff ? 'Turn on camera' : 'Turn off camera'}
        className={`flex h-10 w-10 sm:h-11 sm:w-11 items-center justify-center rounded-full transition-all shrink-0 cursor-pointer ${isCameraOff
            ? 'bg-red-500/90 hover:bg-red-600 text-white shadow-md shadow-red-500/20'
            : 'bg-white/10 hover:bg-white/20 text-white'
          }`}
      >
        {isCameraOff ? <VideoOff className="h-4 w-4 sm:h-5 sm:w-5" /> : <Video className="h-4 w-4 sm:h-5 sm:w-5" />}
      </button>

      {/* Screen Share (Hidden on small mobile where screen sharing is not practical) */}
      <button
        onClick={onToggleScreenShare}
        title={isScreenSharing ? 'Stop presenting' : 'Present now'}
        className={`hidden sm:flex h-10 w-10 sm:h-11 sm:w-11 items-center justify-center rounded-full transition-all shrink-0 cursor-pointer ${isScreenSharing
            ? 'bg-blue-600 text-white shadow-md shadow-blue-500/25'
            : 'bg-white/10 hover:bg-white/20 text-white'
          }`}
      >
        <ScreenShare className="h-4 w-4 sm:h-5 sm:w-5" />
      </button>

      {/* Security Monitor */}
      {onToggleSecurityPanel && (
        <button
          onClick={onToggleSecurityPanel}
          title="Security Monitor & Visualizer"
          className={`flex h-10 w-10 sm:h-11 sm:w-11 items-center justify-center rounded-full transition-all shrink-0 cursor-pointer ${isSecurityPanelOpen
              ? 'bg-blue-600 text-white shadow-md shadow-blue-500/25'
              : 'bg-white/10 hover:bg-white/20 text-white'
            }`}
        >
          <Activity className="h-4 w-4 sm:h-5 sm:w-5" />
        </button>
      )}

      {/* Tester: Integrated AI Face Swap & Attack Simulation Button (Single Button) */}
      {isTester && onToggleTesterDrawer && (
        <button
          onClick={onToggleTesterDrawer}
          title="AI Face Swap & Attack Simulator (Tester Only)"
          className={`flex h-10 w-10 sm:h-11 sm:w-11 items-center justify-center rounded-full transition-all shrink-0 cursor-pointer relative ${isAttackActive || faceSwapActive
              ? 'bg-red-500 text-white shadow-md shadow-red-500/30'
              : isTesterDrawerOpen
                ? 'bg-blue-600 text-white'
                : 'bg-white/10 hover:bg-white/20 text-white'
            }`}
        >
          <ShieldAlert className="h-4 w-4 sm:h-5 sm:w-5" />
          {(isAttackActive || faceSwapActive) && (
            <span className="absolute top-1 right-1 h-2 w-2 rounded-full bg-white animate-ping" />
          )}
        </button>
      )}

      {/* End Call (Red Pill/Circle) */}
      <button
        onClick={onEndCall}
        title="Leave call"
        className="flex h-10 px-4 sm:h-11 sm:px-5 items-center justify-center rounded-full bg-red-600 hover:bg-red-700 text-white transition-all active:scale-95 shrink-0 cursor-pointer shadow-md shadow-red-600/30"
      >
        <PhoneOff className="h-4 w-4 sm:h-5 sm:w-5" />
      </button>
    </div>
  );
};
