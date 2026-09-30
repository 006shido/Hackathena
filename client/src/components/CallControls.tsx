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
    <div className="flex items-center justify-center gap-3 py-2 px-4 rounded-full bg-[#202124] max-w-full">
      {/* Microphone */}
      <button
        onClick={onToggleMic}
        title={isMicMuted ? 'Turn on microphone' : 'Turn off microphone'}
        className={`flex h-11 w-11 items-center justify-center rounded-full transition-colors shrink-0 cursor-pointer ${
          isMicMuted
            ? 'bg-[#ea4335] text-white hover:bg-[#d93025]'
            : 'bg-[#3c4043] text-white hover:bg-[#4a4d51]'
        }`}
      >
        {isMicMuted ? <MicOff className="h-5 w-5" /> : <Mic className="h-5 w-5" />}
      </button>

      {/* Camera */}
      <button
        onClick={onToggleCamera}
        title={isCameraOff ? 'Turn on camera' : 'Turn off camera'}
        className={`flex h-11 w-11 items-center justify-center rounded-full transition-colors shrink-0 cursor-pointer ${
          isCameraOff
            ? 'bg-[#ea4335] text-white hover:bg-[#d93025]'
            : 'bg-[#3c4043] text-white hover:bg-[#4a4d51]'
        }`}
      >
        {isCameraOff ? <VideoOff className="h-5 w-5" /> : <Video className="h-5 w-5" />}
      </button>

      {/* Screen Share */}
      <button
        onClick={onToggleScreenShare}
        title={isScreenSharing ? 'Stop presenting' : 'Present now'}
        className={`hidden xs:flex h-11 w-11 items-center justify-center rounded-full transition-colors shrink-0 cursor-pointer ${
          isScreenSharing
            ? 'bg-[#8ab4f8] text-[#202124] hover:bg-[#aecbfa]'
            : 'bg-[#3c4043] text-white hover:bg-[#4a4d51]'
        }`}
      >
        <ScreenShare className="h-5 w-5" />
      </button>

      {/* Security Monitor - mobile */}
      {onToggleSecurityPanel && (
        <button
          onClick={onToggleSecurityPanel}
          title="Security Monitor"
          className={`flex lg:hidden h-11 w-11 items-center justify-center rounded-full transition-colors shrink-0 cursor-pointer ${
            isSecurityPanelOpen
              ? 'bg-[#8ab4f8] text-[#202124]'
              : 'bg-[#3c4043] text-white hover:bg-[#4a4d51]'
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
          className={`flex h-11 w-11 items-center justify-center rounded-full transition-colors shrink-0 cursor-pointer relative ${
            isAttackActive
              ? 'bg-[#ea4335] text-white hover:bg-[#d93025]'
              : isAttackDrawerOpen
              ? 'bg-[#8ab4f8] text-[#202124]'
              : 'bg-[#3c4043] text-white hover:bg-[#4a4d51]'
          }`}
        >
          <ShieldAlert className="h-5 w-5" />
          {isAttackActive && (
            <span className="absolute top-1 right-1 h-2 w-2 rounded-full bg-white animate-ping" />
          )}
        </button>
      )}

      {/* End Call - Google Meet style pill */}
      <button
        onClick={onEndCall}
        title="Leave call"
        className="flex h-11 px-5 items-center justify-center rounded-full bg-[#ea4335] hover:bg-[#d93025] text-white transition-colors active:scale-95 shrink-0 cursor-pointer"
      >
        <PhoneOff className="h-5 w-5" />
      </button>
    </div>
  );
};
