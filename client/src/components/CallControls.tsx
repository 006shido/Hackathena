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
    <div className="flex items-center justify-center gap-2 sm:gap-3 py-2 sm:py-2.5 px-3 sm:px-4 rounded-md bg-[#070709] border border-zinc-800 shadow-2xl backdrop-blur-xl max-w-full">
      {/* Microphone toggle */}
      <button
        onClick={onToggleMic}
        title={isMicMuted ? 'Unmute Microphone' : 'Mute Microphone'}
        className={`group relative flex h-10 w-10 sm:h-11 sm:w-11 items-center justify-center rounded-md transition-all duration-150 shrink-0 cursor-pointer ${
          isMicMuted
            ? 'bg-rose-500/15 text-rose-400 border border-rose-500/40 hover:bg-rose-500/25'
            : 'bg-zinc-900 text-emerald-400 border border-zinc-700/80 hover:border-emerald-500/60 hover:text-emerald-300'
        }`}
      >
        {isMicMuted ? <MicOff className="h-4 w-4 sm:h-5 sm:w-5" /> : <Mic className="h-4 w-4 sm:h-5 sm:w-5" />}
        <span className="sr-only">Toggle Microphone</span>
      </button>

      {/* Camera toggle */}
      <button
        onClick={onToggleCamera}
        title={isCameraOff ? 'Turn Camera On' : 'Turn Camera Off'}
        className={`group relative flex h-10 w-10 sm:h-11 sm:w-11 items-center justify-center rounded-md transition-all duration-150 shrink-0 cursor-pointer ${
          isCameraOff
            ? 'bg-rose-500/15 text-rose-400 border border-rose-500/40 hover:bg-rose-500/25'
            : 'bg-zinc-900 text-zinc-200 border border-zinc-700/80 hover:border-zinc-500 hover:text-white'
        }`}
      >
        {isCameraOff ? <VideoOff className="h-4 w-4 sm:h-5 sm:w-5" /> : <Video className="h-4 w-4 sm:h-5 sm:w-5" />}
        <span className="sr-only">Toggle Camera</span>
      </button>

      {/* Screen Share */}
      <button
        onClick={onToggleScreenShare}
        title={isScreenSharing ? 'Stop Screen Share' : 'Share Screen'}
        className={`group relative hidden xs:flex h-10 w-10 sm:h-11 sm:w-11 items-center justify-center rounded-md transition-all duration-150 shrink-0 cursor-pointer ${
          isScreenSharing
            ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/50 hover:bg-emerald-500/30'
            : 'bg-zinc-900 text-zinc-200 border border-zinc-700/80 hover:border-zinc-500 hover:text-white'
        }`}
      >
        <ScreenShare className="h-4 w-4 sm:h-5 sm:w-5" />
        <span className="sr-only">Share Screen</span>
      </button>

      {/* Security Monitor toggle button for mobile / smaller screens */}
      {onToggleSecurityPanel && (
        <button
          onClick={onToggleSecurityPanel}
          title="DeepTrace Monitoring"
          className={`flex lg:hidden items-center gap-1.5 px-3 h-10 sm:h-11 rounded-md font-mono text-[11px] sm:text-xs font-semibold tracking-wider uppercase transition-all duration-150 border shrink-0 cursor-pointer ${
            isSecurityPanelOpen
              ? 'bg-orange-500/20 text-orange-300 border-orange-500/50'
              : 'bg-zinc-900 text-zinc-300 border-zinc-700 hover:bg-zinc-800'
          }`}
        >
          <Activity className="h-3.5 w-3.5 text-orange-400" />
          <span className="hidden xs:inline">Monitor</span>
        </button>
      )}

      {/* TESTER ONLY: Attack Simulator Button */}
      {isTester && onToggleAttackDrawer && (
        <button
          onClick={onToggleAttackDrawer}
          title="Open Attack Simulator (Tester Security Testing)"
          className={`flex items-center gap-1.5 px-3 h-10 sm:h-11 rounded-md font-mono text-[11px] sm:text-xs font-semibold tracking-wider uppercase transition-all duration-150 shadow-md shrink-0 cursor-pointer ${
            isAttackActive
              ? 'bg-rose-600/30 text-rose-300 border border-rose-500/60 animate-pulse hover:bg-rose-600/40'
              : isAttackDrawerOpen
              ? 'bg-orange-500/25 text-orange-300 border border-orange-500/60'
              : 'bg-orange-500/15 text-orange-400 border border-orange-500/35 hover:bg-orange-500/25 hover:border-orange-500/60'
          }`}
        >
          <ShieldAlert className={`h-4 w-4 ${isAttackActive ? 'text-rose-400' : 'text-orange-400'}`} />
          <span className="hidden sm:inline">Attack</span>
          {isAttackActive && (
            <span className="flex h-1.5 w-1.5 rounded-full bg-rose-500 animate-ping" />
          )}
        </button>
      )}

      <div className="h-5 w-[1px] bg-zinc-800 mx-1 shrink-0" />

      {/* End Call */}
      <button
        onClick={onEndCall}
        title="End Call"
        className="flex h-10 sm:h-11 items-center justify-center gap-1.5 rounded-md bg-rose-600 px-3.5 sm:px-5 text-xs font-mono font-bold uppercase tracking-wider text-white shadow-lg shadow-rose-950/60 border border-rose-500 transition-all duration-150 hover:bg-rose-500 active:scale-95 shrink-0 cursor-pointer"
      >
        <PhoneOff className="h-4 w-4" />
        <span className="hidden xs:inline">End Call</span>
      </button>
    </div>
  );
};
