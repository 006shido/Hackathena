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
    <div className="flex items-center justify-center gap-1.5 sm:gap-3 py-2 sm:py-3 px-2.5 sm:px-5 rounded-2xl bg-slate-900/95 border border-slate-800/90 shadow-2xl backdrop-blur-xl max-w-full">
      {/* Microphone toggle */}
      <button
        onClick={onToggleMic}
        title={isMicMuted ? 'Unmute Microphone' : 'Mute Microphone'}
        className={`group relative flex h-10 w-10 sm:h-12 sm:w-12 items-center justify-center rounded-xl transition-all duration-200 shrink-0 ${
          isMicMuted
            ? 'bg-rose-500/20 text-rose-400 border border-rose-500/40 hover:bg-rose-500/30'
            : 'bg-slate-800/90 text-slate-200 border border-slate-700/60 hover:bg-slate-700/80 hover:text-white'
        }`}
      >
        {isMicMuted ? <MicOff className="h-4 w-4 sm:h-5 sm:w-5" /> : <Mic className="h-4 w-4 sm:h-5 sm:w-5" />}
        <span className="sr-only">Toggle Microphone</span>
      </button>

      {/* Camera toggle */}
      <button
        onClick={onToggleCamera}
        title={isCameraOff ? 'Turn Camera On' : 'Turn Camera Off'}
        className={`group relative flex h-10 w-10 sm:h-12 sm:w-12 items-center justify-center rounded-xl transition-all duration-200 shrink-0 ${
          isCameraOff
            ? 'bg-rose-500/20 text-rose-400 border border-rose-500/40 hover:bg-rose-500/30'
            : 'bg-slate-800/90 text-slate-200 border border-slate-700/60 hover:bg-slate-700/80 hover:text-white'
        }`}
      >
        {isCameraOff ? <VideoOff className="h-4 w-4 sm:h-5 sm:w-5" /> : <Video className="h-4 w-4 sm:h-5 sm:w-5" />}
        <span className="sr-only">Toggle Camera</span>
      </button>

      {/* Screen Share (hidden on small mobile screens where screen-sharing isn't supported or practical) */}
      <button
        onClick={onToggleScreenShare}
        title={isScreenSharing ? 'Stop Screen Share' : 'Share Screen'}
        className={`group relative hidden xs:flex h-10 w-10 sm:h-12 sm:w-12 items-center justify-center rounded-xl transition-all duration-200 shrink-0 ${
          isScreenSharing
            ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/50 hover:bg-cyan-500/30'
            : 'bg-slate-800/90 text-slate-200 border border-slate-700/60 hover:bg-slate-700/80 hover:text-white'
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
          className={`flex lg:hidden items-center gap-1 sm:gap-1.5 px-2.5 sm:px-3 h-10 sm:h-12 rounded-xl font-mono text-[11px] sm:text-xs font-semibold tracking-wider uppercase transition-all duration-200 border shrink-0 ${
            isSecurityPanelOpen
              ? 'bg-cyan-500/25 text-cyan-300 border-cyan-500/50'
              : 'bg-slate-800/90 text-slate-300 border-slate-700/70 hover:bg-slate-700'
          }`}
        >
          <Activity className="h-3.5 w-3.5 sm:h-4 sm:w-4 text-cyan-400" />
          <span className="hidden xs:inline">Monitor</span>
        </button>
      )}

      {/* TESTER ONLY: Attack Simulator Button */}
      {isTester && onToggleAttackDrawer && (
        <button
          onClick={onToggleAttackDrawer}
          title="Open Attack Simulator (Tester Security Testing)"
          className={`flex items-center gap-1 sm:gap-1.5 px-2.5 sm:px-3 h-10 sm:h-12 rounded-xl font-mono text-[11px] sm:text-xs font-semibold tracking-wider uppercase transition-all duration-200 shadow-md shrink-0 ${
            isAttackActive
              ? 'bg-rose-600/30 text-rose-300 border border-rose-500/60 animate-pulse hover:bg-rose-600/40'
              : isAttackDrawerOpen
              ? 'bg-amber-500/25 text-amber-300 border border-amber-500/50'
              : 'bg-amber-500/10 text-amber-400 border border-amber-500/30 hover:bg-amber-500/20 hover:border-amber-500/50'
          }`}
        >
          <ShieldAlert className={`h-3.5 w-3.5 sm:h-4 sm:w-4 ${isAttackActive ? 'text-rose-400' : 'text-amber-400'}`} />
          <span className="hidden sm:inline">Attack</span>
          {isAttackActive && (
            <span className="flex h-1.5 w-1.5 sm:h-2 sm:w-2 rounded-full bg-rose-500 animate-ping" />
          )}
        </button>
      )}

      <div className="h-5 sm:h-6 w-[1px] bg-slate-800 mx-0.5 sm:mx-1 shrink-0" />

      {/* End Call */}
      <button
        onClick={onEndCall}
        title="End Call"
        className="flex h-10 sm:h-12 items-center justify-center gap-1.5 rounded-xl bg-rose-600 px-3 sm:px-5 text-xs font-semibold uppercase tracking-wider text-white shadow-lg shadow-rose-950/50 transition-all duration-200 hover:bg-rose-500 hover:shadow-rose-900/60 active:scale-95 shrink-0"
      >
        <PhoneOff className="h-4 w-4 sm:h-5 sm:w-5" />
        <span className="hidden xs:inline">End</span>
      </button>
    </div>
  );
};
