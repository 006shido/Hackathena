import React, { useState } from 'react';
import { Shield, Copy, Check, Users, Radio } from 'lucide-react';
import { UserRole } from '../types/auth';

interface RoomHeaderProps {
  roomId: string;
  username: string;
  role: UserRole;
  participantCount: number;
  connectionState?: string;
  iceState?: string;
}

export const RoomHeader: React.FC<RoomHeaderProps> = ({
  roomId,
  role,
  participantCount,
  connectionState,
  iceState,
}) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    if (!roomId) return;
    navigator.clipboard.writeText(roomId);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <header className="flex items-center justify-between gap-2 px-3 sm:px-6 py-2 sm:py-3.5 bg-slate-950/90 border-b border-slate-800/80 backdrop-blur-md shrink-0">
      {/* Brand logo */}
      <div className="flex items-center gap-2 sm:gap-3">
        <div className="flex h-8 w-8 sm:h-9 sm:w-9 items-center justify-center rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 shadow-md shadow-cyan-900/30 shrink-0">
          <Shield className="h-4 w-4 sm:h-5 sm:w-5 text-white" />
        </div>
        <div>
          <div className="flex items-center gap-1.5 sm:gap-2">
            <span className="text-sm sm:text-base font-bold tracking-tight text-white">DeepTrace</span>
            <span className="flex items-center gap-1 rounded bg-cyan-500/10 px-1.5 py-0.2 sm:px-2 sm:py-0.5 text-[9px] sm:text-[10px] font-mono text-cyan-400 border border-cyan-500/30 uppercase tracking-widest">
              <Radio className="h-2 w-2 sm:h-2.5 sm:w-2.5 animate-pulse" /> Live
            </span>
          </div>
          <span className="text-[10px] sm:text-[11px] text-slate-400 tracking-wide hidden xs:inline">Secure Video</span>
        </div>
      </div>

      {/* Room ID Badge, Count, WebRTC State & Role */}
      <div className="flex items-center gap-1.5 sm:gap-2.5">
        <div className="flex items-center gap-1.5 sm:gap-2 rounded-lg sm:rounded-xl bg-slate-900/90 px-2 py-1 sm:px-3 sm:py-1.5 border border-slate-800 shadow-sm">
          <span className="text-[11px] sm:text-xs text-slate-400 font-mono hidden sm:inline">Room:</span>
          <span className="text-[11px] sm:text-xs font-mono font-bold text-cyan-300 tracking-wider">{roomId}</span>
          <button
            onClick={handleCopy}
            title="Copy Room ID"
            className="flex items-center justify-center p-0.5 sm:p-1 rounded text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            {copied ? <Check className="h-3 w-3 sm:h-3.5 sm:w-3.5 text-emerald-400" /> : <Copy className="h-3 w-3 sm:h-3.5 sm:w-3.5" />}
          </button>
        </div>

        {/* Participant Count (Max 2) */}
        <div className="flex items-center gap-1 rounded-lg sm:rounded-xl bg-slate-900/90 px-2 py-1 sm:px-3 sm:py-1.5 border border-slate-800 text-[11px] sm:text-xs font-mono text-slate-300">
          <Users className="h-3 w-3 sm:h-3.5 sm:w-3.5 text-slate-400" />
          <span>{participantCount}/2</span>
          <span className={`h-1.5 w-1.5 sm:h-2 sm:w-2 rounded-full ${participantCount === 2 ? 'bg-emerald-400' : 'bg-amber-400 animate-pulse'}`} />
        </div>

        {/* WebRTC P2P ICE State Badge */}
        {participantCount === 2 && (
          <div className="hidden md:flex items-center gap-1.5 rounded-lg sm:rounded-xl bg-slate-900/90 px-2.5 py-1 sm:py-1.5 border border-slate-800 text-[10px] font-mono">
            <span className={`h-1.5 w-1.5 rounded-full ${
              iceState === 'connected' || iceState === 'completed'
                ? 'bg-emerald-400'
                : iceState === 'checking'
                ? 'bg-cyan-400 animate-ping'
                : iceState === 'failed'
                ? 'bg-rose-500'
                : 'bg-amber-400'
            }`} />
            <span className="text-slate-400">P2P:</span>
            <span className={`${
              iceState === 'connected' || iceState === 'completed'
                ? 'text-emerald-300 font-bold'
                : iceState === 'checking'
                ? 'text-cyan-300 font-bold'
                : iceState === 'failed'
                ? 'text-rose-400 font-bold'
                : 'text-amber-300'
            }`}>
              {iceState === 'connected' || iceState === 'completed' ? 'SECURE P2P' : iceState ? iceState.toUpperCase() : (connectionState || 'CONNECTING').toUpperCase()}
            </span>
          </div>
        )}

        {/* Role badge */}
        {role === 'tester' ? (
          <span className="flex items-center gap-1 rounded-lg sm:rounded-xl bg-amber-500/15 px-2 py-1 sm:px-3 sm:py-1.5 text-[10px] sm:text-xs font-mono font-semibold text-amber-300 border border-amber-500/40 uppercase tracking-wider">
            <Shield className="h-3 w-3 sm:h-3.5 sm:w-3.5" />
            <span className="hidden xs:inline">Tester</span>
          </span>
        ) : (
          <span className="rounded-lg sm:rounded-xl bg-slate-800/80 px-2 py-1 sm:px-3 sm:py-1.5 text-[10px] sm:text-xs font-mono text-slate-300 border border-slate-700 uppercase tracking-wider">
            <span className="hidden xs:inline">User</span>
            <span className="xs:hidden">U</span>
          </span>
        )}
      </div>
    </header>
  );
};
