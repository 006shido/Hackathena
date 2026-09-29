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
    <header className="flex items-center justify-between gap-2 px-3 sm:px-6 py-2.5 bg-[#050507] border-b border-zinc-800/90 shrink-0">
      {/* Brand logo */}
      <div className="flex items-center gap-2.5">
        <div className="flex h-8 w-8 items-center justify-center rounded-md bg-orange-500/10 border border-orange-500/30 text-orange-400 shrink-0">
          <Shield className="h-4 w-4" />
        </div>
        <div>
          <div className="flex items-center gap-1.5 sm:gap-2">
            <span className="text-sm font-bold tracking-tight text-white">DeepTrace</span>
            <span className="flex items-center gap-1 rounded bg-orange-500/10 px-1.5 py-0.5 text-[9px] font-mono text-orange-400 border border-orange-500/30 uppercase tracking-widest">
              <Radio className="h-2 w-2 animate-pulse text-orange-400" /> Live
            </span>
          </div>
          <span className="text-[10px] text-zinc-500 font-mono tracking-wide hidden xs:inline">1-to-1 WebRTC Secure Channel</span>
        </div>
      </div>

      {/* Room ID Badge, Count, WebRTC State & Role */}
      <div className="flex items-center gap-1.5 sm:gap-2">
        <div className="flex items-center gap-1.5 rounded-md bg-zinc-950 px-2.5 py-1 border border-zinc-800 shadow-sm">
          <span className="text-[11px] text-zinc-500 font-mono hidden sm:inline">Room:</span>
          <span className="text-xs font-mono font-bold text-orange-400 tracking-wider">{roomId}</span>
          <button
            onClick={handleCopy}
            title="Copy Room ID"
            className="flex items-center justify-center p-0.5 rounded text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors cursor-pointer"
          >
            {copied ? <Check className="h-3 w-3 text-emerald-400" /> : <Copy className="h-3 w-3" />}
          </button>
        </div>

        {/* Participant Count (Max 2) */}
        <div className="flex items-center gap-1 rounded-md bg-zinc-950 px-2.5 py-1 border border-zinc-800 text-xs font-mono text-zinc-300">
          <Users className="h-3.5 w-3.5 text-zinc-400" />
          <span>{participantCount}/2</span>
          <span className={`h-1.5 w-1.5 rounded-full ${participantCount === 2 ? 'bg-emerald-400' : 'bg-orange-400 animate-pulse'}`} />
        </div>

        {/* WebRTC P2P ICE State Badge */}
        {participantCount === 2 && (
          <div className="hidden md:flex items-center gap-1.5 rounded-md bg-zinc-950 px-2.5 py-1 border border-zinc-800 text-[10px] font-mono">
            <span className={`h-1.5 w-1.5 rounded-full ${
              iceState === 'connected' || iceState === 'completed'
                ? 'bg-emerald-400'
                : iceState === 'checking'
                ? 'bg-orange-400 animate-ping'
                : iceState === 'failed'
                ? 'bg-rose-500'
                : 'bg-zinc-500'
            }`} />
            <span className="text-zinc-500">P2P:</span>
            <span className={`${
              iceState === 'connected' || iceState === 'completed'
                ? 'text-emerald-400 font-bold'
                : iceState === 'checking'
                ? 'text-orange-400 font-bold'
                : iceState === 'failed'
                ? 'text-rose-400 font-bold'
                : 'text-zinc-400'
            }`}>
              {iceState === 'connected' || iceState === 'completed' ? 'SECURE P2P' : iceState ? iceState.toUpperCase() : (connectionState || 'CONNECTING').toUpperCase()}
            </span>
          </div>
        )}

        {/* Role badge */}
        {role === 'tester' ? (
          <span className="flex items-center gap-1 rounded-md bg-orange-500/15 px-2.5 py-1 text-xs font-mono font-semibold text-orange-400 border border-orange-500/40 uppercase tracking-wider">
            <Shield className="h-3 w-3" />
            <span className="hidden xs:inline">Tester</span>
          </span>
        ) : (
          <span className="rounded-md bg-zinc-900 px-2.5 py-1 text-xs font-mono text-zinc-300 border border-zinc-800 uppercase tracking-wider">
            <span className="hidden xs:inline">User</span>
            <span className="xs:hidden">U</span>
          </span>
        )}
      </div>
    </header>
  );
};
