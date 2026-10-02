import React, { useState, useEffect } from 'react';
import { ArrowLeft, Users, Copy, Check, Shield, Circle, Activity } from 'lucide-react';
import { UserRole } from '../types/auth';

interface RoomHeaderProps {
  roomId: string;
  username: string;
  role: UserRole;
  participantCount: number;
  connectionState?: string;
  iceState?: string;
  onExit?: () => void;
  onToggleSecurityPanel?: () => void;
  isSecurityPanelOpen?: boolean;
}

export const RoomHeader: React.FC<RoomHeaderProps> = ({
  roomId,
  role,
  participantCount,
  connectionState,
  iceState,
  onExit,
  onToggleSecurityPanel,
  isSecurityPanelOpen,
}) => {
  const [copied, setCopied] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  // Live elapsed call timer (like the 12:34 in Screenshot 1 right)
  useEffect(() => {
    const timer = setInterval(() => {
      setElapsedSeconds((prev) => prev + 1);
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const formatTimer = (totalSec: number) => {
    const mins = Math.floor(totalSec / 60);
    const secs = totalSec % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handleCopy = () => {
    if (!roomId) return;
    navigator.clipboard.writeText(roomId);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const isConnected = iceState === 'connected' || iceState === 'completed';

  return (
    <header className="flex items-center justify-between gap-3 px-3 sm:px-6 py-3 bg-[#0f1115] text-slate-100 shrink-0 select-none border-b border-white/5">
      {/* Left: Back Arrow + Logo + Meeting Title + Elapsed Timer */}
      <div className="flex items-center gap-2.5 sm:gap-3 min-w-0">
        {onExit && (
          <button
            onClick={onExit}
            title="Leave room"
            className="p-2 rounded-full text-slate-400 hover:text-white hover:bg-white/10 transition-colors cursor-pointer shrink-0"
          >
            <ArrowLeft className="h-4 w-4" />
          </button>
        )}

        <div className="h-8 w-8 sm:h-9 sm:w-9 shrink-0 shadow-xs hidden xs:block">
          <img src="/logo.svg" alt="DeepTrace Logo" className="h-full w-full" />
        </div>

        <div className="flex flex-col min-w-0">
          <div className="flex items-center gap-2">
            <h1 className="text-sm font-semibold text-white tracking-tight truncate max-w-[130px] sm:max-w-[220px]">
              Room {roomId}
            </h1>
            <button
              onClick={handleCopy}
              title={copied ? 'Copied!' : 'Copy room code'}
              className="p-1 rounded-md text-slate-400 hover:text-white hover:bg-white/10 transition-colors cursor-pointer shrink-0"
            >
              {copied ? (
                <Check className="h-3.5 w-3.5 text-emerald-400" />
              ) : (
                <Copy className="h-3 w-3" />
              )}
            </button>
          </div>
          <span className="text-[11px] font-mono text-slate-400 tracking-wider">
            {formatTimer(elapsedSeconds)}
          </span>
        </div>
      </div>

      {/* Right: Participant Count + Status + Role */}
      <div className="flex items-center gap-2 sm:gap-2.5 shrink-0">
        {/* Connection status dot */}
        {participantCount === 2 && (
          <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-white/5 border border-white/10 text-xs">
            <Circle
              className={`h-2 w-2 fill-current ${
                isConnected ? 'text-emerald-400' : 'text-amber-400 animate-pulse'
              }`}
            />
            <span className={isConnected ? 'text-emerald-400' : 'text-slate-400'}>
              {isConnected ? 'P2P Encrypted' : 'Connecting...'}
            </span>
          </div>
        )}

        {/* Participant Count (Matching icon in Screenshot 1 Right) */}
        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-white/5 border border-white/10 text-xs text-slate-300">
          <Users className="h-3.5 w-3.5 text-slate-300" />
          <span className="font-semibold text-white">{participantCount}</span>
        </div>

        {/* Tester Badge */}
        {role === 'tester' && (
          <span className="hidden sm:flex items-center gap-1 px-2.5 py-1 rounded-full bg-blue-500/10 border border-blue-500/30 text-xs font-semibold text-blue-400">
            <Shield className="h-3 w-3" />
            <span>Tester</span>
          </span>
        )}

        {/* Security Monitor Quick Toggle */}
        {onToggleSecurityPanel && (
          <button
            onClick={onToggleSecurityPanel}
            title="Toggle Security Monitor"
            className={`p-2 rounded-full transition-colors cursor-pointer ${
              isSecurityPanelOpen
                ? 'bg-blue-600 text-white shadow-md shadow-blue-500/20'
                : 'text-slate-400 hover:text-white hover:bg-white/10'
            }`}
          >
            <Activity className="h-4 w-4" />
          </button>
        )}
      </div>
    </header>
  );
};
