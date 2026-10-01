import React, { useState, useEffect } from 'react';
import { Shield, Copy, Check, Users, Info, Circle } from 'lucide-react';
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
  const [timeStr, setTimeStr] = useState<string>('');

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setTimeStr(
        now.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
      );
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  const handleCopy = () => {
    if (!roomId) return;
    navigator.clipboard.writeText(roomId);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const isConnected = iceState === 'connected' || iceState === 'completed';

  return (
    <header className="flex items-center justify-between gap-3 px-4 sm:px-6 py-3 bg-[#202124] text-[#e8eaed] shrink-0 select-none">
      {/* Top Left: Logo & Time | Room ID (i) */}
      <div className="flex items-center gap-2.5">
        <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-white p-1 shadow-sm border border-white/20">
          <img src="/logo.svg" alt="DeepTrace Logo" className="h-full w-full object-contain" />
        </div>
        <span className="text-sm font-normal text-[#e8eaed] tracking-wide">
          {timeStr || '8:30 PM'}
        </span>
        <span className="text-[#5f6368] font-light">|</span>
        <div className="flex items-center gap-1.5 text-sm text-[#e8eaed] font-medium">
          <span>{roomId}</span>
          <button
            onClick={handleCopy}
            title={copied ? 'Copied!' : 'Copy meeting code'}
            className="flex items-center justify-center p-1 rounded-full text-[#9aa0a6] hover:text-[#e8eaed] hover:bg-[#3c4043] transition-colors cursor-pointer"
          >
            {copied ? (
              <Check className="h-3.5 w-3.5 text-[#34a853]" />
            ) : (
              <Info className="h-3.5 w-3.5" />
            )}
          </button>
        </div>
      </div>

      {/* Top Right: Status & Participants */}
      <div className="flex items-center gap-2">
        {participantCount === 2 && (
          <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-[#303134] text-xs text-[#9aa0a6]">
            <Circle
              className={`h-2 w-2 fill-current ${
                isConnected ? 'text-[#34a853]' : 'text-[#fbbc04] animate-pulse'
              }`}
            />
            <span className={isConnected ? 'text-[#34a853]' : 'text-[#9aa0a6]'}>
              {isConnected ? 'Connected' : 'Connecting...'}
            </span>
          </div>
        )}

        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-[#303134] text-xs text-[#9aa0a6]">
          <Users className="h-3.5 w-3.5 text-[#e8eaed]" />
          <span className="text-[#e8eaed] font-medium">{participantCount}</span>
        </div>

        {role === 'tester' && (
          <span className="flex items-center gap-1 px-2.5 py-1 rounded-full bg-[#3c4043] text-xs font-medium text-[#8ab4f8]">
            <Shield className="h-3 w-3" />
            <span className="hidden sm:inline">Tester</span>
          </span>
        )}
      </div>
    </header>
  );
};
