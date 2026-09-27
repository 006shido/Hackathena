import React, { useEffect, useRef } from 'react';
import { MicOff, VideoOff, Shield, User as UserIcon } from 'lucide-react';
import { UserRole } from '../types/auth';

interface VideoTileProps {
  stream: MediaStream | null;
  username: string;
  role?: UserRole;
  isLocal?: boolean;
  isMuted?: boolean;
  isVideoOff?: boolean;
  subtitle?: string;
  isFloating?: boolean;
  className?: string;
}

export const VideoTile: React.FC<VideoTileProps> = ({
  stream,
  username,
  role = 'user',
  isLocal = false,
  isMuted = false,
  isVideoOff = false,
  subtitle,
  isFloating = false,
  className = '',
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.srcObject = stream;
    }
  }, [stream]);

  const hasVideoTrack = stream && stream.getVideoTracks().length > 0 && stream.getVideoTracks()[0].enabled && !isVideoOff;

  return (
    <div
      className={`relative overflow-hidden rounded-2xl border border-slate-800/80 bg-slate-950/90 shadow-2xl backdrop-blur-md transition-all duration-300 ${
        isFloating
          ? 'absolute bottom-3 right-3 sm:bottom-6 sm:right-6 z-20 w-28 sm:w-44 md:w-56 aspect-[3/4] sm:aspect-video ring-1 sm:ring-2 ring-cyan-500/40 shadow-cyan-950/40 hover:scale-105'
          : 'w-full h-full min-h-0'
      } ${className}`}
    >
      {/* Video Element */}
      {stream && (
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted={isLocal} // Always mute local video to avoid echo
          className={`h-full w-full object-cover transition-opacity duration-300 ${
            hasVideoTrack ? 'opacity-100' : 'opacity-0'
          } ${isLocal ? 'scale-x-[-1]' : ''}`}
        />
      )}

      {/* Video Disabled / Offline State */}
      {!hasVideoTrack && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-gradient-to-b from-slate-900 via-slate-950 to-slate-900 text-slate-400">
          <div className="relative mb-3 flex h-20 w-20 items-center justify-center rounded-full bg-slate-800/80 border border-slate-700/60 shadow-inner">
            {role === 'tester' ? (
              <Shield className="h-10 w-10 text-amber-400/90" />
            ) : (
              <UserIcon className="h-10 w-10 text-cyan-400/90" />
            )}
            <div className="absolute -bottom-1 -right-1 rounded-full bg-slate-900 p-1.5 border border-slate-700">
              <VideoOff className="h-4 w-4 text-rose-400" />
            </div>
          </div>
          <span className="text-sm font-medium text-slate-300">Camera is disabled</span>
          <span className="text-xs text-slate-500 font-mono mt-0.5">{username}</span>
        </div>
      )}

      {/* Biometric / Cyber Framing Lines */}
      <div className="pointer-events-none absolute inset-0 rounded-2xl border border-white/5" />
      <div className="pointer-events-none absolute top-2 left-2 h-2.5 w-2.5 border-t-2 border-l-2 border-cyan-500/50" />
      <div className="pointer-events-none absolute top-2 right-2 h-2.5 w-2.5 border-t-2 border-r-2 border-cyan-500/50" />
      <div className="pointer-events-none absolute bottom-2 left-2 h-2.5 w-2.5 border-b-2 border-l-2 border-cyan-500/50" />
      <div className="pointer-events-none absolute bottom-2 right-2 h-2.5 w-2.5 border-b-2 border-r-2 border-cyan-500/50" />

      {/* Top Left: Role & Identity Tag */}
      <div className="absolute top-2 left-2 sm:top-3 sm:left-3 z-10 flex items-center gap-1.5 sm:gap-2">
        <div className={`flex items-center gap-1 sm:gap-1.5 rounded-md bg-slate-900/85 px-1.5 py-0.5 sm:px-2.5 sm:py-1 font-mono font-medium backdrop-blur-md border border-slate-700/50 shadow-sm ${
          isFloating ? 'text-[10px] sm:text-xs' : 'text-xs'
        }`}>
          <span className={`h-1.5 w-1.5 sm:h-2 sm:w-2 rounded-full ${isLocal ? 'bg-cyan-400 animate-pulse' : 'bg-emerald-400'}`} />
          <span className="text-slate-200 truncate max-w-[70px] sm:max-w-none">{username}</span>
          {isLocal && <span className="text-[9px] sm:text-[10px] text-cyan-400 font-sans hidden xs:inline">(You)</span>}
        </div>

        {/* Role Badge - hidden on small floating tiles */}
        {!isFloating && (
          role === 'tester' ? (
            <span className="flex items-center gap-1 rounded bg-amber-500/20 px-2 py-0.5 text-[10px] font-mono font-semibold text-amber-300 border border-amber-500/40 uppercase tracking-wider">
              <Shield className="h-2.5 w-2.5" />
              Tester
            </span>
          ) : (
            <span className="rounded bg-slate-800/80 px-2 py-0.5 text-[10px] font-mono text-slate-300 border border-slate-700 uppercase tracking-wider">
              User
            </span>
          )
        )}
      </div>

      {/* Bottom Status Overlay */}
      <div className="absolute bottom-3 left-3 right-3 z-10 flex items-center justify-between pointer-events-none">
        {subtitle ? (
          <span className="rounded bg-slate-900/80 px-2 py-0.5 text-[11px] font-mono text-slate-400 border border-slate-800">
            {subtitle}
          </span>
        ) : <span />}

        {/* Audio status badge */}
        {isMuted && (
          <div className="flex items-center gap-1 rounded-md bg-rose-500/20 px-2 py-1 text-xs text-rose-300 border border-rose-500/30 backdrop-blur-sm">
            <MicOff className="h-3 w-3" />
            <span className="text-[10px] font-mono uppercase tracking-wide">Muted</span>
          </div>
        )}
      </div>
    </div>
  );
};
