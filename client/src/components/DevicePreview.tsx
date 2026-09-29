import React, { useEffect, useRef } from 'react';
import { Mic, MicOff, Video, VideoOff, AlertCircle, Shield, ArrowRight, User } from 'lucide-react';
import { UserRole } from '../types/auth';

interface DevicePreviewProps {
  stream: MediaStream | null;
  isMicMuted: boolean;
  isCameraOff: boolean;
  mediaError: string | null;
  username: string;
  role: UserRole;
  roomId: string;
  onToggleMic: () => void;
  onToggleCamera: () => void;
  onJoin: () => void;
  onCancel: () => void;
}

export const DevicePreview: React.FC<DevicePreviewProps> = ({
  stream,
  isMicMuted,
  isCameraOff,
  mediaError,
  username,
  role,
  roomId,
  onToggleMic,
  onToggleCamera,
  onJoin,
  onCancel,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (videoRef.current && stream) {
      videoRef.current.srcObject = stream;
    }
  }, [stream]);

  const hasVideoTrack = stream && stream.getVideoTracks().length > 0 && !isCameraOff;

  return (
    <div className="flex flex-col items-center justify-center w-full max-w-4xl mx-auto p-5 sm:p-7 bg-[#070709] border border-zinc-800 rounded-lg shadow-2xl relative overflow-hidden">
      {/* Subtle top edge highlight */}
      <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-orange-500/40 to-transparent" />

      {/* Header */}
      <div className="w-full flex items-center justify-between pb-4 border-b border-zinc-800/80 mb-5">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <span>Audio & Video Check</span>
            {role === 'tester' && (
              <span className="flex items-center gap-1 rounded bg-orange-500/15 px-2 py-0.5 text-[10px] font-mono font-semibold text-orange-400 border border-orange-500/40 uppercase">
                <Shield className="h-3 w-3" /> Tester
              </span>
            )}
          </h2>
          <p className="text-xs text-zinc-400 font-mono mt-0.5">
            Target Room: <span className="text-orange-400 font-bold">{roomId}</span>
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs font-mono text-zinc-300 bg-zinc-900/90 px-3 py-1.5 rounded-md border border-zinc-800">
          <User className="h-3.5 w-3.5 text-orange-400" />
          <span>{username}</span>
        </div>
      </div>

      {/* Permission / Device Warning Message */}
      {mediaError && (
        <div className="w-full mb-4 flex items-start gap-3 p-3.5 rounded-md bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs font-mono">
          <AlertCircle className="h-4 w-4 shrink-0 mt-0.5 text-amber-400" />
          <div className="flex-1">
            <span className="font-semibold block mb-0.5">Device Notice</span>
            <span>{mediaError}</span>
          </div>
        </div>
      )}

      {/* Camera Preview Tile - Expansive 16:9 view */}
      <div className="relative w-full aspect-video max-h-[460px] rounded-md overflow-hidden bg-black border border-zinc-800 shadow-inner flex items-center justify-center">
        {hasVideoTrack ? (
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="w-full h-full object-cover scale-x-[-1]"
          />
        ) : (
          <div className="flex flex-col items-center justify-center text-zinc-500 gap-2">
            <div className="h-16 w-16 rounded-md bg-zinc-900 border border-zinc-800 flex items-center justify-center">
              <VideoOff className="h-8 w-8 text-zinc-400" />
            </div>
            <span className="text-xs sm:text-sm font-medium text-zinc-300 font-mono">Camera is currently disabled</span>
            <span className="text-[11px] text-zinc-600 font-mono">Click 'Enable Camera' below to turn on video</span>
          </div>
        )}

        {/* Framing edge markers */}
        <div className="pointer-events-none absolute top-3 left-3 h-3 w-3 border-t border-l border-zinc-600" />
        <div className="pointer-events-none absolute top-3 right-3 h-3 w-3 border-t border-r border-zinc-600" />
        <div className="pointer-events-none absolute bottom-3 left-3 h-3 w-3 border-b border-l border-zinc-600" />
        <div className="pointer-events-none absolute bottom-3 right-3 h-3 w-3 border-b border-r border-zinc-600" />

        {/* Status badges - clean & distinct */}
        <div className="absolute bottom-3 left-3 flex items-center gap-2">
          <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-mono backdrop-blur-md border ${
            isMicMuted ? 'bg-rose-500/15 text-rose-300 border-rose-500/40' : 'bg-black/80 text-emerald-400 border-emerald-800/60'
          }`}>
            {isMicMuted ? <MicOff className="h-3 w-3" /> : <Mic className="h-3 w-3 text-emerald-400" />}
            <span>{isMicMuted ? 'Mic Muted' : 'Mic Active'}</span>
          </div>

          <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-mono backdrop-blur-md border ${
            isCameraOff ? 'bg-rose-500/15 text-rose-300 border-rose-500/40' : 'bg-black/80 text-zinc-200 border-zinc-700'
          }`}>
            {isCameraOff ? <VideoOff className="h-3 w-3" /> : <Video className="h-3 w-3 text-emerald-400" />}
            <span>{isCameraOff ? 'Camera Off' : 'Camera On'}</span>
          </div>
        </div>
      </div>

      {/* Control Buttons & Join Call */}
      <div className="w-full mt-5 flex flex-col sm:flex-row items-center justify-between gap-3 sm:gap-4">
        {/* Hardware toggles */}
        <div className="grid grid-cols-2 sm:flex items-center gap-2 sm:gap-3 w-full sm:w-auto">
          <button
            onClick={onToggleMic}
            className={`flex items-center justify-center gap-2 px-4 py-2.5 rounded-md border text-xs font-medium font-mono transition-all cursor-pointer ${
              isMicMuted
                ? 'bg-rose-500/10 text-rose-300 border-rose-500/40 hover:bg-rose-500/20'
                : 'bg-zinc-900 text-zinc-200 border-zinc-700 hover:border-emerald-500/60 hover:text-white'
            }`}
          >
            {isMicMuted ? <MicOff className="h-4 w-4 shrink-0 text-rose-400" /> : <Mic className="h-4 w-4 shrink-0 text-emerald-400" />}
            <span>{isMicMuted ? 'Unmute Mic' : 'Mute Mic'}</span>
          </button>

          <button
            onClick={onToggleCamera}
            className={`flex items-center justify-center gap-2 px-4 py-2.5 rounded-md border text-xs font-medium font-mono transition-all cursor-pointer ${
              isCameraOff
                ? 'bg-rose-500/10 text-rose-300 border-rose-500/40 hover:bg-rose-500/20'
                : 'bg-zinc-900 text-zinc-200 border-zinc-700 hover:border-emerald-500/60 hover:text-white'
            }`}
          >
            {isCameraOff ? <VideoOff className="h-4 w-4 shrink-0 text-rose-400" /> : <Video className="h-4 w-4 shrink-0 text-emerald-400" />}
            <span>{isCameraOff ? 'Turn Camera On' : 'Turn Camera Off'}</span>
          </button>
        </div>

        {/* Action buttons */}
        <div className="flex items-center gap-2.5 w-full sm:w-auto">
          <button
            onClick={onCancel}
            className="flex-1 sm:flex-none px-4 py-2.5 rounded-md bg-zinc-900 border border-zinc-800 text-zinc-300 text-xs font-mono font-medium hover:bg-zinc-800 hover:text-white transition-colors text-center cursor-pointer"
          >
            Cancel
          </button>

          <button
            onClick={onJoin}
            className="flex-1 sm:flex-none flex items-center justify-center gap-2 px-6 py-2.5 rounded-md bg-orange-600 hover:bg-orange-500 text-white text-xs font-bold font-mono tracking-wider uppercase shadow-lg shadow-orange-950/50 border border-orange-500 transition-all active:scale-95 text-center cursor-pointer"
          >
            <span>Enter Call</span>
            <ArrowRight className="h-4 w-4 shrink-0" />
          </button>
        </div>
      </div>
    </div>
  );
};
