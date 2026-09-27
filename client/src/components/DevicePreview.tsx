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
    <div className="flex flex-col items-center justify-center min-h-[540px] w-full max-w-2xl mx-auto p-6 bg-slate-900/90 border border-slate-800 rounded-3xl shadow-2xl backdrop-blur-xl">
      {/* Header */}
      <div className="w-full flex items-center justify-between pb-4 border-b border-slate-800/80 mb-6">
        <div>
          <h2 className="text-xl font-semibold text-slate-100 flex items-center gap-2">
            <span>Audio & Video Check</span>
            {role === 'tester' && (
              <span className="flex items-center gap-1 rounded bg-amber-500/20 px-2 py-0.5 text-[10px] font-mono font-semibold text-amber-300 border border-amber-500/40 uppercase">
                <Shield className="h-3 w-3" /> Tester
              </span>
            )}
          </h2>
          <p className="text-xs text-slate-400 font-mono mt-0.5">
            Target Room: <span className="text-cyan-400 font-bold">{roomId}</span>
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs font-mono text-slate-400 bg-slate-800/60 px-3 py-1.5 rounded-lg border border-slate-700/50">
          <User className="h-3.5 w-3.5 text-cyan-400" />
          <span>{username}</span>
        </div>
      </div>

      {/* Permission / Device Warning Message */}
      {mediaError && (
        <div className="w-full mb-4 flex items-start gap-3 p-3.5 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs">
          <AlertCircle className="h-4 w-4 shrink-0 mt-0.5 text-amber-400" />
          <div className="flex-1">
            <span className="font-semibold block mb-0.5">Device Notice</span>
            <span>{mediaError}</span>
          </div>
        </div>
      )}

      {/* Camera Preview Tile */}
      <div className="relative w-full aspect-video rounded-2xl overflow-hidden bg-slate-950 border border-slate-800/90 shadow-inner flex items-center justify-center">
        {hasVideoTrack ? (
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="w-full h-full object-cover scale-x-[-1]"
          />
        ) : (
          <div className="flex flex-col items-center justify-center text-slate-500 gap-2">
            <div className="h-16 w-16 rounded-full bg-slate-900 border border-slate-800 flex items-center justify-center">
              <VideoOff className="h-8 w-8 text-slate-400" />
            </div>
            <span className="text-sm font-medium text-slate-400">Camera is off</span>
          </div>
        )}

        {/* Biometric overlay brackets */}
        <div className="pointer-events-none absolute top-3 left-3 h-3 w-3 border-t-2 border-l-2 border-cyan-500/60" />
        <div className="pointer-events-none absolute top-3 right-3 h-3 w-3 border-t-2 border-r-2 border-cyan-500/60" />
        <div className="pointer-events-none absolute bottom-3 left-3 h-3 w-3 border-b-2 border-l-2 border-cyan-500/60" />
        <div className="pointer-events-none absolute bottom-3 right-3 h-3 w-3 border-b-2 border-r-2 border-cyan-500/60" />

        {/* Status badges */}
        <div className="absolute bottom-3 left-3 flex items-center gap-2">
          <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-mono backdrop-blur-md border ${
            isMicMuted ? 'bg-rose-500/20 text-rose-300 border-rose-500/30' : 'bg-slate-900/80 text-emerald-300 border-slate-700/60'
          }`}>
            {isMicMuted ? <MicOff className="h-3 w-3" /> : <Mic className="h-3 w-3" />}
            <span>{isMicMuted ? 'Mic Muted' : 'Mic Active'}</span>
          </div>

          <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-mono backdrop-blur-md border ${
            isCameraOff ? 'bg-rose-500/20 text-rose-300 border-rose-500/30' : 'bg-slate-900/80 text-cyan-300 border-slate-700/60'
          }`}>
            {isCameraOff ? <VideoOff className="h-3 w-3" /> : <Video className="h-3 w-3" />}
            <span>{isCameraOff ? 'Camera Off' : 'Camera On'}</span>
          </div>
        </div>
      </div>

      {/* Control Buttons & Join Call */}
      <div className="w-full mt-4 sm:mt-6 flex flex-col sm:flex-row items-center justify-between gap-3 sm:gap-4">
        {/* Hardware toggles */}
        <div className="grid grid-cols-2 sm:flex items-center gap-2 sm:gap-3 w-full sm:w-auto">
          <button
            onClick={onToggleMic}
            className={`flex items-center justify-center gap-1.5 sm:gap-2 px-3 sm:px-4 py-2 sm:py-2.5 rounded-xl border text-xs font-medium font-mono transition-all ${
              isMicMuted
                ? 'bg-rose-500/20 text-rose-300 border-rose-500/40 hover:bg-rose-500/30'
                : 'bg-slate-800/80 text-slate-200 border-slate-700/70 hover:bg-slate-700/80'
            }`}
          >
            {isMicMuted ? <MicOff className="h-4 w-4 shrink-0" /> : <Mic className="h-4 w-4 shrink-0" />}
            <span>{isMicMuted ? 'Unmute' : 'Mute'}</span>
          </button>

          <button
            onClick={onToggleCamera}
            className={`flex items-center justify-center gap-1.5 sm:gap-2 px-3 sm:px-4 py-2 sm:py-2.5 rounded-xl border text-xs font-medium font-mono transition-all ${
              isCameraOff
                ? 'bg-rose-500/20 text-rose-300 border-rose-500/40 hover:bg-rose-500/30'
                : 'bg-slate-800/80 text-slate-200 border-slate-700/70 hover:bg-slate-700/80'
            }`}
          >
            {isCameraOff ? <VideoOff className="h-4 w-4 shrink-0" /> : <Video className="h-4 w-4 shrink-0" />}
            <span>{isCameraOff ? 'Camera On' : 'Camera Off'}</span>
          </button>
        </div>

        {/* Action buttons */}
        <div className="flex items-center gap-2 sm:gap-3 w-full sm:w-auto">
          <button
            onClick={onCancel}
            className="flex-1 sm:flex-none px-4 py-2 sm:py-2.5 rounded-xl bg-slate-800/80 border border-slate-700/70 text-slate-300 text-xs font-medium hover:bg-slate-700/80 transition-colors text-center"
          >
            Cancel
          </button>

          <button
            onClick={onJoin}
            className="flex-1 sm:flex-none flex items-center justify-center gap-1.5 sm:gap-2 px-5 sm:px-6 py-2 sm:py-2.5 rounded-xl bg-gradient-to-r from-cyan-600 to-blue-600 text-white text-xs font-bold font-mono tracking-wider uppercase shadow-lg shadow-cyan-900/40 hover:from-cyan-500 hover:to-blue-500 transition-all active:scale-95 text-center"
          >
            <span>Join Call</span>
            <ArrowRight className="h-4 w-4 shrink-0" />
          </button>
        </div>
      </div>
    </div>
  );
};
