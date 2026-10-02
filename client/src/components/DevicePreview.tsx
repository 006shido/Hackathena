import React, { useEffect, useRef } from 'react';
import { Mic, MicOff, Video, VideoOff, AlertCircle, ArrowRight, User, Shield } from 'lucide-react';
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
    const video = videoRef.current;
    if (!video) return;

    if (stream) {
      video.muted = true;
      video.defaultMuted = true;
      if (video.srcObject !== stream) {
        video.srcObject = stream;
      }
      video.play().catch((err) => {
        console.warn('[DevicePreview] Video play error:', err);
      });
    } else {
      video.srcObject = null;
    }
  }, [stream, isCameraOff]);

  const hasVideoTrack = Boolean(stream && stream.getVideoTracks().length > 0 && !isCameraOff);
  const initials = username.split(' ').map((n) => n[0]).join('').toUpperCase().slice(0, 2) || 'U';

  return (
    <div className="flex flex-col items-center justify-center w-full max-w-xl mx-auto p-5 sm:p-8 bg-[#16181f] border border-white/10 rounded-3xl shadow-2xl text-slate-100">
      {/* Header */}
      <div className="w-full flex items-center justify-between pb-4 border-b border-white/10 mb-5">
        <div>
          <h2 className="text-lg font-bold text-white tracking-tight">Ready to join?</h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Room: <span className="text-blue-400 font-mono font-medium">{roomId}</span>
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs text-slate-300 bg-white/5 border border-white/10 px-3 py-1.5 rounded-full">
          <User className="h-3.5 w-3.5 text-slate-400" />
          <span className="font-medium">{username}</span>
          {role === 'tester' && (
            <span className="text-[10px] text-blue-400 bg-blue-500/20 px-2 py-0.5 rounded-full font-semibold">Tester</span>
          )}
        </div>
      </div>

      {/* Permission Warning */}
      {mediaError && (
        <div className="w-full mb-4 flex items-center gap-2 p-3 rounded-2xl bg-red-500/10 border border-red-500/30 text-red-400 text-xs sm:text-sm">
          <AlertCircle className="h-4 w-4 shrink-0 text-red-400" />
          <span>{mediaError}</span>
        </div>
      )}

      {/* Camera Preview Tile */}
      <div className="relative w-full aspect-video max-h-[360px] rounded-2xl sm:rounded-3xl overflow-hidden bg-[#0f1115] border border-white/10 flex items-center justify-center shadow-inner">
        <video
          ref={(el) => {
            videoRef.current = el;
            if (el && stream) {
              el.muted = true;
              el.defaultMuted = true;
              if (el.srcObject !== stream) {
                el.srcObject = stream;
              }
              el.play().catch(() => {});
            }
          }}
          autoPlay
          playsInline
          muted
          onLoadedMetadata={(e) => {
            const el = e.currentTarget;
            el.muted = true;
            el.play().catch(() => {});
          }}
          className={`w-full h-full object-cover scale-x-[-1] transition-opacity duration-200 ${
            hasVideoTrack ? 'opacity-100' : 'opacity-0 absolute pointer-events-none'
          }`}
        />

        {!hasVideoTrack && (
          <div className="flex flex-col items-center justify-center gap-3">
            <div className="h-16 w-16 sm:h-20 sm:w-20 rounded-full bg-blue-600 shadow-lg shadow-blue-600/30 flex items-center justify-center text-white text-xl sm:text-2xl font-bold">
              {initials}
            </div>
            <span className="text-xs text-slate-400 font-medium">Camera is off</span>
          </div>
        )}

        {/* Status badges overlay */}
        <div className="absolute bottom-3 left-3 flex items-center gap-2">
          <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] backdrop-blur-md ${
            isMicMuted ? 'bg-red-500 text-white' : 'bg-black/60 text-white'
          }`}>
            {isMicMuted ? <MicOff className="h-3 w-3" /> : <Mic className="h-3 w-3" />}
            <span>{isMicMuted ? 'Muted' : 'Mic on'}</span>
          </div>

          <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] backdrop-blur-md ${
            isCameraOff ? 'bg-red-500 text-white' : 'bg-black/60 text-white'
          }`}>
            {isCameraOff ? <VideoOff className="h-3 w-3" /> : <Video className="h-3 w-3" />}
            <span>{isCameraOff ? 'Camera off' : 'Camera on'}</span>
          </div>
        </div>
      </div>

      {/* Controls & Join Action */}
      <div className="w-full mt-6 flex flex-col sm:flex-row items-center justify-between gap-3">
        {/* Toggle Controls */}
        <div className="flex items-center gap-2 w-full sm:w-auto">
          <button
            onClick={onToggleMic}
            className={`flex-1 sm:flex-none flex items-center justify-center gap-2 px-4 py-2.5 rounded-full text-xs font-semibold transition-all cursor-pointer ${
              isMicMuted
                ? 'bg-red-500 hover:bg-red-600 text-white'
                : 'bg-white/10 hover:bg-white/15 text-slate-200'
            }`}
          >
            {isMicMuted ? <MicOff className="h-3.5 w-3.5" /> : <Mic className="h-3.5 w-3.5" />}
            <span>{isMicMuted ? 'Unmute' : 'Mute'}</span>
          </button>

          <button
            onClick={onToggleCamera}
            className={`flex-1 sm:flex-none flex items-center justify-center gap-2 px-4 py-2.5 rounded-full text-xs font-semibold transition-all cursor-pointer ${
              isCameraOff
                ? 'bg-red-500 hover:bg-red-600 text-white'
                : 'bg-white/10 hover:bg-white/15 text-slate-200'
            }`}
          >
            {isCameraOff ? <VideoOff className="h-3.5 w-3.5" /> : <Video className="h-3.5 w-3.5" />}
            <span>{isCameraOff ? 'Start Video' : 'Stop Video'}</span>
          </button>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-2.5 w-full sm:w-auto">
          <button
            onClick={onCancel}
            className="flex-1 sm:flex-none px-5 py-2.5 rounded-full bg-white/5 hover:bg-white/10 text-slate-300 text-xs font-semibold transition-colors cursor-pointer"
          >
            Cancel
          </button>

          <button
            onClick={onJoin}
            className="flex-1 sm:flex-none flex items-center justify-center gap-2 px-6 py-2.5 rounded-full bg-blue-600 hover:bg-blue-700 active:scale-95 text-white text-xs font-semibold shadow-md shadow-blue-500/25 transition-all cursor-pointer"
          >
            <span>Join now</span>
            <ArrowRight className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
};
