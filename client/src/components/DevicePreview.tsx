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

    if (stream && !isCameraOff) {
      video.muted = true;
      video.defaultMuted = true;
      if (video.srcObject !== stream) {
        video.srcObject = stream;
      }
      video.play().catch((err) => {
        console.warn('[DevicePreview] Video play error:', err);
      });
    } else {
      video.pause();
      video.srcObject = null;
    }

    return () => {
      if (video) {
        try {
          video.pause();
          video.srcObject = null;
          video.load();
        } catch (e) {}
      }
    };
  }, [stream, isCameraOff]);

  const hasVideoTrack = Boolean(stream && stream.getVideoTracks().length > 0 && !isCameraOff);
  const initials = username.split(' ').map((n) => n[0]).join('').toUpperCase().slice(0, 2) || 'U';

  return (
    <div className="flex flex-col items-center justify-center w-full max-w-xl lg:max-w-2xl mx-auto p-4 sm:p-6 lg:p-8 bg-white border border-slate-200/80 rounded-2xl sm:rounded-3xl shadow-sm text-slate-900 font-sans">
      {/* Header */}
      <div className="w-full flex items-center justify-between pb-3 sm:pb-4 border-b border-slate-100 mb-4 sm:mb-5">
        <div>
          <h2 className="text-lg sm:text-2xl font-bold text-slate-900 tracking-tight">Ready to join?</h2>
          <p className="text-xs sm:text-sm text-slate-500 mt-0.5">
            Room: <span className="text-blue-600 font-mono font-semibold">{roomId}</span>
          </p>
        </div>
        <div className="flex items-center gap-1.5 px-2.5 sm:px-3 py-1 sm:py-1.5 rounded-full bg-slate-100/90 border border-slate-200/60 text-xs sm:text-sm font-medium text-slate-700 shadow-2xs">
          <User className="h-3.5 w-3.5 text-slate-500 stroke-[2]" />
          <span className="truncate max-w-[90px] xs:max-w-[120px]">{username}</span>
          {role === 'tester' && (
            <span className="text-[10px] text-amber-700 bg-amber-50 border border-amber-200/60 px-2 py-0.5 rounded-full font-bold uppercase tracking-wider ml-1">
              Tester
            </span>
          )}
        </div>
      </div>

      {/* Permission Warning */}
      {mediaError && (
        <div className="w-full mb-4 flex items-center gap-2.5 p-3 sm:p-3.5 rounded-xl sm:rounded-2xl bg-red-50 border border-red-200 text-red-700 text-xs sm:text-sm animate-shake">
          <AlertCircle className="h-4 w-4 shrink-0 text-red-600" />
          <span>{mediaError}</span>
        </div>
      )}

      {/* Camera Preview Tile */}
      <div className="relative w-full aspect-video max-h-[220px] xs:max-h-[280px] sm:max-h-[360px] rounded-xl sm:rounded-2xl overflow-hidden bg-slate-950 border border-slate-200/80 flex items-center justify-center shadow-inner">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
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
          <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium backdrop-blur-md transition-colors ${
            isMicMuted ? 'bg-red-500 text-white' : 'bg-black/60 text-white'
          }`}>
            {isMicMuted ? <MicOff className="h-3 w-3" /> : <Mic className="h-3 w-3" />}
            <span>{isMicMuted ? 'Muted' : 'Mic on'}</span>
          </div>

          <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium backdrop-blur-md transition-colors ${
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
            type="button"
            onClick={onToggleMic}
            className={`flex-1 sm:flex-none flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl text-xs sm:text-sm font-medium transition-all cursor-pointer shadow-2xs ${
              isMicMuted
                ? 'bg-red-50 hover:bg-red-100 text-red-600 border border-red-200'
                : 'bg-white hover:bg-slate-50 text-slate-700 border border-slate-200/80 hover:border-slate-300'
            }`}
          >
            {isMicMuted ? <MicOff className="h-3.5 w-3.5" /> : <Mic className="h-3.5 w-3.5" />}
            <span>{isMicMuted ? 'Unmute' : 'Mute'}</span>
          </button>

          <button
            type="button"
            onClick={onToggleCamera}
            className={`flex-1 sm:flex-none flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl text-xs sm:text-sm font-medium transition-all cursor-pointer shadow-2xs ${
              isCameraOff
                ? 'bg-red-50 hover:bg-red-100 text-red-600 border border-red-200'
                : 'bg-white hover:bg-slate-50 text-slate-700 border border-slate-200/80 hover:border-slate-300'
            }`}
          >
            {isCameraOff ? <VideoOff className="h-3.5 w-3.5" /> : <Video className="h-3.5 w-3.5" />}
            <span>{isCameraOff ? 'Start Video' : 'Stop Video'}</span>
          </button>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-2.5 w-full sm:w-auto">
          <button
            type="button"
            onClick={onCancel}
            className="flex-1 sm:flex-none px-5 py-2.5 rounded-xl bg-white hover:bg-slate-50 text-slate-600 border border-slate-200 text-xs sm:text-sm font-medium transition-colors cursor-pointer shadow-2xs"
          >
            Cancel
          </button>

          <button
            type="button"
            onClick={onJoin}
            className="flex-1 sm:flex-none flex items-center justify-center gap-2 px-6 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-700 active:scale-[0.99] text-white text-xs sm:text-sm font-semibold shadow-xs transition-all cursor-pointer"
          >
            <span>Join now</span>
            <ArrowRight className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );
};
