import React, { useEffect, useRef } from 'react';
import { Mic, MicOff, Video, VideoOff, AlertCircle, ArrowRight, User } from 'lucide-react';
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

  const hasVideoTrack = stream && stream.getVideoTracks().length > 0 && !isCameraOff;
  const initials = username.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2) || 'U';

  return (
    <div className="flex flex-col items-center justify-center w-full max-w-2xl mx-auto p-6 sm:p-8 bg-[#28292c] border border-[#3c4043] rounded-2xl shadow-2xl">
      {/* Header */}
      <div className="w-full flex items-center justify-between pb-4 border-b border-[#3c4043] mb-5">
        <div>
          <h2 className="text-lg font-medium text-[#e8eaed]">Ready to join?</h2>
          <p className="text-sm text-[#9aa0a6] mt-0.5">
            Room: <span className="text-[#e8eaed] font-medium">{roomId}</span>
          </p>
        </div>
        <div className="flex items-center gap-2 text-sm text-[#e8eaed] bg-[#3c4043] px-3 py-1.5 rounded-full">
          <User className="h-3.5 w-3.5 text-[#9aa0a6]" />
          <span>{username}</span>
          {role === 'tester' && (
            <span className="text-[11px] text-[#8ab4f8] bg-[#1a73e8]/20 px-2 py-0.5 rounded-full font-medium">Tester</span>
          )}
        </div>
      </div>

      {/* Permission Warning */}
      {mediaError && (
        <div className="w-full mb-4 flex items-center gap-2 p-3 rounded-xl bg-[#ea4335]/15 border border-[#ea4335]/30 text-[#ea4335] text-sm">
          <AlertCircle className="h-4 w-4 shrink-0" />
          <span>{mediaError}</span>
        </div>
      )}

      {/* Camera Preview */}
      <div className="relative w-full aspect-video max-h-[400px] rounded-2xl overflow-hidden bg-[#202124] border border-[#3c4043] flex items-center justify-center">
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
            <div className="h-20 w-20 rounded-full bg-[#1a73e8] flex items-center justify-center text-white text-2xl font-medium">
              {initials}
            </div>
            <span className="text-sm text-[#9aa0a6]">Camera is off</span>
          </div>
        )}

        {/* Status badges */}
        <div className="absolute bottom-3 left-3 flex items-center gap-2">
          <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs backdrop-blur-md ${
            isMicMuted ? 'bg-[#ea4335] text-white' : 'bg-black/50 text-white'
          }`}>
            {isMicMuted ? <MicOff className="h-3 w-3" /> : <Mic className="h-3 w-3" />}
            <span>{isMicMuted ? 'Microphone off' : 'Microphone on'}</span>
          </div>

          <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs backdrop-blur-md ${
            isCameraOff ? 'bg-[#ea4335] text-white' : 'bg-black/50 text-white'
          }`}>
            {isCameraOff ? <VideoOff className="h-3 w-3" /> : <Video className="h-3 w-3" />}
            <span>{isCameraOff ? 'Camera off' : 'Camera on'}</span>
          </div>
        </div>
      </div>

      {/* Controls & Join */}
      <div className="w-full mt-6 flex flex-col sm:flex-row items-center justify-between gap-3">
        <div className="flex items-center gap-2 w-full sm:w-auto">
          <button
            onClick={onToggleMic}
            className={`flex-1 sm:flex-none flex items-center justify-center gap-2 px-4 py-2.5 rounded-full text-sm font-medium transition-colors cursor-pointer ${
              isMicMuted
                ? 'bg-[#ea4335] text-white hover:bg-[#d93025]'
                : 'bg-[#3c4043] text-[#e8eaed] hover:bg-[#4a4d51]'
            }`}
          >
            {isMicMuted ? <MicOff className="h-4 w-4 shrink-0" /> : <Mic className="h-4 w-4 shrink-0" />}
            <span>{isMicMuted ? 'Unmute' : 'Mute'}</span>
          </button>

          <button
            onClick={onToggleCamera}
            className={`flex-1 sm:flex-none flex items-center justify-center gap-2 px-4 py-2.5 rounded-full text-sm font-medium transition-colors cursor-pointer ${
              isCameraOff
                ? 'bg-[#ea4335] text-white hover:bg-[#d93025]'
                : 'bg-[#3c4043] text-[#e8eaed] hover:bg-[#4a4d51]'
            }`}
          >
            {isCameraOff ? <VideoOff className="h-4 w-4 shrink-0" /> : <Video className="h-4 w-4 shrink-0" />}
            <span>{isCameraOff ? 'Start Video' : 'Stop Video'}</span>
          </button>
        </div>

        <div className="flex items-center gap-2.5 w-full sm:w-auto">
          <button
            onClick={onCancel}
            className="flex-1 sm:flex-none px-5 py-2.5 rounded-full bg-[#3c4043] text-[#e8eaed] text-sm font-medium hover:bg-[#4a4d51] transition-colors cursor-pointer"
          >
            Cancel
          </button>

          <button
            onClick={onJoin}
            className="flex-1 sm:flex-none flex items-center justify-center gap-2 px-6 py-2.5 rounded-full bg-[#1a73e8] hover:bg-[#1557b0] text-white text-sm font-medium shadow-md transition-colors active:scale-95 cursor-pointer"
          >
            <span>Join now</span>
            <ArrowRight className="h-4 w-4 shrink-0" />
          </button>
        </div>
      </div>
    </div>
  );
};
