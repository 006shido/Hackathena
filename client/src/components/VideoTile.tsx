import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  MicOff,
  VideoOff,
  VolumeX,
  Maximize2,
  GripHorizontal,
  AlertTriangle,
} from 'lucide-react';
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
  isDeepfakeAlert?: boolean;
  deepfakeScore?: number;
}

type SizePreset = 'sm' | 'md' | 'lg' | 'xl';

export const VideoTile: React.FC<VideoTileProps> = ({
  stream,
  username,
  role: _role = 'user',
  isLocal = false,
  isMuted = false,
  isVideoOff = false,
  subtitle,
  isFloating = false,
  className = '',
  isDeepfakeAlert = false,
  deepfakeScore = 0,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const tileRef = useRef<HTMLDivElement>(null);
  const [, setTrackRevision] = useState<number>(0);
  const [audioBlocked, setAudioBlocked] = useState<boolean>(false);
  const [isHardwareBuffering, setIsHardwareBuffering] = useState<boolean>(false);

  // Floating movable & resizable state
  const [position, setPosition] = useState<{ x: number; y: number } | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [sizePreset, setSizePreset] = useState<SizePreset>('md');
  const dragStateRef = useRef<{
    startX: number;
    startY: number;
    startLeft: number;
    startTop: number;
    isDragging: boolean;
  }>({
    startX: 0,
    startY: 0,
    startLeft: 0,
    startTop: 0,
    isDragging: false,
  });

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    if (stream) {
      if (video.srcObject !== stream) {
        video.srcObject = stream;
      }

      const attemptPlay = async () => {
        try {
          if (isLocal) {
            video.muted = true;
            video.defaultMuted = true;
          }
          await video.play();
          setAudioBlocked(false);
        } catch (err: unknown) {
          console.warn('[VideoTile] Autoplay prevented, fallback to muted:', (err as Error)?.message);
          if (!isLocal) {
            video.muted = true;
            video.defaultMuted = true;
            try {
              await video.play();
              setAudioBlocked(true);
            } catch (playErr) {
              console.error('[VideoTile] Video play error after mute:', playErr);
            }
          }
        }
      };

      const videoTrack = stream.getVideoTracks()[0];
      let watchdogTimer: any = null;

      const checkFrameArrival = () => {
        if (!isLocal || isVideoOff || !videoTrack) return;
        if (videoTrack.muted) {
          setIsHardwareBuffering(true);
          return;
        }
        if (video && (video.videoWidth === 0 || video.readyState < 2)) {
          setIsHardwareBuffering(true);
        } else {
          setIsHardwareBuffering(false);
        }
      };

      const handlePlaying = () => {
        setIsHardwareBuffering(false);
        if (watchdogTimer) clearTimeout(watchdogTimer);
      };

      const handleTrackMute = () => {
        if (isLocal) setIsHardwareBuffering(true);
      };

      const handleTrackUnmute = () => {
        if (isLocal) {
          setIsHardwareBuffering(false);
          attemptPlay();
        }
      };

      if (isLocal && videoTrack && !isVideoOff) {
        if (videoTrack.muted) {
          setIsHardwareBuffering(true);
        } else {
          watchdogTimer = setTimeout(checkFrameArrival, 2000);
        }
        videoTrack.addEventListener('mute', handleTrackMute);
        videoTrack.addEventListener('unmute', handleTrackUnmute);
      } else {
        setIsHardwareBuffering(false);
      }

      video.addEventListener('playing', handlePlaying);
      video.addEventListener('loadeddata', handlePlaying);

      const handleLoadedMetadata = () => attemptPlay();
      video.addEventListener('loadedmetadata', handleLoadedMetadata);
      attemptPlay();

      const handleTracksChange = () => {
        setTrackRevision((prev: number) => prev + 1);
        attemptPlay();
      };

      const tracks = stream.getTracks();
      tracks.forEach((track) => track.addEventListener('unmute', attemptPlay));
      stream.addEventListener('addtrack', handleTracksChange);
      stream.addEventListener('removetrack', handleTracksChange);

      return () => {
        if (watchdogTimer) clearTimeout(watchdogTimer);
        video.removeEventListener('playing', handlePlaying);
        video.removeEventListener('loadeddata', handlePlaying);
        video.removeEventListener('loadedmetadata', handleLoadedMetadata);
        tracks.forEach((track) => track.removeEventListener('unmute', attemptPlay));
        if (videoTrack) {
          videoTrack.removeEventListener('mute', handleTrackMute);
          videoTrack.removeEventListener('unmute', handleTrackUnmute);
        }
        stream.removeEventListener('addtrack', handleTracksChange);
        stream.removeEventListener('removetrack', handleTracksChange);
      };
    } else {
      try {
        video.pause();
        video.srcObject = null;
      } catch (e) {}
      setIsHardwareBuffering(false);
    }
  }, [stream, isLocal, isVideoOff]);

  useEffect(() => {
    return () => {
      const video = videoRef.current;
      if (video) {
        try {
          video.pause();
          video.srcObject = null;
          video.load();
        } catch (e) {}
      }
    };
  }, []);


  const handleUnmuteClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (videoRef.current && !isLocal) {
      videoRef.current.muted = false;
      videoRef.current
        .play()
        .then(() => setAudioBlocked(false))
        .catch((err) => console.warn('Unmute error:', err));
    }
  };

  const clampToBounds = useCallback(
    (x: number, y: number, currentTile?: HTMLElement | null, currentParent?: HTMLElement | null) => {
      const tile = currentTile || tileRef.current;
      const parent = currentParent || tile?.parentElement;
      if (!tile || !parent) return { x, y };

      const parentWidth = parent.clientWidth || parent.getBoundingClientRect().width;
      const parentHeight = parent.clientHeight || parent.getBoundingClientRect().height;
      const tileWidth = tile.offsetWidth || 260;
      const tileHeight = tile.offsetHeight || 150;

      const padding = 16;
      const maxX = Math.max(padding, parentWidth - tileWidth - padding);
      const maxY = Math.max(padding, parentHeight - tileHeight - padding);

      return {
        x: Math.min(Math.max(padding, x), maxX),
        y: Math.min(Math.max(padding, y), maxY),
      };
    },
    []
  );

  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!isFloating || !tileRef.current) return;
    const parent = tileRef.current.parentElement;
    if (!parent) return;

    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch (_) {}

    const parentRect = parent.getBoundingClientRect();
    const tileRect = tileRef.current.getBoundingClientRect();

    const currentLeft = tileRect.left - parentRect.left - (parent.clientLeft || 0);
    const currentTop = tileRect.top - parentRect.top - (parent.clientTop || 0);

    const clamped = clampToBounds(currentLeft, currentTop, tileRef.current, parent);

    dragStateRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      startLeft: clamped.x,
      startTop: clamped.y,
      isDragging: true,
    };

    setPosition(clamped);
    setIsDragging(true);
  };

  useEffect(() => {
    if (!isDragging) return;

    const handlePointerMove = (e: PointerEvent) => {
      if (!dragStateRef.current.isDragging || !tileRef.current) return;
      const parent = tileRef.current.parentElement;
      if (!parent) return;

      const deltaX = e.clientX - dragStateRef.current.startX;
      const deltaY = e.clientY - dragStateRef.current.startY;

      const rawX = dragStateRef.current.startLeft + deltaX;
      const rawY = dragStateRef.current.startTop + deltaY;

      const clamped = clampToBounds(rawX, rawY, tileRef.current, parent);
      setPosition(clamped);
    };

    const handlePointerUp = () => {
      dragStateRef.current.isDragging = false;
      setIsDragging(false);
    };

    window.addEventListener('pointermove', handlePointerMove, { passive: true });
    window.addEventListener('pointerup', handlePointerUp);
    window.addEventListener('pointercancel', handlePointerUp);

    return () => {
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerUp);
      window.removeEventListener('pointercancel', handlePointerUp);
    };
  }, [isDragging, clampToBounds]);

  useEffect(() => {
    if (!isFloating) return;

    const handleReclamp = () => {
      requestAnimationFrame(() => {
        if (!tileRef.current) return;
        const parent = tileRef.current.parentElement;
        if (!parent) return;

        setPosition((prev) => {
          if (!prev) return null;
          return clampToBounds(prev.x, prev.y, tileRef.current, parent);
        });
      });
    };

    handleReclamp();
    window.addEventListener('resize', handleReclamp);
    return () => window.removeEventListener('resize', handleReclamp);
  }, [sizePreset, isFloating, clampToBounds]);

  const cycleSize = (e: React.MouseEvent) => {
    e.stopPropagation();
    const order: SizePreset[] = ['sm', 'md', 'lg', 'xl'];
    const nextIndex = (order.indexOf(sizePreset) + 1) % order.length;
    setSizePreset(order[nextIndex]);
  };

  const hasVideoTrack = Boolean(
    stream &&
    stream.getVideoTracks().length > 0 &&
    (isLocal ? (stream.getVideoTracks()[0].enabled && !isVideoOff) : true)
  );

  const initials = username.split(' ').map((n) => n[0]).join('').toUpperCase().slice(0, 2) || 'U';

  const sizeClasses: Record<SizePreset, string> = {
    sm: 'w-36 sm:w-48',
    md: 'w-48 sm:w-60 md:w-68',
    lg: 'w-56 sm:w-72 md:w-80',
    xl: 'w-64 sm:w-84 md:w-96',
  };

  return (
    <div
      ref={tileRef}
      onClick={audioBlocked ? handleUnmuteClick : undefined}
      style={
        isFloating && position
          ? {
              left: `${position.x}px`,
              top: `${position.y}px`,
              bottom: 'auto',
              right: 'auto',
            }
          : undefined
      }
      className={`${
        isFloating
          ? `absolute ${position ? '' : 'bottom-4 right-4 sm:bottom-6 sm:right-6'} z-30 ${sizeClasses[sizePreset]} max-w-[calc(100%-32px)] max-h-[calc(100%-32px)] aspect-video border border-white/15 shadow-2xl hover:border-blue-500/50`
          : 'relative w-full h-full min-h-0'
      } ${
        isDeepfakeAlert && !isLocal
          ? 'ring-2 ring-red-500 shadow-red-500/20 shadow-lg'
          : ''
      } overflow-hidden rounded-2xl sm:rounded-3xl bg-[#16181d] shadow-xl transition-[width,height,border-color] duration-150 select-none ${className}`}
    >
      {/* Video Element */}
      <video
        ref={(el) => {
          videoRef.current = el;
          if (el && stream) {
            if (isLocal) {
              el.muted = true;
              el.defaultMuted = true;
            }
            if (el.srcObject !== stream) {
              el.srcObject = stream;
            }
            el.play().catch(() => {});
          }
        }}
        autoPlay
        playsInline
        muted={isLocal}
        onLoadedMetadata={(e) => {
          const el = e.currentTarget;
          if (isLocal) {
            el.muted = true;
            el.defaultMuted = true;
          }
          el.play().catch(() => {});
        }}
        className={`absolute inset-0 h-full w-full object-cover transition-opacity duration-200 pointer-events-none ${
          hasVideoTrack && !isHardwareBuffering ? 'opacity-100' : 'opacity-0'
        } ${isLocal ? 'scale-x-[-1]' : ''}`}
      />

      {/* Hardware Muted Shutter Notification */}
      {isLocal && isHardwareBuffering && !isVideoOff && (
        <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-black/85 backdrop-blur-md p-4 text-center select-none animate-in fade-in duration-200">
          <div className="h-11 w-11 rounded-full bg-amber-500/15 border border-amber-500/30 flex items-center justify-center mb-2 text-amber-400">
            <VideoOff className="h-5 w-5" />
          </div>
          <h4 className="text-xs sm:text-sm font-semibold text-white mb-1">
            Webcam Sensor Muted
          </h4>
          <p className="text-[11px] text-slate-400 max-w-xs leading-relaxed">
            Check your physical camera shutter slider or keyboard hotkey.
          </p>
        </div>
      )}

      {/* Camera Off Avatar Placeholder */}
      {(!hasVideoTrack || isVideoOff) && !isHardwareBuffering && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-gradient-to-b from-[#1c1f26] to-[#14161c] p-3 pointer-events-none">
          <div className="flex items-center justify-center h-16 w-16 sm:h-20 sm:w-20 rounded-full mb-2 text-white text-xl sm:text-2xl font-bold bg-blue-600 shadow-lg shadow-blue-600/30">
            {initials}
          </div>
          <span className="text-xs font-medium text-slate-400">{username}</span>
        </div>
      )}

      {/* Floating Drag Bar (for PiP view) */}
      {isFloating && (
        <div
          onPointerDown={handlePointerDown}
          className="absolute top-0 left-0 right-0 z-20 flex items-center justify-between px-2.5 py-1.5 bg-gradient-to-b from-black/80 to-transparent cursor-move select-none"
          style={{ touchAction: 'none' }}
          title="Drag to reposition"
        >
          <div className="flex items-center gap-1.5 pointer-events-none">
            <GripHorizontal className="h-3.5 w-3.5 text-white/70 shrink-0" />
            <span className="text-[10px] font-medium text-white/90 truncate max-w-[100px]">
              You
            </span>
          </div>

          <button
            type="button"
            onPointerDown={(e) => e.stopPropagation()}
            onClick={cycleSize}
            title={`Size: ${sizePreset.toUpperCase()}`}
            className="flex items-center gap-1 px-1.5 py-0.5 rounded-md bg-white/10 hover:bg-white/20 text-white/90 text-[10px] font-medium transition-colors cursor-pointer"
          >
            <Maximize2 className="h-2.5 w-2.5" />
            <span>{sizePreset.toUpperCase()}</span>
          </button>
        </div>
      )}

      {/* Autonomous Deepfake Alert Floating Pill */}
      {isDeepfakeAlert && !isLocal && (
        <div className="absolute top-3 right-3 z-30 flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-red-600 text-white text-xs font-semibold shadow-lg shadow-red-600/40 animate-pulse">
          <AlertTriangle className="h-3.5 w-3.5" />
          <span>Deepfake Alert ({deepfakeScore}%)</span>
        </div>
      )}

      {/* Unmute Fallback Banner */}
      {audioBlocked && !isLocal && (
        <button
          onClick={handleUnmuteClick}
          className="absolute top-3 right-3 z-30 flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-blue-600 hover:bg-blue-700 text-white text-xs font-medium shadow-lg transition-colors cursor-pointer"
        >
          <VolumeX className="h-3.5 w-3.5" />
          <span>Click to Unmute</span>
        </button>
      )}

      {/* Bottom Name Badge / Subtitle (Matching Screenshot 1 Right "You" pill) */}
      <div className="absolute bottom-3 left-3 z-10 flex items-center gap-2 pointer-events-none">
        <div className="flex items-center gap-1.5 bg-black/60 backdrop-blur-md border border-white/10 rounded-md px-2 py-1">
          {isMuted ? (
            <MicOff className="h-3 w-3 text-red-400" />
          ) : (
            <div className="flex items-end gap-[2px] h-3">
              <div className="w-[2px] h-[40%] bg-emerald-400 rounded-full animate-pulse" />
              <div className="w-[2px] h-[80%] bg-emerald-400 rounded-full animate-pulse" style={{ animationDelay: '0.1s' }} />
              <div className="w-[2px] h-[100%] bg-emerald-400 rounded-full animate-pulse" style={{ animationDelay: '0.2s' }} />
            </div>
          )}
          <span className="text-[11px] sm:text-xs text-white font-medium drop-shadow-sm">
            {isLocal ? 'You' : username}
          </span>
          {subtitle && subtitle !== 'You' && (
            <span className="text-[10px] text-blue-400 font-mono">
              {subtitle}
            </span>
          )}
        </div>
      </div>
    </div>
  );
};
