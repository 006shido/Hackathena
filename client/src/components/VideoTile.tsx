import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  Mic,
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
  role = 'user',
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
      video.srcObject = null;
      setIsHardwareBuffering(false);
    }
  }, [stream, isLocal, isVideoOff]);

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
      const tileWidth = tile.offsetWidth || 280;
      const tileHeight = tile.offsetHeight || 160;

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

  const initials = username.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2) || 'U';

  const sizeClasses: Record<SizePreset, string> = {
    sm: 'w-44 sm:w-52',
    md: 'w-56 sm:w-68 md:w-72',
    lg: 'w-68 sm:w-80 md:w-88',
    xl: 'w-76 sm:w-92 md:w-[380px]',
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
          ? `absolute ${position ? '' : 'bottom-4 right-4 sm:bottom-6 sm:right-6'} z-30 ${sizeClasses[sizePreset]} max-w-[calc(100%-32px)] max-h-[calc(100%-32px)] aspect-video border-[#5f6368] shadow-2xl hover:border-[#8ab4f8]`
          : 'relative w-full h-full min-h-0'
      } ${
        isDeepfakeAlert && !isLocal
          ? 'border-[#ea4335] ring-2 ring-[#ea4335]/40'
          : 'border-transparent'
      } overflow-hidden rounded-2xl bg-[#3c4043] shadow-lg transition-[width,height,border-color] duration-150 select-none ${className}`}
    >
      {/* Video */}
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

      {/* Hardware Muted / Privacy Shutter Notification */}
      {isLocal && isHardwareBuffering && !isVideoOff && (
        <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-[#202124]/90 backdrop-blur-sm p-4 text-center select-none animate-in fade-in duration-200">
          <div className="h-12 w-12 rounded-full bg-[#fbbc04]/15 border border-[#fbbc04]/30 flex items-center justify-center mb-2.5 text-[#fbbc04]">
            <VideoOff className="h-5 w-5" />
          </div>
          <h4 className="text-sm font-medium text-[#e8eaed] mb-1">
            Webcam Sensor Muted / In Use
          </h4>
          <p className="text-xs text-[#9aa0a6] max-w-xs leading-relaxed">
            Check your physical camera shutter slider, keyboard hotkey (<span className="text-[#8ab4f8] font-mono">Fn + F10</span> / <span className="text-[#8ab4f8] font-mono">F9</span>), or close other apps using the camera.
          </p>
        </div>
      )}

      {/* Camera Off Avatar */}
      {(!hasVideoTrack || isVideoOff) && !isHardwareBuffering && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-[#28292c] p-3 pointer-events-none">
          <div className="flex items-center justify-center h-16 w-16 sm:h-20 sm:w-20 rounded-full mb-2 text-white text-xl sm:text-2xl font-semibold bg-[#1a73e8]">
            {initials}
          </div>
          <span className="text-xs text-[#9aa0a6]">{username}</span>
        </div>
      )}

      {/* Floating Drag Bar */}
      {isFloating ? (
        <div
          onPointerDown={handlePointerDown}
          className="absolute top-0 left-0 right-0 z-20 flex items-center justify-between px-2.5 py-1.5 bg-gradient-to-b from-black/80 to-transparent cursor-move select-none"
          style={{ touchAction: 'none' }}
          title="Drag to reposition"
        >
          <div className="flex items-center gap-1.5 pointer-events-none">
            <GripHorizontal className="h-3.5 w-3.5 text-white/70 shrink-0" />
            <span className="text-[11px] font-medium text-white/90 truncate max-w-[100px] sm:max-w-[130px]">
              {username} (You)
            </span>
          </div>

          <button
            type="button"
            onPointerDown={(e) => e.stopPropagation()}
            onClick={cycleSize}
            title={`Size: ${sizePreset.toUpperCase()}`}
            className="flex items-center gap-1 px-1.5 py-0.5 rounded-md bg-white/10 hover:bg-white/20 text-white/80 hover:text-white text-[10px] font-medium transition-colors cursor-pointer"
          >
            <Maximize2 className="h-2.5 w-2.5" />
            <span>{sizePreset.toUpperCase()}</span>
          </button>
        </div>
      ) : null}

      {/* Deepfake Alert */}
      {isDeepfakeAlert && !isLocal && (
        <div className="absolute top-3 right-3 z-30 flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#ea4335] text-white text-xs font-medium shadow-lg animate-pulse">
          <AlertTriangle className="h-3.5 w-3.5" />
          <span>Deepfake Alert ({deepfakeScore}%)</span>
        </div>
      )}

      {/* Unmute Banner */}
      {audioBlocked && !isLocal && (
        <button
          onClick={handleUnmuteClick}
          className="absolute top-3 right-3 z-30 flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#1a73e8] hover:bg-[#1557b0] text-white text-xs font-medium shadow-lg transition-colors cursor-pointer"
        >
          <VolumeX className="h-3.5 w-3.5" />
          <span>Click to Unmute</span>
        </button>
      )}

      {/* Bottom Name + Mic - like Google Meet screenshot */}
      <div className="absolute bottom-3 left-3 z-10 flex items-center gap-2 pointer-events-none">
        <div className="flex items-center gap-1.5 bg-black/50 backdrop-blur-sm rounded-md px-2 py-1">
          {isMuted ? (
            <MicOff className="h-3.5 w-3.5 text-[#ea4335]" />
          ) : (
            <div className="flex items-end gap-[2px] h-3">
              <div className="w-[2px] h-[40%] bg-[#34a853] rounded-full animate-pulse" />
              <div className="w-[2px] h-[75%] bg-[#34a853] rounded-full animate-pulse" style={{ animationDelay: '0.1s' }} />
              <div className="w-[2px] h-[100%] bg-[#34a853] rounded-full animate-pulse" style={{ animationDelay: '0.2s' }} />
              <div className="w-[2px] h-[60%] bg-[#34a853] rounded-full animate-pulse" style={{ animationDelay: '0.15s' }} />
            </div>
          )}
          <span className="text-xs text-white font-medium drop-shadow-sm">
            {username}{isLocal ? ' (You)' : ''}
          </span>
        </div>
      </div>
    </div>
  );
};
