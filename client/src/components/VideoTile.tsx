import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  Mic,
  MicOff,
  VideoOff,
  Shield,
  VolumeX,
  Maximize2,
  GripHorizontal,
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
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const tileRef = useRef<HTMLDivElement>(null);
  const [, setTrackRevision] = useState<number>(0);
  const [audioBlocked, setAudioBlocked] = useState<boolean>(false);

  // Floating movable & resizable state
  const [position, setPosition] = useState<{ x: number; y: number } | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [sizePreset, setSizePreset] = useState<SizePreset>('md');
  const [dragOffset, setDragOffset] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

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
        video.removeEventListener('loadedmetadata', handleLoadedMetadata);
        tracks.forEach((track) => track.removeEventListener('unmute', attemptPlay));
        stream.removeEventListener('addtrack', handleTracksChange);
        stream.removeEventListener('removetrack', handleTracksChange);
      };
    } else {
      video.srcObject = null;
    }
  }, [stream, isLocal]);

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

  // --- Boundary clamping helper ---
  const clampToBounds = useCallback(
    (x: number, y: number, currentTile?: HTMLElement | null, currentParent?: HTMLElement | null) => {
      const tile = currentTile || tileRef.current;
      const parent = currentParent || tile?.parentElement;
      if (!tile || !parent) return { x, y };

      const parentRect = parent.getBoundingClientRect();
      const tileWidth = tile.offsetWidth || 280;
      const tileHeight = tile.offsetHeight || 160;

      const padding = 12;
      const maxX = Math.max(padding, parentRect.width - tileWidth - padding);
      const maxY = Math.max(padding, parentRect.height - tileHeight - padding);

      return {
        x: Math.max(padding, Math.min(x, maxX)),
        y: Math.max(padding, Math.min(y, maxY)),
      };
    },
    []
  );

  // --- Movable / Dragging via pointer down ---
  const handlePointerDown = (e: React.PointerEvent) => {
    if (!isFloating || !tileRef.current) return;
    const parent = tileRef.current.parentElement;
    if (!parent) return;

    const parentRect = parent.getBoundingClientRect();
    const tileRect = tileRef.current.getBoundingClientRect();

    const currentX = tileRect.left - parentRect.left;
    const currentY = tileRect.top - parentRect.top;

    setDragOffset({
      x: e.clientX - tileRect.left,
      y: e.clientY - tileRect.top,
    });
    setPosition({ x: currentX, y: currentY });
    setIsDragging(true);
  };

  // --- Window-level pointer listeners to guarantee smooth, un-escapable dragging ---
  useEffect(() => {
    if (!isDragging) return;

    const handlePointerMove = (e: PointerEvent) => {
      if (!tileRef.current) return;
      const parent = tileRef.current.parentElement;
      if (!parent) return;

      const parentRect = parent.getBoundingClientRect();
      const rawX = e.clientX - parentRect.left - dragOffset.x;
      const rawY = e.clientY - parentRect.top - dragOffset.y;

      const clamped = clampToBounds(rawX, rawY, tileRef.current, parent);
      setPosition(clamped);
    };

    const handlePointerUp = () => {
      setIsDragging(false);
    };

    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', handlePointerUp);
    window.addEventListener('pointercancel', handlePointerUp);

    return () => {
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerUp);
      window.removeEventListener('pointercancel', handlePointerUp);
    };
  }, [isDragging, dragOffset, clampToBounds]);

  // --- Auto-reclamp on sizePreset change or window resize (keeps it 100% visible inside the window) ---
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

  // --- Size Cycle / Toggle ---
  const cycleSize = (e: React.MouseEvent) => {
    e.stopPropagation();
    const order: SizePreset[] = ['sm', 'md', 'lg', 'xl'];
    const nextIndex = (order.indexOf(sizePreset) + 1) % order.length;
    setSizePreset(order[nextIndex]);
  };

  const hasVideoTrack = stream && stream.getVideoTracks().length > 0 && stream.getVideoTracks()[0].enabled && !isVideoOff;

  // Responsive size styling with max constraints so it never overflows parent bounds
  const sizeClasses: Record<SizePreset, string> = {
    sm: 'w-44 sm:w-52',
    md: 'w-60 sm:w-72',
    lg: 'w-72 sm:w-96',
    xl: 'w-84 sm:w-[440px]',
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
      className={`relative overflow-hidden rounded-md border border-zinc-800 bg-[#070709] shadow-2xl transition-[width,height,border-color] duration-150 select-none ${
        isFloating
          ? `absolute ${position ? '' : 'bottom-4 right-4 sm:bottom-6 sm:right-6'} z-30 ${sizeClasses[sizePreset]} max-w-[calc(100%-24px)] max-h-[calc(100%-24px)] aspect-video border-orange-500/40 shadow-black/80 hover:border-orange-500/70`
          : 'w-full h-full min-h-0'
      } ${className}`}
    >
      {/* Video Element */}
      {stream && (
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted={isLocal}
          className={`absolute inset-0 h-full w-full object-cover transition-opacity duration-200 pointer-events-none ${
            hasVideoTrack ? 'opacity-100' : 'opacity-0'
          } ${isLocal ? 'scale-x-[-1]' : ''}`}
        />
      )}

      {/* Camera Disabled / Offline State */}
      {!hasVideoTrack && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/90 text-zinc-400 p-3 pointer-events-none">
          <div className="flex items-center justify-center h-10 w-10 sm:h-12 sm:w-12 rounded-md bg-zinc-900 border border-zinc-800 mb-2">
            <VideoOff className="h-5 w-5 sm:h-6 sm:w-6 text-rose-400" />
          </div>
          <span className="text-[11px] sm:text-xs font-mono font-medium text-zinc-300">Camera is off</span>
          <span className="text-[10px] sm:text-[11px] text-zinc-500 font-mono mt-0.5">{username}</span>
        </div>
      )}

      {/* Framing Markers */}
      <div className="pointer-events-none absolute top-1.5 left-1.5 h-2 w-2 border-t border-l border-zinc-600" />
      <div className="pointer-events-none absolute top-1.5 right-1.5 h-2 w-2 border-t border-r border-zinc-600" />
      <div className="pointer-events-none absolute bottom-1.5 left-1.5 h-2 w-2 border-b border-l border-zinc-600" />
      <div className="pointer-events-none absolute bottom-1.5 right-1.5 h-2 w-2 border-b border-r border-zinc-600" />

      {/* Top Bar: Floating Drag Bar & Size Toggle */}
      {isFloating ? (
        <div
          onPointerDown={handlePointerDown}
          className="absolute top-0 left-0 right-0 z-20 flex items-center justify-between px-2.5 py-1.5 bg-gradient-to-b from-black/95 via-black/80 to-transparent cursor-move select-none border-b border-zinc-800/40"
          style={{ touchAction: 'none' }}
          title="Drag to reposition window"
        >
          {/* Identity & Drag Grip */}
          <div className="flex items-center gap-1.5 pointer-events-none">
            <GripHorizontal className="h-3.5 w-3.5 text-orange-400 shrink-0" />
            <span className="h-1.5 w-1.5 rounded-full bg-orange-400 animate-pulse shrink-0" />
            <span className="text-[10px] sm:text-[11px] font-mono font-semibold text-zinc-200 truncate max-w-[80px] sm:max-w-[130px]">
              {username} (You)
            </span>
          </div>

          {/* Size Preset Toggle */}
          <button
            type="button"
            onPointerDown={(e) => e.stopPropagation()}
            onClick={cycleSize}
            title={`Size: ${sizePreset.toUpperCase()} (Click to toggle S/M/L/XL)`}
            className="flex items-center gap-1 px-2 py-0.5 rounded bg-zinc-900/90 hover:bg-orange-600/20 text-orange-300 hover:text-orange-200 border border-zinc-700/80 hover:border-orange-500/60 text-[10px] font-mono font-bold transition-colors cursor-pointer"
          >
            <Maximize2 className="h-2.5 w-2.5" />
            <span>{sizePreset.toUpperCase()}</span>
          </button>
        </div>
      ) : (
        /* Non-floating Top Bar */
        <div className="absolute top-3 left-3 z-10 flex items-center gap-2 pointer-events-none">
          <div className="flex items-center gap-1.5 rounded bg-black/80 px-2.5 py-1 font-mono text-xs font-medium backdrop-blur-md border border-zinc-800">
            <span className={`h-2 w-2 rounded-full ${isLocal ? 'bg-orange-400' : 'bg-emerald-400'}`} />
            <span className="text-zinc-200">{username}</span>
          </div>

          {role === 'tester' ? (
            <span className="flex items-center gap-1 rounded bg-orange-500/15 px-2 py-0.5 text-[10px] font-mono font-semibold text-orange-400 border border-orange-500/40 uppercase">
              <Shield className="h-2.5 w-2.5" /> Tester
            </span>
          ) : (
            <span className="rounded bg-zinc-900/80 px-2 py-0.5 text-[10px] font-mono text-zinc-400 border border-zinc-800 uppercase">
              User
            </span>
          )}
        </div>
      )}

      {/* Autoplay Audio Block Banner */}
      {audioBlocked && !isLocal && (
        <button
          onClick={handleUnmuteClick}
          className="absolute top-3 right-3 z-30 flex items-center gap-1.5 px-3 py-1.5 rounded bg-orange-600 hover:bg-orange-500 text-white font-mono text-xs font-bold tracking-wider shadow-lg transition-colors cursor-pointer"
        >
          <VolumeX className="h-3.5 w-3.5" />
          <span>Click to Unmute Audio</span>
        </button>
      )}

      {/* Bottom Status Row */}
      <div className="absolute bottom-2 left-2.5 right-2.5 z-10 flex items-center justify-between pointer-events-none">
        {subtitle ? (
          <span className="rounded bg-black/85 px-1.5 py-0.5 text-[10px] font-mono text-zinc-400 border border-zinc-800/90 truncate max-w-[120px]">
            {subtitle}
          </span>
        ) : (
          <span />
        )}

        {/* Audio status indicator */}
        <div
          className={`flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-mono border backdrop-blur-sm ${
            isMuted
              ? 'bg-rose-500/15 text-rose-400 border-rose-500/40'
              : 'bg-black/80 text-emerald-400 border-emerald-800/50'
          }`}
        >
          {isMuted ? <MicOff className="h-2.5 w-2.5 text-rose-400" /> : <Mic className="h-2.5 w-2.5 text-emerald-400" />}
          <span className="uppercase tracking-wider">{isMuted ? 'Muted' : 'Live'}</span>
        </div>
      </div>
    </div>
  );
};
