import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Mic,
  MicOff,
  Video,
  VideoOff,
  ScreenShare,
  PhoneOff,
  ShieldAlert,
  Activity,
  ChevronsRight,
  Check,
  X,
} from 'lucide-react';

interface CallControlsProps {
  isMicMuted: boolean;
  isCameraOff: boolean;
  isScreenSharing: boolean;
  isTester: boolean;
  isTesterDrawerOpen: boolean;
  isSecurityPanelOpen?: boolean;
  attackMode: string;
  faceSwapActive?: boolean;
  isSliderActive?: boolean;
  onToggleSlider?: (active: boolean) => void;
  onToggleMic: () => void;
  onToggleCamera: () => void;
  onToggleScreenShare: () => void;
  onEndCall: () => void;
  onToggleTesterDrawer?: () => void;
  onToggleSecurityPanel?: () => void;
}

export const CallControls: React.FC<CallControlsProps> = ({
  isMicMuted,
  isCameraOff,
  isScreenSharing,
  isTester,
  isTesterDrawerOpen,
  isSecurityPanelOpen,
  attackMode,
  faceSwapActive,
  isSliderActive: externalSliderActive,
  onToggleSlider,
  onToggleMic,
  onToggleCamera,
  onToggleScreenShare,
  onEndCall,
  onToggleTesterDrawer,
  onToggleSecurityPanel,
}) => {
  const isAttackActive = attackMode !== 'none';

  // Internal slider state if not controlled externally
  const [internalSliderActive, setInternalSliderActive] = useState(false);
  const isSlider = externalSliderActive !== undefined ? externalSliderActive : internalSliderActive;

  const setSlider = (val: boolean) => {
    setInternalSliderActive(val);
    onToggleSlider?.(val);
  };

  const [isCompleted, setIsCompleted] = useState(false);

  const containerRef = useRef<HTMLDivElement>(null);
  const trackRef = useRef<HTMLDivElement>(null);
  const knobRef = useRef<HTMLDivElement>(null);
  const fillRef = useRef<HTMLDivElement>(null);
  const textRef = useRef<HTMLDivElement>(null);

  const isDraggingRef = useRef(false);
  const startXRef = useRef(0);
  const currentDragXRef = useRef(0);
  const maxDistanceRef = useRef(0);
  const rafIdRef = useRef<number | null>(null);

  // Compute maximum travel distance for knob
  const updateMaxDistance = useCallback(() => {
    if (trackRef.current && knobRef.current) {
      const trackWidth = trackRef.current.clientWidth;
      const knobWidth = knobRef.current.clientWidth;
      // 8px padding total + 36px space for the right cancel button
      maxDistanceRef.current = Math.max(0, trackWidth - knobWidth - 40);
    }
  }, []);

  // Reset knob and fill positions smoothly
  const resetSliderVisuals = useCallback((animated = true) => {
    currentDragXRef.current = 0;
    if (knobRef.current) {
      knobRef.current.style.transition = animated
        ? 'transform 0.32s cubic-bezier(0.16, 1, 0.3, 1)'
        : 'none';
      knobRef.current.style.transform = 'translate3d(0px, 0px, 0px)';
    }
    if (fillRef.current) {
      fillRef.current.style.transition = animated
        ? 'width 0.32s cubic-bezier(0.16, 1, 0.3, 1)'
        : 'none';
      fillRef.current.style.width = '12%';
    }
    if (textRef.current) {
      textRef.current.style.transition = animated ? 'opacity 0.2s ease' : 'none';
      textRef.current.style.opacity = '1';
    }
  }, []);

  useEffect(() => {
    if (isSlider) {
      setIsCompleted(false);
      resetSliderVisuals(false);

      const timer = setTimeout(() => {
        updateMaxDistance();
      }, 40);

      window.addEventListener('resize', updateMaxDistance);
      return () => {
        clearTimeout(timer);
        window.removeEventListener('resize', updateMaxDistance);
      };
    }
  }, [isSlider, updateMaxDistance, resetSliderVisuals]);

  // Click outside or ESC key to cancel slider and return to buttons
  useEffect(() => {
    if (!isSlider) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !isCompleted) {
        setSlider(false);
      }
    };

    const handlePointerDownOutside = (e: MouseEvent | TouchEvent) => {
      if (
        containerRef.current &&
        !containerRef.current.contains(e.target as Node) &&
        !isCompleted
      ) {
        setSlider(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    document.addEventListener('pointerdown', handlePointerDownOutside);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      document.removeEventListener('pointerdown', handlePointerDownOutside);
    };
  }, [isSlider, isCompleted]);

  // Direct GPU pointer drag
  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (isCompleted) return;
    updateMaxDistance();
    isDraggingRef.current = true;
    startXRef.current = e.clientX - currentDragXRef.current;
    (e.currentTarget as HTMLDivElement).setPointerCapture(e.pointerId);

    if (knobRef.current) knobRef.current.style.transition = 'none';
    if (fillRef.current) fillRef.current.style.transition = 'none';
    if (textRef.current) textRef.current.style.transition = 'none';
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!isDraggingRef.current || isCompleted) return;
    const rawDelta = e.clientX - startXRef.current;
    const max = maxDistanceRef.current;
    const clamped = Math.max(0, Math.min(rawDelta, max));

    currentDragXRef.current = clamped;

    if (rafIdRef.current !== null) {
      cancelAnimationFrame(rafIdRef.current);
    }

    rafIdRef.current = requestAnimationFrame(() => {
      if (knobRef.current) {
        knobRef.current.style.transform = `translate3d(${clamped}px, 0px, 0px)`;
      }

      const progress = max > 0 ? clamped / max : 0;
      if (fillRef.current) {
        const fillWidth = Math.min(100, Math.max(12, progress * 100 + 10));
        fillRef.current.style.width = `${fillWidth}%`;
      }

      if (textRef.current) {
        const opacity = Math.max(0, 1 - progress / 0.55);
        textRef.current.style.opacity = `${opacity}`;
      }

      // If reached completion threshold (85%), trigger finish
      if (max > 0 && progress >= 0.85) {
        triggerComplete(max);
      }
    });
  };

  const triggerComplete = (maxDistance: number) => {
    isDraggingRef.current = false;
    setIsCompleted(true);
    currentDragXRef.current = maxDistance;

    if (knobRef.current) {
      knobRef.current.style.transition = 'transform 0.2s cubic-bezier(0.16, 1, 0.3, 1)';
      knobRef.current.style.transform = `translate3d(${maxDistance}px, 0px, 0px)`;
    }
    if (fillRef.current) {
      fillRef.current.style.transition = 'width 0.2s cubic-bezier(0.16, 1, 0.3, 1)';
      fillRef.current.style.width = '100%';
    }
    if (textRef.current) {
      textRef.current.style.opacity = '0';
    }

    if (typeof navigator !== 'undefined' && navigator.vibrate) {
      navigator.vibrate([30, 40]);
    }

    setTimeout(() => {
      onEndCall();
    }, 240);
  };

  const handlePointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!isDraggingRef.current || isCompleted) return;
    isDraggingRef.current = false;
    try {
      (e.currentTarget as HTMLDivElement).releasePointerCapture(e.pointerId);
    } catch {
      // Ignore if pointer capture already released
    }

    const max = maxDistanceRef.current;
    if (max > 0 && currentDragXRef.current / max >= 0.85) {
      triggerComplete(max);
    } else {
      // Spring smoothly back to start
      resetSliderVisuals(true);
    }
  };

  return (
    <div ref={containerRef} className="flex justify-center select-none">
      {!isSlider ? (
        /* State 1: Normal Action Buttons (Image 2) - Snug fit, zero excess space */
        <div className="flex items-center justify-center gap-2 sm:gap-2.5 py-1.5 sm:py-2 px-2.5 sm:px-3 rounded-full bg-[#121316]/95 backdrop-blur-xl border border-[#22242a] shadow-2xl w-fit mx-auto animate-slider-fade-in">
          {/* Microphone */}
          <button
            onClick={onToggleMic}
            title={isMicMuted ? 'Turn on microphone' : 'Turn off microphone'}
            className={`flex h-10 w-10 sm:h-11 sm:w-11 items-center justify-center rounded-full transition-all shrink-0 cursor-pointer ${
              isMicMuted
                ? 'bg-[#ea3812] hover:bg-[#d6320e] text-white shadow-[0_0_16px_rgba(234,56,18,0.35)]'
                : 'bg-[#2b2d35] hover:bg-[#343740] text-white shadow-xs'
            }`}
          >
            {isMicMuted ? <MicOff className="h-4 w-4 sm:h-5 sm:w-5" /> : <Mic className="h-4 w-4 sm:h-5 sm:w-5" />}
          </button>

          {/* Camera */}
          <button
            onClick={onToggleCamera}
            title={isCameraOff ? 'Turn on camera' : 'Turn off camera'}
            className={`flex h-10 w-10 sm:h-11 sm:w-11 items-center justify-center rounded-full transition-all shrink-0 cursor-pointer ${
              isCameraOff
                ? 'bg-[#ea3812] hover:bg-[#d6320e] text-white shadow-[0_0_16px_rgba(234,56,18,0.35)]'
                : 'bg-[#2b2d35] hover:bg-[#343740] text-white shadow-xs'
            }`}
          >
            {isCameraOff ? <VideoOff className="h-4 w-4 sm:h-5 sm:w-5" /> : <Video className="h-4 w-4 sm:h-5 sm:w-5" />}
          </button>

          {/* Screen Share (Hidden on small mobile) */}
          <button
            onClick={onToggleScreenShare}
            title={isScreenSharing ? 'Stop presenting' : 'Present now'}
            className={`hidden sm:flex h-10 w-10 sm:h-11 sm:w-11 items-center justify-center rounded-full transition-all shrink-0 cursor-pointer ${
              isScreenSharing
                ? 'bg-blue-500 text-white shadow-md shadow-blue-500/25'
                : 'bg-[#2b2d35] hover:bg-[#343740] text-white shadow-xs'
            }`}
          >
            <ScreenShare className="h-4 w-4 sm:h-5 sm:w-5" />
          </button>

          {/* Security Monitor */}
          {onToggleSecurityPanel && (
            <button
              onClick={onToggleSecurityPanel}
              title="Security Monitor & Visualizer"
              className={`flex h-10 w-10 sm:h-11 sm:w-11 items-center justify-center rounded-full transition-all shrink-0 cursor-pointer ${
                isSecurityPanelOpen
                  ? 'bg-blue-500 text-white shadow-md shadow-blue-500/25'
                  : 'bg-[#2b2d35] hover:bg-[#343740] text-white shadow-xs'
              }`}
            >
              <Activity className="h-4 w-4 sm:h-5 sm:w-5" />
            </button>
          )}

          {/* Tester: Integrated AI Face Swap & Attack Simulation Button */}
          {isTester && onToggleTesterDrawer && (
            <button
              onClick={onToggleTesterDrawer}
              title="AI Face Swap & Attack Simulator (Tester Only)"
              className={`flex h-10 w-10 sm:h-11 sm:w-11 items-center justify-center rounded-full transition-all shrink-0 cursor-pointer relative ${
                isAttackActive || faceSwapActive
                  ? 'bg-[#ea3812] text-white shadow-[0_0_16px_rgba(234,56,18,0.35)]'
                  : isTesterDrawerOpen
                  ? 'bg-blue-500 text-white'
                  : 'bg-[#2b2d35] hover:bg-[#343740] text-white shadow-xs'
              }`}
            >
              <ShieldAlert className="h-4 w-4 sm:h-5 sm:w-5" />
              {(isAttackActive || faceSwapActive) && (
                <span className="absolute top-1 right-1 h-2 w-2 rounded-full bg-white animate-ping" />
              )}
            </button>
          )}

          {/* End Call (Clicks to morph into the Slide to End Call bar!) */}
          <button
            onClick={() => setSlider(true)}
            title="Slide to end call"
            className="flex h-10 w-10 sm:h-11 sm:w-11 items-center justify-center rounded-full bg-[#ea3812] hover:bg-[#d6320e] text-white transition-all active:scale-95 shrink-0 cursor-pointer shadow-[0_0_18px_rgba(234,56,18,0.45)] hover:shadow-[0_0_24px_rgba(234,56,18,0.6)]"
          >
            <PhoneOff className="h-4 w-4 sm:h-5 sm:w-5" />
          </button>
        </div>
      ) : (
        /* State 2: Transformed Inline Slider (Image 1) */
        <div
          ref={trackRef}
          className="relative flex items-center w-[285px] sm:w-[325px] h-14 sm:h-15 p-1 rounded-full bg-[#121316]/95 backdrop-blur-xl border border-[#22242a] shadow-2xl mx-auto overflow-hidden animate-slider-fade-in"
          style={{ touchAction: 'none' }}
        >
          {/* Dynamic Fill Bar */}
          <div
            ref={fillRef}
            className="absolute left-0 top-0 bottom-0 bg-gradient-to-r from-[#ea3812]/15 via-[#ea3812]/30 to-[#ea3812]/55 border-r border-[#ea3812]/60 pointer-events-none rounded-full"
            style={{ width: '12%' }}
          />

          {/* Guide Text inside Track */}
          <div
            ref={textRef}
            className="absolute inset-0 flex items-center justify-center pointer-events-none pl-11 pr-8 select-none"
          >
            <span className="flex items-center gap-1.5 text-[11px] sm:text-xs font-bold tracking-widest uppercase text-[#7d8290]">
              <span>SLIDE TO END CALL</span>
              <ChevronsRight className="h-3.5 w-3.5 text-[#ea3812] shrink-0" />
            </span>
          </div>

          {/* Completed State Notice */}
          {isCompleted && (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none animate-fade-in pl-10 pr-8">
              <span className="text-xs sm:text-sm font-bold tracking-wide uppercase text-[#ea3812]">
                Call Ended
              </span>
            </div>
          )}

          {/* Draggable Knob (The Red Circular Button from Image 2!) */}
          <div
            ref={knobRef}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerCancel={handlePointerUp}
            className={`relative z-10 h-10 w-10 sm:h-11 sm:w-11 rounded-full flex items-center justify-center select-none shadow-lg cursor-grab active:cursor-grabbing shrink-0 will-change-transform ${
              isCompleted
                ? 'bg-gradient-to-br from-emerald-500 to-emerald-600 text-white shadow-[0_0_20px_rgba(16,185,129,0.5)]'
                : 'bg-[#ea3812] hover:bg-[#d6320e] text-white shadow-[0_0_20px_rgba(234,56,18,0.55)] hover:shadow-[0_0_26px_rgba(234,56,18,0.7)]'
            }`}
            style={{
              transform: 'translate3d(0px, 0px, 0px)',
            }}
          >
            {isCompleted ? (
              <Check className="h-4 w-4 sm:h-5 sm:w-5 text-white animate-checkmark-pop stroke-[2.5]" />
            ) : (
              <PhoneOff className="h-4 w-4 sm:h-5 sm:w-5 text-white stroke-[2.2]" />
            )}
          </div>

          {/* Cancel Button on Right (Reverts smoothly to State 1) */}
          <button
            onClick={() => setSlider(false)}
            disabled={isCompleted}
            title="Cancel (Esc)"
            className="absolute right-2.5 z-10 p-1.5 rounded-full text-[#7d8290] hover:text-white hover:bg-white/10 transition-colors disabled:opacity-30 cursor-pointer"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      )}
    </div>
  );
};
