import React, { useEffect, useRef } from 'react';
import { DetectionStatus } from '../types/detection';

interface AudioSpectrumVisualizerProps {
  frequencyData: Uint8Array;
  timeDomainData?: Uint8Array;
  status: DetectionStatus;
  isVoiceActive: boolean;
  className?: string;
}

export const AudioSpectrumVisualizer: React.FC<AudioSpectrumVisualizerProps> = ({
  frequencyData,
  timeDomainData,
  status,
  isVoiceActive,
  className = '',
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    // Background subtle grid lines
    ctx.strokeStyle = 'rgba(30, 41, 59, 0.4)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.moveTo(0, height * 0.75);
    ctx.lineTo(width, height * 0.75);
    ctx.stroke();

    const isDeepfake = status === 'deepfake';
    const isSuspicious = status === 'suspicious';

    // 1. Draw Frequency Spectrum Bars (0 to ~5 kHz range, first 120 bins)
    const binsToDraw = 48;
    const barWidth = width / binsToDraw;

    for (let i = 0; i < binsToDraw; i++) {
      const val = frequencyData[i] || 0;
      const barHeight = isVoiceActive ? (val / 255) * (height - 6) : 2;

      let barColor: string;
      if (isDeepfake) {
        barColor = i < 4 || (i >= 12 && i <= 24) ? '#f43f5e' : '#fb7185';
      } else if (isSuspicious) {
        barColor = '#f59e0b';
      } else if (isVoiceActive) {
        barColor = '#06b6d4';
      } else {
        barColor = '#334155';
      }

      ctx.fillStyle = barColor;
      // Draw rounded top bar
      const x = i * barWidth;
      const y = height - barHeight;
      ctx.fillRect(x + 1, y, Math.max(1, barWidth - 2), barHeight);
    }

    // 2. Draw Real-Time Oscilloscope Waveform overlay
    if (timeDomainData && isVoiceActive) {
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = isDeepfake
        ? 'rgba(244, 63, 94, 0.9)'
        : isSuspicious
        ? 'rgba(245, 158, 11, 0.8)'
        : 'rgba(34, 211, 238, 0.8)';
      ctx.beginPath();

      const sliceWidth = width / timeDomainData.length;
      let x = 0;

      for (let i = 0; i < timeDomainData.length; i++) {
        const v = timeDomainData[i] / 128.0;
        const y = (v * height) / 2;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
        x += sliceWidth;
      }
      ctx.stroke();
    }
  }, [frequencyData, timeDomainData, status, isVoiceActive]);

  return (
    <div className={`relative h-[68px] shrink-0 overflow-hidden rounded-md bg-black border border-zinc-800 ${className}`}>
      <canvas
        ref={canvasRef}
        width={280}
        height={68}
        className="w-full h-full block"
      />
      {/* Overlay label */}
      <div className="absolute top-1.5 left-2 flex items-center gap-1.5 pointer-events-none">
        <span
          className={`h-1.5 w-1.5 rounded-full ${
            status === 'deepfake'
              ? 'bg-rose-500 animate-ping'
              : status === 'suspicious'
              ? 'bg-amber-400'
              : isVoiceActive
              ? 'bg-cyan-400 animate-pulse'
              : 'bg-slate-600'
          }`}
        />
        <span className="text-[9px] font-mono uppercase tracking-wider text-slate-400">
          {status === 'deepfake'
            ? 'Spectral Comb Anomaly'
            : isVoiceActive
            ? 'Incoming Audio FFT'
            : 'Listening for Audio'}
        </span>
      </div>
    </div>
  );
};
