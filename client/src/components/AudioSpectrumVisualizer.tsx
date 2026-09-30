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
        barColor = '#ea4335';
      } else if (isSuspicious) {
        barColor = '#fbbc04';
      } else if (isVoiceActive) {
        barColor = '#34a853';
      } else {
        barColor = '#3c4043';
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
        ? 'rgba(234, 67, 53, 0.9)'
        : isSuspicious
        ? 'rgba(251, 188, 4, 0.8)'
        : 'rgba(52, 168, 83, 0.8)';
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
    <div className={`relative h-[68px] shrink-0 overflow-hidden rounded-xl bg-[#202124] border border-[#3c4043] ${className}`}>
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
              ? 'bg-[#ea4335] animate-ping'
              : status === 'suspicious'
              ? 'bg-[#fbbc04]'
              : isVoiceActive
              ? 'bg-[#34a853] animate-pulse'
              : 'bg-[#5f6368]'
          }`}
        />
        <span className="text-[10px] text-[#9aa0a6] font-medium">
          {status === 'deepfake'
            ? 'Voice Anomaly Detected'
            : isVoiceActive
            ? 'Incoming Audio Spectrum'
            : 'Listening for Audio'}
        </span>
      </div>
    </div>
  );
};
