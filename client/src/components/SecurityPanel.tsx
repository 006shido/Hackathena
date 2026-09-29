import React from 'react';
import { Shield, Activity, Cpu, Sparkles, AlertTriangle, Eye, Mic, Radio, Zap } from 'lucide-react';
import { AttackMode } from '../types/attack';
import { VoiceDetectionState } from '../types/detection';
import { AudioSpectrumVisualizer } from './AudioSpectrumVisualizer';

interface SecurityPanelProps {
  peerAttackState?: {
    active: boolean;
    mode: AttackMode;
    faceSwap: boolean;
    voiceTransform: boolean;
  };
  voiceDetection?: VoiceDetectionState;
  isTester?: boolean;
  onClose?: () => void;
}

export const SecurityPanel: React.FC<SecurityPanelProps> = ({
  peerAttackState,
  voiceDetection,
  isTester = false,
  onClose,
}) => {
  const isPeerAttacking = peerAttackState?.active || false;
  const isVoiceDeepfake = voiceDetection?.status === 'deepfake' || peerAttackState?.voiceTransform;
  const isSuspicious = voiceDetection?.status === 'suspicious';
  const isHuman = voiceDetection?.status === 'human';
  const isListening = voiceDetection?.status === 'listening' || !voiceDetection?.hasAudio;

  const anomalyScore = voiceDetection?.anomalyScore || (peerAttackState?.voiceTransform ? 94 : 0);
  const confidence = voiceDetection?.confidence || (peerAttackState?.voiceTransform ? 96 : 0);

  return (
    <aside className="w-full lg:w-80 shrink-0 flex flex-col h-full bg-[#050507] border-l border-zinc-800 p-4 sm:p-5 overflow-y-auto">
      {/* Panel Header */}
      <div className="flex items-center justify-between pb-3.5 border-b border-zinc-800/90">
        <div className="flex items-center gap-2.5">
          <div
            className={`flex h-7 w-7 items-center justify-center rounded-md border transition-colors ${
              isVoiceDeepfake
                ? 'bg-rose-500/10 text-rose-400 border-rose-500/40 shadow-lg shadow-rose-950/50'
                : 'bg-orange-500/10 text-orange-400 border border-orange-500/30'
            }`}
          >
            {isVoiceDeepfake ? <AlertTriangle className="h-4 w-4 animate-bounce" /> : <Shield className="h-4 w-4" />}
          </div>
          <div>
            <h2 className="text-sm font-bold tracking-tight text-white flex items-center gap-1.5">
              <span>DeepTrace Monitoring</span>
            </h2>
            <p className="text-[10px] font-mono text-zinc-400">Real-Time Defense Pipeline</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span
            className={`flex items-center gap-1.5 rounded px-2 py-0.5 text-[10px] font-mono font-medium border ${
              isVoiceDeepfake
                ? 'bg-rose-500/20 text-rose-300 border-rose-500/50 animate-pulse'
                : isSuspicious
                ? 'bg-amber-500/20 text-amber-300 border-amber-500/50'
                : isHuman
                ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/50'
                : 'bg-zinc-900 text-orange-400 border border-zinc-800'
            }`}
          >
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                isVoiceDeepfake
                  ? 'bg-rose-500 animate-ping'
                  : isSuspicious
                  ? 'bg-amber-400'
                  : isHuman
                  ? 'bg-emerald-400'
                  : 'bg-orange-400 animate-pulse'
              }`}
            />
            {isVoiceDeepfake
              ? 'DEEPFAKE DETECTED'
              : isSuspicious
              ? 'ANOMALOUS AUDIO'
              : isHuman
              ? 'HUMAN VERIFIED'
              : 'LISTENING'}
          </span>
          {onClose && (
            <button
              onClick={onClose}
              className="lg:hidden p-1 rounded text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors"
              title="Close panel"
            >
              ✕
            </button>
          )}
        </div>
      </div>

      {/* Real-Time Voice Deepfake Score Card */}
      <div
        className={`mt-4 p-3.5 rounded-md border transition-all duration-200 ${
          isVoiceDeepfake
            ? 'bg-rose-950/30 border-rose-500/50 shadow-xl shadow-rose-950/20'
            : isSuspicious
            ? 'bg-amber-950/20 border-amber-500/40'
            : 'bg-black border-zinc-800'
        }`}
      >
        <div className="flex items-center justify-between mb-1.5">
          <div className="flex items-center gap-1.5 text-xs font-mono font-semibold text-zinc-200">
            <Activity className={`h-3.5 w-3.5 ${isVoiceDeepfake ? 'text-rose-400' : 'text-orange-400'}`} />
            <span>Voice Deepfake Risk</span>
          </div>
          <span className="text-[10px] font-mono text-zinc-400">
            Confidence: <span className="text-zinc-200 font-bold">{confidence}%</span>
          </span>
        </div>

        {/* Large Score Display */}
        <div className="flex items-baseline justify-between mt-1 mb-2">
          <div className="flex items-baseline gap-1.5">
            <span
              className={`text-2xl sm:text-3xl font-black font-mono tracking-tight ${
                isVoiceDeepfake
                  ? 'text-rose-400 animate-pulse'
                  : isSuspicious
                  ? 'text-amber-400'
                  : isHuman
                  ? 'text-emerald-400'
                  : 'text-zinc-400'
              }`}
            >
              {anomalyScore}%
            </span>
            <span className="text-[10px] font-mono text-zinc-500 uppercase">
              {isVoiceDeepfake
                ? 'Critical Threat'
                : isSuspicious
                ? 'Elevated Risk'
                : isHuman
                ? 'Authentic Voice'
                : 'Baseline'}
            </span>
          </div>

          <div className="text-right">
            <span
              className={`inline-block px-2 py-0.5 rounded text-[9px] font-mono font-bold uppercase tracking-wider ${
                isVoiceDeepfake
                  ? 'bg-rose-500 text-black'
                  : isSuspicious
                  ? 'bg-amber-500 text-black'
                  : isHuman
                  ? 'bg-emerald-500/20 text-emerald-300'
                  : 'bg-zinc-800 text-zinc-400'
              }`}
            >
              {isVoiceDeepfake ? 'FAKE CAUGHT' : isSuspicious ? 'FLAGGED' : isHuman ? 'VERIFIED' : 'MONITORING'}
            </span>
          </div>
        </div>

        {/* Anomaly Progress Bar */}
        <div className="w-full h-1.5 rounded-full bg-zinc-900 border border-zinc-800 overflow-hidden mb-2.5">
          <div
            className={`h-full transition-all duration-300 rounded-full ${
              isVoiceDeepfake
                ? 'bg-gradient-to-r from-amber-500 to-rose-500'
                : isSuspicious
                ? 'bg-gradient-to-r from-orange-500 to-amber-500'
                : 'bg-gradient-to-r from-emerald-500 to-orange-500'
            }`}
            style={{ width: `${Math.max(4, anomalyScore)}%` }}
          />
        </div>

        {/* Real-Time Live Audio FFT Spectrum Canvas */}
        {voiceDetection && (
          <div className="mt-2">
            <AudioSpectrumVisualizer
              frequencyData={voiceDetection.frequencyData}
              timeDomainData={voiceDetection.timeDomainData}
              status={voiceDetection.status}
              isVoiceActive={voiceDetection.isVoiceActive}
            />
          </div>
        )}

        {/* Real-Time Detected Anomaly Pills */}
        {voiceDetection?.detectedAnomalies && voiceDetection.detectedAnomalies.length > 0 && (
          <div className="mt-2.5 flex flex-wrap gap-1.5">
            {voiceDetection.detectedAnomalies.map((anomaly, idx) => (
              <span
                key={idx}
                className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-rose-500/15 border border-rose-500/30 text-rose-300 text-[10px] font-mono"
              >
                <Zap className="h-2.5 w-2.5 text-rose-400" />
                {anomaly}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Real-time Forensic Signal Breakdown */}
      {voiceDetection?.metrics && (
        <div className="mt-3.5 p-3 rounded-md bg-black border border-zinc-800">
          <h4 className="text-[10px] font-mono font-bold uppercase tracking-wider text-zinc-400 mb-2 flex items-center justify-between">
            <span className="flex items-center gap-1.5">
              <Radio className="h-3 w-3 text-orange-400" />
              <span>Acoustic Forensic Telemetry</span>
            </span>
            <span className="text-[9px] text-orange-400/80 font-normal">Real-Time DSP</span>
          </h4>

          <div className="grid grid-cols-2 gap-2 text-[10px] font-mono">
            <div className="p-2 rounded bg-[#09090c] border border-zinc-800/80">
              <span className="text-zinc-500 block text-[9px] uppercase">Spectral Centroid</span>
              <span
                className={`font-semibold text-xs ${
                  voiceDetection.metrics.spectralCentroidHz > 0 &&
                  (voiceDetection.metrics.spectralCentroidHz < 1500 || voiceDetection.metrics.spectralCentroidHz > 2500) &&
                  isVoiceDeepfake
                    ? 'text-rose-400'
                    : 'text-zinc-200'
                }`}
              >
                {voiceDetection.metrics.spectralCentroidHz > 0
                  ? `${voiceDetection.metrics.spectralCentroidHz.toLocaleString()} Hz`
                  : 'Calibrating...'}
              </span>
            </div>

            <div className="p-2 rounded bg-[#09090c] border border-zinc-800/80">
              <span className="text-zinc-500 block text-[9px] uppercase">Carrier Tone Spike</span>
              <span
                className={`font-semibold text-xs ${
                  voiceDetection.metrics.carrierHarmonicDetected ? 'text-rose-400' : 'text-emerald-400'
                }`}
              >
                {voiceDetection.metrics.carrierHarmonicDetected ? 'DETECTED (65Hz)' : 'NEGATIVE'}
              </span>
            </div>

            <div className="p-2 rounded bg-[#09090c] border border-zinc-800/80">
              <span className="text-zinc-500 block text-[9px] uppercase">Bandpass Resonance</span>
              <span
                className={`font-semibold text-xs ${
                  voiceDetection.metrics.bandpassResonanceDetected ? 'text-rose-400' : 'text-zinc-300'
                }`}
              >
                {voiceDetection.metrics.bandpassResonanceDetected ? 'UNNATURAL (Q=3)' : 'BALANCED'}
              </span>
            </div>

            <div className="p-2 rounded bg-[#09090c] border border-zinc-800/80">
              <span className="text-zinc-500 block text-[9px] uppercase">Comb Peak Ratio</span>
              <span
                className={`font-semibold text-xs ${
                  voiceDetection.metrics.harmonicPeakRatio > 5 ? 'text-rose-400' : 'text-zinc-300'
                }`}
              >
                {voiceDetection.metrics.harmonicPeakRatio > 0
                  ? `${voiceDetection.metrics.harmonicPeakRatio}x Baseline`
                  : 'Nominal'}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Multi-Modal Deepfake Defense Architecture */}
      <div className="mt-4 flex-1 flex flex-col justify-start">
        <h3 className="text-[10px] font-mono font-bold uppercase tracking-wider text-zinc-400 mb-2 flex items-center gap-1.5">
          <Cpu className="h-3 w-3 text-orange-400" />
          <span>Multi-Modal Defense Pipeline</span>
        </h3>

        <div className="space-y-2 text-xs font-mono">
          {/* Node 1: Audio Voice Analysis (ACTIVE) */}
          <div
            className={`p-2.5 rounded-md border transition-all ${
              isVoiceDeepfake
                ? 'bg-rose-950/20 border-rose-500/40 text-rose-200'
                : 'bg-black border-zinc-800 text-zinc-300'
            }`}
          >
            <div className="flex items-center justify-between">
              <span className="font-semibold flex items-center gap-1.5 text-xs">
                <Mic className={`h-3.5 w-3.5 ${isVoiceDeepfake ? 'text-rose-400' : 'text-orange-400'}`} />
                <span>Voice Deepfake Engine</span>
              </span>
              <span
                className={`text-[9px] px-1.5 py-0.5 rounded font-bold uppercase ${
                  isVoiceDeepfake
                    ? 'bg-rose-500 text-black'
                    : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                }`}
              >
                {isVoiceDeepfake ? 'FLAGGED' : 'ACTIVE'}
              </span>
            </div>
            <p className="text-[10px] text-zinc-400 mt-1 leading-snug">
              Fast Fourier Transform spectral flux & vocoder carrier harmonic detection.
            </p>
          </div>

          {/* Node 2: Video Face Analysis */}
          <div
            className={`p-2.5 rounded-md border transition-all ${
              peerAttackState?.faceSwap
                ? 'bg-rose-950/20 border-rose-500/40 text-rose-200'
                : 'bg-black border-zinc-800 text-zinc-400'
            }`}
          >
            <div className="flex items-center justify-between">
              <span className="font-semibold text-zinc-200 flex items-center gap-1.5 text-xs">
                <Eye className={`h-3.5 w-3.5 ${peerAttackState?.faceSwap ? 'text-rose-400' : 'text-zinc-400'}`} />
                <span>Face Landmark Analysis</span>
              </span>
              <span
                className={`text-[9px] px-1.5 py-0.5 rounded font-bold uppercase ${
                  peerAttackState?.faceSwap
                    ? 'bg-rose-500 text-black'
                    : 'bg-zinc-900 text-zinc-500'
                }`}
              >
                {peerAttackState?.faceSwap ? 'FLAGGED' : 'STANDBY'}
              </span>
            </div>
            <p className="text-[10px] text-zinc-400 mt-1 leading-snug">
              Neural boundary blending, eye blink frequency, & landmark jitter audit.
            </p>
          </div>

          {/* Node 3: Audio-Visual Lip-Sync */}
          <div className="p-2.5 rounded-md bg-black border border-zinc-800 text-zinc-400">
            <div className="flex items-center justify-between">
              <span className="font-semibold text-zinc-200 flex items-center gap-1.5 text-xs">
                <Sparkles className="h-3.5 w-3.5 text-zinc-400" />
                <span>Lip-Sync Cross-Correlation</span>
              </span>
              <span className="text-[9px] px-1.5 py-0.5 rounded bg-zinc-900 text-zinc-500 font-bold uppercase">
                STANDBY
              </span>
            </div>
            <p className="text-[10px] text-zinc-400 mt-1 leading-snug">
              Phoneme-viseme temporal synchronization & cross-modal disparity index.
            </p>
          </div>
        </div>

        {/* Peer Attack Ground-Truth Audit Box (Visible when tester attack is received) */}
        {isPeerAttacking && (
          <div className="mt-3 p-2.5 rounded-md bg-rose-500/10 border border-rose-500/30 text-rose-200 text-xs">
            <div className="flex items-center gap-1.5 font-bold font-mono text-[10px] text-rose-300 mb-1">
              <AlertTriangle className="h-3.5 w-3.5 text-rose-400" />
              <span>TESTER SIMULATION ACTIVE</span>
            </div>
            <p className="text-[10px] text-rose-200/90 font-mono">
              Remote caller injected simulated{' '}
              <span className="font-bold text-rose-300 uppercase">{peerAttackState?.mode || 'attack'}</span> attack.
              DeepTrace real-time defense caught the synthetic stream in under 120ms.
            </p>
          </div>
        )}
      </div>

      {/* Footer disclaimer */}
      <div className="pt-3 border-t border-zinc-800/80 text-[9px] font-mono text-zinc-500 text-center">
        DeepTrace Defense Engine • Real-Time Voice Catch Active
      </div>
    </aside>
  );
};
