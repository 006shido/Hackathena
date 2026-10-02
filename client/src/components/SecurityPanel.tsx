import React from 'react';
import { Shield, Activity, AlertTriangle, Eye, Mic, Radio, X } from 'lucide-react';
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
  const isVoiceDeepfake = voiceDetection?.status === 'deepfake';
  const isSuspicious = voiceDetection?.status === 'suspicious';
  const isHuman = voiceDetection?.status === 'human';

  const anomalyScore = voiceDetection?.anomalyScore || 0;
  const confidence = voiceDetection?.confidence || 0;

  return (
    <aside className="w-full lg:w-84 shrink-0 flex flex-col h-full bg-[#16181f] border-l border-white/10 p-4 overflow-y-auto overflow-x-hidden select-none text-slate-100 font-sans">
      {/* Header */}
      <div className="flex items-center justify-between pb-3.5 border-b border-white/10 shrink-0">
        <div className="flex items-center gap-2.5">
          <div
            className={`flex h-8 w-8 items-center justify-center rounded-xl shrink-0 ${
              isVoiceDeepfake
                ? 'bg-red-500/20 text-red-400'
                : 'bg-blue-500/20 text-blue-400'
            }`}
          >
            {isVoiceDeepfake ? <AlertTriangle className="h-4 w-4" /> : <Shield className="h-4 w-4" />}
          </div>
          <div>
            <h2 className="text-sm font-bold text-white tracking-tight">Security Monitor</h2>
            <p className="text-[11px] text-slate-400">Autonomous deepfake analysis</p>
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <span
            className={`flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-semibold ${
              isVoiceDeepfake
                ? 'bg-red-500/20 text-red-400 border border-red-500/30'
                : isSuspicious
                ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                : isHuman
                ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                : 'bg-white/5 text-slate-400 border border-white/10'
            }`}
          >
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                isVoiceDeepfake
                  ? 'bg-red-400 animate-pulse'
                  : isSuspicious
                  ? 'bg-amber-400'
                  : isHuman
                  ? 'bg-emerald-400'
                  : 'bg-blue-400 animate-pulse'
              }`}
            />
            {isVoiceDeepfake
              ? 'Deepfake'
              : isSuspicious
              ? 'Suspicious'
              : isHuman
              ? 'Verified'
              : 'Listening'}
          </span>

          {onClose && (
            <button
              onClick={onClose}
              className="p-1 rounded-full text-slate-400 hover:text-white hover:bg-white/10 transition-colors cursor-pointer"
              title="Close panel"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>
      </div>

      {/* Voice Risk Score Card */}
      <div
        className={`mt-3.5 p-4 rounded-2xl border transition-all shrink-0 ${
          isVoiceDeepfake
            ? 'bg-red-500/10 border-red-500/30'
            : isSuspicious
            ? 'bg-amber-500/10 border-amber-500/30'
            : 'bg-white/5 border-white/10'
        }`}
      >
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-1.5 text-xs text-slate-200 font-semibold">
            <Activity className={`h-4 w-4 ${isVoiceDeepfake ? 'text-red-400' : 'text-blue-400'}`} />
            <span>Voice Anomaly Index</span>
          </div>
          <span className="text-[11px] text-slate-400">{confidence}% confidence</span>
        </div>

        <div className="flex items-baseline gap-2 mb-2">
          <span
            className={`text-3xl font-extrabold tracking-tight ${
              isVoiceDeepfake
                ? 'text-red-400'
                : isSuspicious
                ? 'text-amber-400'
                : isHuman
                ? 'text-emerald-400'
                : 'text-slate-400'
            }`}
          >
            {anomalyScore}%
          </span>
          <span className="text-xs font-medium text-slate-400">
            {isVoiceDeepfake ? 'Critical Risk' : isSuspicious ? 'Suspicious' : isHuman ? 'Natural Human' : 'Baseline'}
          </span>
        </div>

        {/* Progress Bar */}
        <div className="w-full h-1.5 rounded-full bg-white/10 overflow-hidden mb-3">
          <div
            className={`h-full transition-all duration-300 rounded-full ${
              isVoiceDeepfake ? 'bg-red-500' : isSuspicious ? 'bg-amber-500' : 'bg-emerald-500'
            }`}
            style={{ width: `${Math.max(4, anomalyScore)}%` }}
          />
        </div>

        {/* Audio Spectrum Visualizer */}
        {voiceDetection && (
          <AudioSpectrumVisualizer
            frequencyData={voiceDetection.frequencyData}
            timeDomainData={voiceDetection.timeDomainData}
            status={voiceDetection.status}
            isVoiceActive={voiceDetection.isVoiceActive}
          />
        )}

        {/* Anomalies Detected Chip */}
        <div className="mt-2.5 min-h-[24px] flex items-center">
          {voiceDetection?.detectedAnomalies && voiceDetection.detectedAnomalies.length > 0 ? (
            <div className="flex items-center gap-1 flex-wrap w-full">
              {voiceDetection.detectedAnomalies.slice(0, 2).map((anomaly, idx) => (
                <span
                  key={idx}
                  className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-red-500/20 text-red-300 text-[10px] font-medium truncate"
                  title={anomaly}
                >
                  <span className="truncate">{anomaly}</span>
                </span>
              ))}
            </div>
          ) : (
            <span className="text-[11px] text-slate-400">
              {isHuman ? 'Natural acoustic harmonics verified' : 'Sampling incoming audio stream...'}
            </span>
          )}
        </div>
      </div>

      {/* Signal Metrics */}
      {voiceDetection?.metrics && (
        <div className="mt-3 p-3.5 rounded-2xl bg-white/5 border border-white/10 shrink-0">
          <h4 className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-2.5 flex items-center gap-1.5">
            <Radio className="h-3 w-3 text-blue-400" />
            <span>Spectral Diagnostics</span>
          </h4>

          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="p-2.5 rounded-xl bg-black/30 border border-white/5">
              <span className="text-slate-400 block text-[10px]">Spectral Centroid</span>
              <span
                className={`font-semibold mt-0.5 block ${
                  voiceDetection.metrics.spectralCentroidHz > 0 &&
                  (voiceDetection.metrics.spectralCentroidHz < 1500 ||
                    voiceDetection.metrics.spectralCentroidHz > 2500) &&
                  isVoiceDeepfake
                    ? 'text-red-400'
                    : 'text-slate-200'
                }`}
              >
                {voiceDetection.metrics.spectralCentroidHz > 0
                  ? `${voiceDetection.metrics.spectralCentroidHz.toLocaleString()} Hz`
                  : 'Calibrating...'}
              </span>
            </div>

            <div className="p-2.5 rounded-xl bg-black/30 border border-white/5">
              <span className="text-slate-400 block text-[10px]">Carrier Tone</span>
              <span
                className={`font-semibold mt-0.5 block ${
                  voiceDetection.metrics.carrierHarmonicDetected ? 'text-red-400' : 'text-emerald-400'
                }`}
              >
                {voiceDetection.metrics.carrierHarmonicDetected ? 'Detected' : 'Clean'}
              </span>
            </div>

            <div className="p-2.5 rounded-xl bg-black/30 border border-white/5">
              <span className="text-slate-400 block text-[10px]">Bandpass Resonance</span>
              <span
                className={`font-semibold mt-0.5 block ${
                  voiceDetection.metrics.bandpassResonanceDetected ? 'text-red-400' : 'text-slate-200'
                }`}
              >
                {voiceDetection.metrics.bandpassResonanceDetected ? 'Synthetic' : 'Natural'}
              </span>
            </div>

            <div className="p-2.5 rounded-xl bg-black/30 border border-white/5">
              <span className="text-slate-400 block text-[10px]">Harmonic Peak Ratio</span>
              <span
                className={`font-semibold mt-0.5 block ${
                  voiceDetection.metrics.harmonicPeakRatio > 5 ? 'text-red-400' : 'text-slate-200'
                }`}
              >
                {voiceDetection.metrics.harmonicPeakRatio > 0
                  ? `${voiceDetection.metrics.harmonicPeakRatio}x`
                  : 'Nominal'}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Defense Modules */}
      <div className="mt-3 flex-1 flex flex-col justify-start shrink-0">
        <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-2 flex items-center gap-1.5">
          <Shield className="h-3 w-3 text-blue-400" />
          <span>Active Defenses</span>
        </h3>

        <div className="space-y-2">
          <div
            className={`p-3 rounded-xl border transition-all ${
              isVoiceDeepfake
                ? 'bg-red-500/10 border-red-500/30'
                : 'bg-white/5 border-white/10'
            }`}
          >
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-2 text-xs font-semibold text-slate-200">
                <Mic className={`h-3.5 w-3.5 ${isVoiceDeepfake ? 'text-red-400' : 'text-blue-400'}`} />
                <span>Voice Verification</span>
              </span>
              <span
                className={`text-[10px] px-2 py-0.5 rounded-full font-semibold ${
                  isVoiceDeepfake
                    ? 'bg-red-500 text-white'
                    : 'bg-emerald-500/20 text-emerald-400'
                }`}
              >
                {isVoiceDeepfake ? 'Alert' : 'Active'}
              </span>
            </div>
          </div>

          <div
            className={`p-3 rounded-xl border transition-all ${
              peerAttackState?.faceSwap
                ? 'bg-red-500/10 border-red-500/30'
                : 'bg-white/5 border-white/10'
            }`}
          >
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-2 text-xs font-semibold text-slate-200">
                <Eye className={`h-3.5 w-3.5 ${peerAttackState?.faceSwap ? 'text-red-400' : 'text-slate-400'}`} />
                <span>Face Landmark Analysis</span>
              </span>
              <span
                className={`text-[10px] px-2 py-0.5 rounded-full font-semibold ${
                  peerAttackState?.faceSwap
                    ? 'bg-red-500 text-white'
                    : 'bg-white/10 text-slate-400'
                }`}
              >
                {peerAttackState?.faceSwap ? 'Alert' : 'Standby'}
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="pt-3 mt-3 border-t border-white/10 text-[10px] text-slate-400 text-center shrink-0">
        DeepTrace Defense Engine • Real-time Protection
      </div>
    </aside>
  );
};
