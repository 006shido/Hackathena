import React from 'react';
import { Shield, Activity, AlertTriangle, Eye, Mic, Radio } from 'lucide-react';
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
    <aside className="w-full lg:w-80 shrink-0 flex flex-col h-full bg-[#28292c] border-l border-[#3c4043] p-4 overflow-y-auto overflow-x-hidden select-none">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-[#3c4043] shrink-0">
        <div className="flex items-center gap-2">
          <div className={`flex h-8 w-8 items-center justify-center rounded-lg shrink-0 ${
            isVoiceDeepfake
              ? 'bg-[#ea4335]/15 text-[#ea4335]'
              : 'bg-[#8ab4f8]/15 text-[#8ab4f8]'
          }`}>
            {isVoiceDeepfake ? <AlertTriangle className="h-4 w-4" /> : <Shield className="h-4 w-4" />}
          </div>
          <div>
            <h2 className="text-sm font-medium text-[#e8eaed]">Security Monitor</h2>
            <p className="text-[11px] text-[#9aa0a6]">Real-time analysis</p>
          </div>
        </div>

        <div className="flex items-center gap-1.5 shrink-0">
          <span className={`flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-medium ${
            isVoiceDeepfake
              ? 'bg-[#ea4335]/15 text-[#ea4335]'
              : isSuspicious
              ? 'bg-[#fbbc04]/15 text-[#fbbc04]'
              : isHuman
              ? 'bg-[#34a853]/15 text-[#34a853]'
              : 'bg-[#3c4043] text-[#9aa0a6]'
          }`}>
            <span className={`h-1.5 w-1.5 rounded-full ${
              isVoiceDeepfake
                ? 'bg-[#ea4335] animate-pulse'
                : isSuspicious
                ? 'bg-[#fbbc04]'
                : isHuman
                ? 'bg-[#34a853]'
                : 'bg-[#8ab4f8] animate-pulse'
            }`} />
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
              className="lg:hidden p-1 rounded-full text-[#9aa0a6] hover:text-[#e8eaed] hover:bg-[#3c4043] transition-colors"
              title="Close"
            >
              ✕
            </button>
          )}
        </div>
      </div>

      {/* Voice Risk Score */}
      <div className={`mt-3 p-3.5 rounded-2xl border transition-all shrink-0 ${
        isVoiceDeepfake
          ? 'bg-[#ea4335]/10 border-[#ea4335]/30'
          : isSuspicious
          ? 'bg-[#fbbc04]/10 border-[#fbbc04]/30'
          : 'bg-[#303134] border-[#3c4043]'
      }`}>
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-1.5 text-sm text-[#e8eaed] font-medium">
            <Activity className={`h-4 w-4 ${isVoiceDeepfake ? 'text-[#ea4335]' : 'text-[#8ab4f8]'}`} />
            <span>Voice Risk Score</span>
          </div>
          <span className="text-xs text-[#9aa0a6]">{confidence}% confidence</span>
        </div>

        <div className="flex items-baseline gap-2 mb-2">
          <span className={`text-3xl font-semibold ${
            isVoiceDeepfake
              ? 'text-[#ea4335]'
              : isSuspicious
              ? 'text-[#fbbc04]'
              : isHuman
              ? 'text-[#34a853]'
              : 'text-[#9aa0a6]'
          }`}>
            {anomalyScore}%
          </span>
          <span className="text-xs text-[#9aa0a6]">
            {isVoiceDeepfake ? 'Critical' : isSuspicious ? 'Elevated' : isHuman ? 'Normal' : 'Baseline'}
          </span>
        </div>

        <div className="w-full h-1.5 rounded-full bg-[#202124] overflow-hidden mb-3">
          <div
            className={`h-full transition-all duration-300 rounded-full ${
              isVoiceDeepfake ? 'bg-[#ea4335]' : isSuspicious ? 'bg-[#fbbc04]' : 'bg-[#34a853]'
            }`}
            style={{ width: `${Math.max(4, anomalyScore)}%` }}
          />
        </div>

        {voiceDetection && (
          <AudioSpectrumVisualizer
            frequencyData={voiceDetection.frequencyData}
            timeDomainData={voiceDetection.timeDomainData}
            status={voiceDetection.status}
            isVoiceActive={voiceDetection.isVoiceActive}
          />
        )}

        <div className="mt-2 h-6 flex items-center overflow-hidden">
          {voiceDetection?.detectedAnomalies && voiceDetection.detectedAnomalies.length > 0 ? (
            <div className="flex items-center gap-1 overflow-hidden w-full">
              {voiceDetection.detectedAnomalies.slice(0, 2).map((anomaly, idx) => (
                <span
                  key={idx}
                  className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-[#ea4335]/15 text-[#ea4335] text-[10px] truncate"
                  title={anomaly}
                >
                  <span className="truncate">{anomaly}</span>
                </span>
              ))}
            </div>
          ) : (
            <span className="text-[11px] text-[#9aa0a6]">
              {isHuman ? 'No anomalies detected' : 'Listening for audio...'}
            </span>
          )}
        </div>
      </div>

      {/* Signal Metrics */}
      {voiceDetection?.metrics && (
        <div className="mt-3 p-3.5 rounded-2xl bg-[#303134] border border-[#3c4043] shrink-0">
          <h4 className="text-xs font-medium text-[#9aa0a6] mb-2 flex items-center gap-1.5">
            <Radio className="h-3 w-3 text-[#8ab4f8]" />
            Signal Analysis
          </h4>

          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="p-2 rounded-xl bg-[#202124]">
              <span className="text-[#9aa0a6] block text-[10px]">Spectral Centroid</span>
              <span className={`font-medium ${
                voiceDetection.metrics.spectralCentroidHz > 0 &&
                (voiceDetection.metrics.spectralCentroidHz < 1500 || voiceDetection.metrics.spectralCentroidHz > 2500) &&
                isVoiceDeepfake ? 'text-[#ea4335]' : 'text-[#e8eaed]'
              }`}>
                {voiceDetection.metrics.spectralCentroidHz > 0
                  ? `${voiceDetection.metrics.spectralCentroidHz.toLocaleString()} Hz`
                  : 'Calibrating...'}
              </span>
            </div>

            <div className="p-2 rounded-xl bg-[#202124]">
              <span className="text-[#9aa0a6] block text-[10px]">Carrier Tone</span>
              <span className={`font-medium ${
                voiceDetection.metrics.carrierHarmonicDetected ? 'text-[#ea4335]' : 'text-[#34a853]'
              }`}>
                {voiceDetection.metrics.carrierHarmonicDetected ? 'Detected' : 'Clean'}
              </span>
            </div>

            <div className="p-2 rounded-xl bg-[#202124]">
              <span className="text-[#9aa0a6] block text-[10px]">Resonance</span>
              <span className={`font-medium ${
                voiceDetection.metrics.bandpassResonanceDetected ? 'text-[#ea4335]' : 'text-[#e8eaed]'
              }`}>
                {voiceDetection.metrics.bandpassResonanceDetected ? 'Unnatural' : 'Normal'}
              </span>
            </div>

            <div className="p-2 rounded-xl bg-[#202124]">
              <span className="text-[#9aa0a6] block text-[10px]">Peak Ratio</span>
              <span className={`font-medium ${
                voiceDetection.metrics.harmonicPeakRatio > 5 ? 'text-[#ea4335]' : 'text-[#e8eaed]'
              }`}>
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
        <h3 className="text-xs font-medium text-[#9aa0a6] mb-2 flex items-center gap-1.5">
          <Shield className="h-3 w-3 text-[#8ab4f8]" />
          Defense Modules
        </h3>

        <div className="space-y-1.5">
          <div className={`p-2.5 rounded-xl border transition-all ${
            isVoiceDeepfake
              ? 'bg-[#ea4335]/10 border-[#ea4335]/30'
              : 'bg-[#303134] border-[#3c4043]'
          }`}>
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-1.5 text-xs font-medium text-[#e8eaed]">
                <Mic className={`h-3.5 w-3.5 ${isVoiceDeepfake ? 'text-[#ea4335]' : 'text-[#8ab4f8]'}`} />
                Voice Analysis
              </span>
              <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${
                isVoiceDeepfake
                  ? 'bg-[#ea4335] text-white'
                  : 'bg-[#34a853]/15 text-[#34a853]'
              }`}>
                {isVoiceDeepfake ? 'Alert' : 'Active'}
              </span>
            </div>
          </div>

          <div className={`p-2.5 rounded-xl border transition-all ${
            peerAttackState?.faceSwap
              ? 'bg-[#ea4335]/10 border-[#ea4335]/30'
              : 'bg-[#303134] border-[#3c4043]'
          }`}>
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-1.5 text-xs font-medium text-[#e8eaed]">
                <Eye className={`h-3.5 w-3.5 ${peerAttackState?.faceSwap ? 'text-[#ea4335]' : 'text-[#9aa0a6]'}`} />
                Face Analysis
              </span>
              <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${
                peerAttackState?.faceSwap
                  ? 'bg-[#ea4335] text-white'
                  : 'bg-[#3c4043] text-[#9aa0a6]'
              }`}>
                {peerAttackState?.faceSwap ? 'Alert' : 'Standby'}
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="pt-3 mt-2 border-t border-[#3c4043] text-[10px] text-[#9aa0a6] text-center shrink-0">
        DeepTrace Security • Real-time Protection
      </div>
    </aside>
  );
};
