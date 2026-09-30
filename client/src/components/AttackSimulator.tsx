import React from 'react';
import {
  ShieldAlert,
  Flame,
  RotateCcw,
  Sparkles,
  Mic,
  Eye,
  Sliders,
  AlertTriangle,
  X,
} from 'lucide-react';
import { AttackState } from '../types/attack';

interface AttackSimulatorProps {
  attackState: AttackState;
  onToggleFaceSwap: () => void;
  onToggleVoiceTransform: () => void;
  onActivateCombined: () => void;
  onReset: () => void;
  onSelectFacePreset: (preset: 'neural-clone' | 'biometric-mask' | 'synthetic-executive') => void;
  onSelectVoicePreset: (preset: 'robotic-vocoder' | 'deep-pitch-neural' | 'synthetic-clone') => void;
  onClose?: () => void;
}

export const AttackSimulator: React.FC<AttackSimulatorProps> = ({
  attackState,
  onToggleFaceSwap,
  onToggleVoiceTransform,
  onActivateCombined,
  onReset,
  onSelectFacePreset,
  onSelectVoicePreset,
  onClose,
}) => {
  const { faceSwap, voiceTransform, mode, facePreset, voicePreset } = attackState;

  let statusText = 'Standby';
  let statusStyle = 'bg-[#303134] text-[#9aa0a6] border-[#3c4043]';

  if (mode === 'combined') {
    statusText = 'Full Attack Active';
    statusStyle = 'bg-[#ea4335]/10 text-[#ea4335] border-[#ea4335]/40';
  } else if (mode === 'face') {
    statusText = 'Face Swap Active';
    statusStyle = 'bg-[#8ab4f8]/10 text-[#8ab4f8] border-[#8ab4f8]/40';
  } else if (mode === 'voice') {
    statusText = 'Voice Transform Active';
    statusStyle = 'bg-[#8ab4f8]/10 text-[#8ab4f8] border-[#8ab4f8]/40';
  }

  return (
    <div className="w-full lg:w-88 shrink-0 flex flex-col h-full bg-[#28292c] border-l border-[#3c4043] p-4 overflow-y-auto shadow-2xl">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-[#3c4043]">
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#8ab4f8]/15 text-[#8ab4f8]">
            <ShieldAlert className="h-4 w-4" />
          </div>
          <div>
            <h2 className="text-sm font-medium text-[#e8eaed]">Attack Simulator</h2>
            <p className="text-[11px] text-[#9aa0a6]">Tester tools</p>
          </div>
        </div>

        {onClose && (
          <button
            onClick={onClose}
            className="p-1.5 rounded-full text-[#9aa0a6] hover:text-[#e8eaed] hover:bg-[#3c4043] transition-colors cursor-pointer"
          >
            <X className="h-4 w-4" />
          </button>
        )}
      </div>

      {/* Status */}
      <div className={`mt-3 p-2.5 rounded-xl border text-center text-sm font-medium transition-all ${statusStyle}`}>
        {mode !== 'none' && <Flame className="h-4 w-4 inline mr-1.5" />}
        {statusText}
      </div>

      {/* Notice */}
      <div className="mt-3 flex items-center gap-2 p-2.5 rounded-xl bg-[#fbbc04]/10 border border-[#fbbc04]/20 text-xs text-[#fbbc04]">
        <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
        <span>Injected streams replace outgoing tracks in real time.</span>
      </div>

      {/* Face Simulation */}
      <div className="mt-4 p-3.5 rounded-2xl bg-[#303134] border border-[#3c4043]">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Eye className="h-4 w-4 text-[#8ab4f8]" />
            <h3 className="text-xs font-medium text-[#e8eaed]">Face Simulation</h3>
          </div>
          <span className={`text-[10px] font-medium px-2 py-0.5 rounded-full ${
            faceSwap ? 'bg-[#8ab4f8]/15 text-[#8ab4f8]' : 'bg-[#202124] text-[#9aa0a6]'
          }`}>
            {faceSwap ? 'Active' : 'Off'}
          </span>
        </div>

        <button
          onClick={onToggleFaceSwap}
          className={`w-full py-2 px-3 rounded-full text-sm font-medium transition-colors border cursor-pointer ${
            faceSwap
              ? 'bg-[#8ab4f8]/15 text-[#8ab4f8] border-[#8ab4f8]/40 hover:bg-[#8ab4f8]/25'
              : 'bg-[#202124] text-[#e8eaed] border-[#3c4043] hover:bg-[#3c4043]'
          }`}
        >
          <Sparkles className="h-3.5 w-3.5 inline mr-1.5" />
          {faceSwap ? 'Disable Face Swap' : 'Enable Face Swap'}
        </button>

        <div className="mt-3 space-y-1">
          <label className="text-[11px] text-[#9aa0a6] block mb-1.5">Face Model:</label>
          {[
            { id: 'neural-clone', label: 'Synthetic Persona' },
            { id: 'synthetic-executive', label: 'Corporate Executive' },
            { id: 'biometric-mask', label: 'Biometric Mesh' },
          ].map((p) => (
            <button
              key={p.id}
              onClick={() => onSelectFacePreset(p.id as any)}
              className={`w-full py-1.5 px-2.5 rounded-lg text-left text-xs transition-colors cursor-pointer ${
                facePreset === p.id
                  ? 'bg-[#8ab4f8]/15 text-[#8ab4f8] font-medium'
                  : 'text-[#9aa0a6] hover:bg-[#202124] hover:text-[#e8eaed]'
              }`}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      {/* Voice Simulation */}
      <div className="mt-3 p-3.5 rounded-2xl bg-[#303134] border border-[#3c4043]">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Mic className="h-4 w-4 text-[#8ab4f8]" />
            <h3 className="text-xs font-medium text-[#e8eaed]">Voice Simulation</h3>
          </div>
          <span className={`text-[10px] font-medium px-2 py-0.5 rounded-full ${
            voiceTransform ? 'bg-[#8ab4f8]/15 text-[#8ab4f8]' : 'bg-[#202124] text-[#9aa0a6]'
          }`}>
            {voiceTransform ? 'Active' : 'Off'}
          </span>
        </div>

        <button
          onClick={onToggleVoiceTransform}
          className={`w-full py-2 px-3 rounded-full text-sm font-medium transition-colors border cursor-pointer ${
            voiceTransform
              ? 'bg-[#8ab4f8]/15 text-[#8ab4f8] border-[#8ab4f8]/40 hover:bg-[#8ab4f8]/25'
              : 'bg-[#202124] text-[#e8eaed] border-[#3c4043] hover:bg-[#3c4043]'
          }`}
        >
          <Sliders className="h-3.5 w-3.5 inline mr-1.5" />
          {voiceTransform ? 'Disable Voice Transform' : 'Enable Voice Transform'}
        </button>

        <div className="mt-3 space-y-1">
          <label className="text-[11px] text-[#9aa0a6] block mb-1.5">Voice Model:</label>
          {[
            { id: 'robotic-vocoder', label: 'Robotic Vocoder' },
            { id: 'deep-pitch-neural', label: 'Deep Pitch Shift' },
            { id: 'synthetic-clone', label: 'AI Voice Clone' },
          ].map((p) => (
            <button
              key={p.id}
              onClick={() => onSelectVoicePreset(p.id as any)}
              className={`w-full py-1.5 px-2.5 rounded-lg text-left text-xs transition-colors cursor-pointer ${
                voicePreset === p.id
                  ? 'bg-[#8ab4f8]/15 text-[#8ab4f8] font-medium'
                  : 'text-[#9aa0a6] hover:bg-[#202124] hover:text-[#e8eaed]'
              }`}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      {/* Combined & Reset */}
      <div className="mt-4 space-y-2">
        <button
          onClick={onActivateCombined}
          className="w-full py-2.5 px-4 rounded-full bg-[#ea4335] hover:bg-[#d93025] text-white text-sm font-medium shadow-md transition-colors flex items-center justify-center gap-2 active:scale-95 cursor-pointer"
        >
          <Flame className="h-4 w-4" />
          Full Attack (Face + Voice)
        </button>

        <button
          onClick={onReset}
          className="w-full py-2 px-4 rounded-full bg-[#3c4043] border border-[#5f6368] text-[#e8eaed] text-sm hover:bg-[#4a4d51] transition-colors flex items-center justify-center gap-2 cursor-pointer"
        >
          <RotateCcw className="h-3.5 w-3.5" />
          Reset to Clean Feed
        </button>
      </div>

      <div className="mt-auto pt-4 text-[10px] text-[#9aa0a6] text-center">
        Role: <span className="text-[#34a853] font-medium">Tester (Verified)</span>
      </div>
    </div>
  );
};
