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

  // Derive high-level status banner text
  let statusBannerText = 'SIMULATOR STANDBY';
  let statusBannerStyle = 'bg-zinc-900/80 text-zinc-300 border-zinc-800';

  if (mode === 'combined') {
    statusBannerText = 'DEEPFAKE ATTACK ACTIVE';
    statusBannerStyle = 'bg-rose-500/15 text-rose-300 border-rose-500/60 shadow-lg shadow-rose-950/40 animate-pulse';
  } else if (mode === 'face') {
    statusBannerText = 'FACE SIMULATION ACTIVE';
    statusBannerStyle = 'bg-orange-500/15 text-orange-300 border-orange-500/50 shadow-lg shadow-orange-950/30';
  } else if (mode === 'voice') {
    statusBannerText = 'VOICE SIMULATION ACTIVE';
    statusBannerStyle = 'bg-orange-500/15 text-orange-300 border-orange-500/50 shadow-lg shadow-orange-950/30';
  }

  return (
    <div className="w-full lg:w-88 shrink-0 flex flex-col h-full bg-[#050507] border-l border-orange-500/30 p-4 sm:p-5 overflow-y-auto shadow-2xl">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-zinc-800">
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-md bg-orange-500/15 text-orange-400 border border-orange-500/40 shadow-sm">
            <ShieldAlert className="h-4.5 w-4.5" />
          </div>
          <div>
            <h2 className="text-sm font-bold tracking-tight text-white flex items-center gap-1.5">
              <span>Attack Simulator</span>
              <span className="rounded bg-orange-500/20 px-1.5 py-0.5 text-[9px] font-mono text-orange-400 uppercase font-semibold">
                Tester
              </span>
            </h2>
            <p className="text-[11px] text-zinc-400 font-mono">Adversarial Stream Injection</p>
          </div>
        </div>

        {onClose && (
          <button
            onClick={onClose}
            className="p-1 rounded text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors cursor-pointer"
          >
            <X className="h-4 w-4" />
          </button>
        )}
      </div>

      {/* Prominent Status Callout */}
      <div className={`mt-4 p-2.5 rounded-md border text-center font-mono text-xs font-bold tracking-wider transition-all duration-200 ${statusBannerStyle}`}>
        <div className="flex items-center justify-center gap-2">
          {mode !== 'none' && <Flame className="h-4 w-4 shrink-0 text-current" />}
          <span>{statusBannerText}</span>
        </div>
      </div>

      {/* Safety Notice */}
      <div className="mt-3 flex items-start gap-2 p-2.5 rounded-md bg-orange-500/10 border border-orange-500/25 text-[11px] text-orange-300 font-mono">
        <AlertTriangle className="h-3.5 w-3.5 shrink-0 mt-0.5 text-orange-400" />
        <span>Injected streams replace outgoing WebRTC tracks in real time.</span>
      </div>

      {/* Section 1: Face Simulation */}
      <div className="mt-4 p-3.5 rounded-md bg-black border border-zinc-800 shadow-sm">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Eye className="h-4 w-4 text-orange-400" />
            <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-200 font-mono">
              Face Simulation
            </h3>
          </div>
          <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded uppercase ${
            faceSwap ? 'bg-orange-500/20 text-orange-300 border border-orange-500/40' : 'bg-zinc-900 text-zinc-500 border border-zinc-800'
          }`}>
            {faceSwap ? 'ACTIVE' : 'OFF'}
          </span>
        </div>

        <button
          onClick={onToggleFaceSwap}
          className={`w-full py-2 px-3 rounded-md text-xs font-mono font-semibold tracking-wider transition-all duration-150 border flex items-center justify-center gap-2 shadow-sm cursor-pointer ${
            faceSwap
              ? 'bg-orange-600/30 text-orange-200 border-orange-500 hover:bg-orange-600/40'
              : 'bg-zinc-900 text-zinc-200 border-zinc-700 hover:bg-zinc-800 hover:text-white'
          }`}
        >
          <Sparkles className="h-3.5 w-3.5" />
          <span>{faceSwap ? 'Disable Face Swap' : 'Enable Face Swap'}</span>
        </button>

        {/* Preset Selector */}
        <div className="mt-3">
          <label className="text-[10px] font-mono text-zinc-400 uppercase tracking-wider block mb-1.5">
            Synthetic Face Model:
          </label>
          <div className="grid grid-cols-1 gap-1 text-xs font-mono">
            {[
              { id: 'neural-clone', label: 'Synthetic Persona' },
              { id: 'synthetic-executive', label: 'Corporate Executive' },
              { id: 'biometric-mask', label: 'Biometric Mesh HUD' },
            ].map((p) => (
              <button
                key={p.id}
                onClick={() => onSelectFacePreset(p.id as any)}
                className={`py-1.5 px-2.5 rounded text-left text-[11px] transition-all border cursor-pointer ${
                  facePreset === p.id
                    ? 'bg-orange-500/15 text-orange-300 border-orange-500/40 font-semibold'
                    : 'bg-zinc-950 text-zinc-400 border-zinc-850 hover:bg-zinc-900 hover:text-zinc-200'
                }`}
              >
                • {p.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Section 2: Voice Simulation */}
      <div className="mt-3.5 p-3.5 rounded-md bg-black border border-zinc-800 shadow-sm">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Mic className="h-4 w-4 text-orange-400" />
            <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-200 font-mono">
              Voice Simulation
            </h3>
          </div>
          <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded uppercase ${
            voiceTransform ? 'bg-orange-500/20 text-orange-300 border border-orange-500/40' : 'bg-zinc-900 text-zinc-500 border border-zinc-800'
          }`}>
            {voiceTransform ? 'ACTIVE' : 'OFF'}
          </span>
        </div>

        <button
          onClick={onToggleVoiceTransform}
          className={`w-full py-2 px-3 rounded-md text-xs font-mono font-semibold tracking-wider transition-all duration-150 border flex items-center justify-center gap-2 shadow-sm cursor-pointer ${
            voiceTransform
              ? 'bg-orange-600/30 text-orange-200 border-orange-500 hover:bg-orange-600/40'
              : 'bg-zinc-900 text-zinc-200 border-zinc-700 hover:bg-zinc-800 hover:text-white'
          }`}
        >
          <Sliders className="h-3.5 w-3.5" />
          <span>{voiceTransform ? 'Disable Voice Transform' : 'Enable Voice Transformation'}</span>
        </button>

        {/* Voice Presets */}
        <div className="mt-3">
          <label className="text-[10px] font-mono text-zinc-400 uppercase tracking-wider block mb-1.5">
            Voice Modulation Graph:
          </label>
          <div className="grid grid-cols-1 gap-1 text-xs font-mono">
            {[
              { id: 'robotic-vocoder', label: 'Robotic Vocoder Drone' },
              { id: 'deep-pitch-neural', label: 'Deep Pitch Neural Shift' },
              { id: 'synthetic-clone', label: 'AI Voice Resynthesizer' },
            ].map((p) => (
              <button
                key={p.id}
                onClick={() => onSelectVoicePreset(p.id as any)}
                className={`py-1.5 px-2.5 rounded text-left text-[11px] transition-all border cursor-pointer ${
                  voicePreset === p.id
                    ? 'bg-orange-500/15 text-orange-300 border-orange-500/40 font-semibold'
                    : 'bg-zinc-950 text-zinc-400 border-zinc-850 hover:bg-zinc-900 hover:text-zinc-200'
                }`}
              >
                • {p.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Section 3: Combined Attack & Reset */}
      <div className="mt-4 space-y-2">
        <button
          onClick={onActivateCombined}
          className="w-full py-2.5 px-4 rounded-md bg-rose-600 hover:bg-rose-500 text-white text-xs font-mono font-bold tracking-wider uppercase shadow-lg shadow-rose-950/60 border border-rose-500 transition-all flex items-center justify-center gap-2 active:scale-95 cursor-pointer"
        >
          <Flame className="h-4 w-4" />
          <span>Face + Voice Attack</span>
        </button>

        <button
          onClick={onReset}
          className="w-full py-2 px-4 rounded-md bg-zinc-900 border border-zinc-700 text-zinc-300 text-xs font-mono font-medium hover:bg-zinc-800 hover:text-white transition-colors flex items-center justify-center gap-2 cursor-pointer"
        >
          <RotateCcw className="h-3.5 w-3.5" />
          <span>Reset to Clean Feed</span>
        </button>
      </div>

      {/* Footer Audit Info */}
      <div className="mt-auto pt-4 text-[10px] font-mono text-zinc-500 text-center">
        Server Role Verification: <span className="text-emerald-400">PASSED (TESTER)</span>
      </div>
    </div>
  );
};
