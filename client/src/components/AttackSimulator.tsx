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
  let statusBannerText = 'NORMAL';
  let statusBannerStyle = 'bg-slate-800 text-slate-300 border-slate-700';

  if (mode === 'combined') {
    statusBannerText = 'SIMULATED DEEPFAKE ATTACK ACTIVE';
    statusBannerStyle = 'bg-rose-500/20 text-rose-300 border-rose-500/60 shadow-lg shadow-rose-950/40 animate-pulse';
  } else if (mode === 'face') {
    statusBannerText = 'FACE SIMULATION: ACTIVE';
    statusBannerStyle = 'bg-amber-500/20 text-amber-300 border-amber-500/50 shadow-lg shadow-amber-950/30';
  } else if (mode === 'voice') {
    statusBannerText = 'VOICE SIMULATION: ACTIVE';
    statusBannerStyle = 'bg-purple-500/20 text-purple-300 border-purple-500/50 shadow-lg shadow-purple-950/30';
  }

  return (
    <div className="w-full lg:w-88 shrink-0 flex flex-col h-full bg-slate-950/95 border-l border-amber-500/30 p-4 sm:p-5 overflow-y-auto backdrop-blur-xl shadow-2xl">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800/80">
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-amber-500/20 text-amber-400 border border-amber-500/40 shadow-sm">
            <ShieldAlert className="h-4.5 w-4.5" />
          </div>
          <div>
            <h2 className="text-sm font-bold tracking-tight text-white flex items-center gap-1.5">
              <span>Attack Simulator</span>
              <span className="rounded bg-amber-500/20 px-1.5 py-0.5 text-[9px] font-mono text-amber-300 uppercase">
                Tester Only
              </span>
            </h2>
            <p className="text-[11px] text-slate-400 font-mono">Controlled security testing</p>
          </div>
        </div>

        {onClose && (
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X className="h-4 w-4" />
          </button>
        )}
      </div>

      {/* Prominent Status Callout */}
      <div className={`mt-4 p-3 rounded-xl border text-center font-mono text-xs font-bold tracking-wider transition-all duration-300 ${statusBannerStyle}`}>
        <div className="flex items-center justify-center gap-2">
          {mode !== 'none' && <Flame className="h-4 w-4 shrink-0 text-current" />}
          <span>{statusBannerText}</span>
        </div>
      </div>

      {/* Safety Notice */}
      <div className="mt-3 flex items-start gap-2 p-2.5 rounded-lg bg-amber-500/10 border border-amber-500/20 text-[11px] text-amber-300/90 font-mono">
        <AlertTriangle className="h-3.5 w-3.5 shrink-0 mt-0.5 text-amber-400" />
        <span>Simulations strictly modify outgoing media for test purposes.</span>
      </div>

      {/* Section 1: Face Simulation */}
      <div className="mt-5 p-4 rounded-2xl bg-slate-900/60 border border-slate-800/90 shadow-sm">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Eye className="h-4 w-4 text-cyan-400" />
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200 font-mono">
              Face Simulation
            </h3>
          </div>
          <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded uppercase ${
            faceSwap ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40' : 'bg-slate-800 text-slate-400'
          }`}>
            {faceSwap ? 'ON' : 'OFF'}
          </span>
        </div>

        <button
          onClick={onToggleFaceSwap}
          className={`w-full py-2.5 px-4 rounded-xl text-xs font-mono font-semibold tracking-wider transition-all duration-200 border flex items-center justify-center gap-2 shadow-sm ${
            faceSwap
              ? 'bg-amber-600/30 text-amber-200 border-amber-500 hover:bg-amber-600/40'
              : 'bg-slate-800/90 text-slate-200 border-slate-700/80 hover:bg-slate-700 hover:text-white'
          }`}
        >
          <Sparkles className="h-3.5 w-3.5" />
          <span>{faceSwap ? 'Disable Face Swap' : 'Enable Face Swap'}</span>
        </button>

        {/* Preset Selector */}
        <div className="mt-3">
          <label className="text-[10px] font-mono text-slate-400 uppercase tracking-wider block mb-1.5">
            Face Swap Preset:
          </label>
          <div className="grid grid-cols-1 gap-1.5 text-xs font-mono">
            {[
              { id: 'neural-clone', label: 'Synthetic Persona' },
              { id: 'synthetic-executive', label: 'Corporate Executive' },
              { id: 'biometric-mask', label: 'Biometric Mesh HUD' },
            ].map((p) => (
              <button
                key={p.id}
                onClick={() => onSelectFacePreset(p.id as any)}
                className={`py-1.5 px-3 rounded-lg text-left text-[11px] transition-all border ${
                  facePreset === p.id
                    ? 'bg-cyan-500/15 text-cyan-300 border-cyan-500/40 font-semibold'
                    : 'bg-slate-900/60 text-slate-400 border-slate-800 hover:bg-slate-800 hover:text-slate-200'
                }`}
              >
                • {p.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Section 2: Voice Simulation */}
      <div className="mt-4 p-4 rounded-2xl bg-slate-900/60 border border-slate-800/90 shadow-sm">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Mic className="h-4 w-4 text-purple-400" />
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200 font-mono">
              Voice Simulation
            </h3>
          </div>
          <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded uppercase ${
            voiceTransform ? 'bg-purple-500/20 text-purple-300 border border-purple-500/40' : 'bg-slate-800 text-slate-400'
          }`}>
            {voiceTransform ? 'ON' : 'OFF'}
          </span>
        </div>

        <button
          onClick={onToggleVoiceTransform}
          className={`w-full py-2.5 px-4 rounded-xl text-xs font-mono font-semibold tracking-wider transition-all duration-200 border flex items-center justify-center gap-2 shadow-sm ${
            voiceTransform
              ? 'bg-purple-600/30 text-purple-200 border-purple-500 hover:bg-purple-600/40'
              : 'bg-slate-800/90 text-slate-200 border-slate-700/80 hover:bg-slate-700 hover:text-white'
          }`}
        >
          <Sliders className="h-3.5 w-3.5" />
          <span>{voiceTransform ? 'Disable Voice Transform' : 'Enable Voice Transformation'}</span>
        </button>

        {/* Voice Presets */}
        <div className="mt-3">
          <label className="text-[10px] font-mono text-slate-400 uppercase tracking-wider block mb-1.5">
            Voice Modulation:
          </label>
          <div className="grid grid-cols-1 gap-1.5 text-xs font-mono">
            {[
              { id: 'robotic-vocoder', label: 'Robotic Vocoder Drone' },
              { id: 'deep-pitch-neural', label: 'Deep Pitch Neural Shift' },
              { id: 'synthetic-clone', label: 'AI Voice Resynthesizer' },
            ].map((p) => (
              <button
                key={p.id}
                onClick={() => onSelectVoicePreset(p.id as any)}
                className={`py-1.5 px-3 rounded-lg text-left text-[11px] transition-all border ${
                  voicePreset === p.id
                    ? 'bg-purple-500/15 text-purple-300 border-purple-500/40 font-semibold'
                    : 'bg-slate-900/60 text-slate-400 border-slate-800 hover:bg-slate-800 hover:text-slate-200'
                }`}
              >
                • {p.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Section 3: Combined Attack & Reset */}
      <div className="mt-5 space-y-2.5">
        <button
          onClick={onActivateCombined}
          className="w-full py-3 px-4 rounded-xl bg-gradient-to-r from-rose-600 via-red-600 to-amber-600 text-white text-xs font-mono font-bold tracking-wider uppercase shadow-lg shadow-rose-950/60 hover:from-rose-500 hover:to-amber-500 transition-all flex items-center justify-center gap-2 active:scale-95"
        >
          <Flame className="h-4 w-4" />
          <span>Face + Voice Attack</span>
        </button>

        <button
          onClick={onReset}
          className="w-full py-2.5 px-4 rounded-xl bg-slate-900 border border-slate-700/80 text-slate-300 text-xs font-mono font-medium hover:bg-slate-800 hover:text-white transition-colors flex items-center justify-center gap-2"
        >
          <RotateCcw className="h-3.5 w-3.5" />
          <span>Reset Attack</span>
        </button>
      </div>

      {/* Footer Audit Info */}
      <div className="mt-auto pt-6 text-[10px] font-mono text-slate-500 text-center">
        Server Role Verification: <span className="text-emerald-400">PASSED (TESTER)</span>
      </div>
    </div>
  );
};
