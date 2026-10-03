import React, { useRef } from 'react';
import {
  ShieldAlert,
  Flame,
  RotateCcw,
  Sparkles,
  Mic,
  Eye,
  Sliders,
  X,
  Upload,
} from 'lucide-react';
import { AttackState, FacePreset } from '../types/attack';

interface AttackSimulatorProps {
  attackState: AttackState;
  onToggleFaceSwap: () => void;
  onToggleVoiceTransform: () => void;
  onActivateCombined: () => void;
  onReset: () => void;
  onSelectFacePreset: (preset: FacePreset) => void;
  onSelectVoicePreset: (preset: 'robotic-vocoder' | 'deep-pitch-neural' | 'synthetic-clone') => void;
  onUploadCustomFace?: (dataUrl: string) => void;
  onUploadGalleryFile?: (file: File) => Promise<any>;
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
  onUploadCustomFace,
  onUploadGalleryFile,
  onClose,
}) => {
  const { faceSwap, voiceTransform, mode, facePreset, voicePreset } = attackState;
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (onUploadGalleryFile) {
        onUploadGalleryFile(file);
      } else if (onUploadCustomFace) {
        const reader = new FileReader();
        reader.onload = () => {
          if (typeof reader.result === 'string') {
            onUploadCustomFace(reader.result);
          }
        };
        reader.readAsDataURL(file);
      }
    }
  };

  return (
    <div className="w-full lg:w-88 shrink-0 flex flex-col h-full bg-[#16181f] border-l border-white/10 p-4 overflow-y-auto shadow-2xl text-slate-100 font-sans">
      {/* Header */}
      <div className="flex items-center justify-between pb-3.5 border-b border-white/10">
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-red-500/15 text-red-400 border border-red-500/20">
            <ShieldAlert className="h-4 w-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-sm font-bold text-white tracking-tight">Attack Simulator</h2>
              <span
                className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-semibold border ${
                  mode !== 'none'
                    ? 'bg-red-500/20 text-red-400 border-red-500/40'
                    : 'bg-white/5 text-slate-400 border-white/10'
                }`}
              >
                <span
                  className={`h-1.5 w-1.5 rounded-full ${
                    mode !== 'none' ? 'bg-red-400 animate-ping' : 'bg-slate-400'
                  }`}
                />
                {mode !== 'none' ? (mode === 'combined' ? 'Full Active' : `${mode} Active`) : 'Standby'}
              </span>
            </div>
            <p className="text-[11px] text-slate-400 mt-0.5">Tester Injection Lab</p>
          </div>
        </div>

        {onClose && (
          <button
            onClick={onClose}
            className="p-1 rounded-full text-slate-400 hover:text-white hover:bg-white/10 transition-colors cursor-pointer"
            title="Close"
          >
            <X className="h-4 w-4" />
          </button>
        )}
      </div>

      {/* Info notice */}
      <div className="mt-3 flex items-start gap-2.5 px-3 py-2.5 rounded-xl bg-white/5 border border-white/10 text-[11px] text-slate-300 leading-snug">
        <Sparkles className="h-3.5 w-3.5 text-blue-400 shrink-0 mt-0.5" />
        <span>Injected media streams replace outgoing WebRTC tracks in real time for penetration testing.</span>
      </div>

      {/* Face Simulation */}
      <div className="mt-3.5 p-3.5 rounded-2xl bg-white/5 border border-white/10">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Eye className="h-4 w-4 text-blue-400" />
            <h3 className="text-xs font-semibold text-slate-200">Face Simulation</h3>
          </div>
          <span
            className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${
              faceSwap ? 'bg-blue-500/20 text-blue-400' : 'bg-white/5 text-slate-400'
            }`}
          >
            {faceSwap ? 'Active' : 'Off'}
          </span>
        </div>

        <button
          onClick={onToggleFaceSwap}
          className={`w-full py-2 px-3 rounded-full text-xs font-semibold transition-all border cursor-pointer ${
            faceSwap
              ? 'bg-blue-600 text-white border-blue-500 shadow-md shadow-blue-500/20'
              : 'bg-white/5 text-slate-200 border-white/10 hover:bg-white/10'
          }`}
        >
          <Sparkles className="h-3.5 w-3.5 inline mr-1.5" />
          {faceSwap ? 'Disable Face Swap' : 'Enable Face Swap'}
        </button>

        <div className="mt-3.5 space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">
              Human Face Swap Personas:
            </label>
            <span className="text-[10px] text-blue-400 font-medium">Original Human 3D</span>
          </div>

          <div className="grid grid-cols-2 gap-2">
            {[
              {
                id: 'mona-lisa',
                name: 'Emma',
                tag: 'Studio Headshot',
                image: '/mona_lisa.jpg',
              },
              {
                id: 'cyber-agent',
                name: 'Alex',
                tag: 'Clean Headshot',
                image: '/cyber_agent.jpg',
              },
              {
                id: 'astronaut',
                name: 'Sophia',
                tag: 'Natural Portrait',
                image: '/astronaut.jpg',
              },
              {
                id: 'neural-clone',
                name: 'Marcus',
                tag: 'Confident Portrait',
                image: '/synthetic_face_avatar.jpg',
              },
              {
                id: 'synthetic-executive',
                name: 'David',
                tag: 'Corporate Exec',
                image: '/synthetic_executive.jpg',
              },
              {
                id: 'custom-upload',
                name: 'Custom Photo',
                tag: 'Your Upload',
                image: null,
              },
            ].map((p) => {
              const isSelected = facePreset === p.id;
              return (
                <button
                  key={p.id}
                  type="button"
                  onClick={() => {
                    onSelectFacePreset(p.id as FacePreset);
                    if (!faceSwap) {
                      onToggleFaceSwap();
                    }
                  }}
                  className={`group relative flex flex-col items-center p-2 rounded-xl text-left transition-all cursor-pointer border ${
                    isSelected
                      ? 'bg-blue-600/20 border-blue-500 shadow-lg shadow-blue-500/20'
                      : 'bg-white/5 border-white/10 hover:bg-white/10 hover:border-white/20'
                  }`}
                >
                  <div className="relative w-14 h-14 rounded-full overflow-hidden bg-slate-800 border-2 mb-1.5 transition-transform group-hover:scale-105 shrink-0 flex items-center justify-center">
                    {p.image ? (
                      <img
                        src={p.image}
                        alt={p.name}
                        className="w-full h-full object-cover"
                        loading="lazy"
                      />
                    ) : (
                      <div className="w-full h-full flex flex-col items-center justify-center bg-gradient-to-tr from-purple-900/60 to-blue-900/60 text-slate-300">
                        <Upload className="h-5 w-5 text-blue-400" />
                      </div>
                    )}
                    {isSelected && (
                      <div className="absolute inset-0 bg-blue-600/20 flex items-center justify-center">
                        <span className="h-2.5 w-2.5 rounded-full bg-blue-400 shadow-sm shadow-blue-300 animate-pulse" />
                      </div>
                    )}
                  </div>

                  <span
                    className={`text-xs font-semibold truncate w-full text-center ${
                      isSelected ? 'text-blue-400' : 'text-slate-200'
                    }`}
                  >
                    {p.name}
                  </span>
                  <span className="text-[9px] text-slate-400 truncate w-full text-center">
                    {p.tag}
                  </span>

                  {isSelected && (
                    <span className="absolute top-1.5 right-1.5 h-2 w-2 rounded-full bg-blue-500 shadow-[0_0_8px_#3b82f6]" />
                  )}
                </button>
              );
            })}
          </div>

          {/* Custom Face Photo Upload Button */}
          <div className="pt-1.5">
            <input
              type="file"
              ref={fileInputRef}
              accept="image/*"
              className="hidden"
              onChange={handleFileUpload}
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="w-full py-2 px-3 rounded-xl bg-white/5 hover:bg-white/10 border border-dashed border-white/20 text-xs font-medium text-slate-300 flex items-center justify-center gap-2 transition-all cursor-pointer hover:border-blue-400"
            >
              <Upload className="h-3.5 w-3.5 text-blue-400" />
              <span>Upload Custom Face Image</span>
            </button>
          </div>

          {/* Geometric & Biometric Filters */}
          <div className="pt-2">
            <label className="text-[10px] uppercase font-bold text-slate-400 tracking-wider block mb-1.5">
              Geometric & FX Filters:
            </label>
            <div className="space-y-1">
              {[
                { id: 'biometric-mask', label: 'Biometric Mesh (468-pt Wireframe)', desc: 'Real-time Delaunay geometry' },
                { id: 'cyber-filter', label: 'Cyber Augmented Visor', desc: 'HUD targeting reticle overlay' },
              ].map((p) => {
                const isSelected = facePreset === p.id;
                return (
                  <button
                    key={p.id}
                    type="button"
                    onClick={() => {
                      onSelectFacePreset(p.id as FacePreset);
                      if (!faceSwap) {
                        onToggleFaceSwap();
                      }
                    }}
                    className={`w-full py-1.5 px-2.5 rounded-lg text-left text-xs transition-colors cursor-pointer flex items-center justify-between ${
                      isSelected
                        ? 'bg-blue-600/20 text-blue-400 font-semibold border border-blue-500/30'
                        : 'text-slate-400 hover:bg-white/5 hover:text-slate-200'
                    }`}
                  >
                    <span>{p.label}</span>
                    {isSelected && <span className="h-1.5 w-1.5 rounded-full bg-blue-400" />}
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </div>


      {/* Voice Simulation */}
      <div className="mt-3 p-3.5 rounded-2xl bg-white/5 border border-white/10">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Mic className="h-4 w-4 text-blue-400" />
            <h3 className="text-xs font-semibold text-slate-200">Voice Simulation</h3>
          </div>
          <span
            className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${
              voiceTransform ? 'bg-blue-500/20 text-blue-400' : 'bg-white/5 text-slate-400'
            }`}
          >
            {voiceTransform ? 'Active' : 'Off'}
          </span>
        </div>

        <button
          onClick={onToggleVoiceTransform}
          className={`w-full py-2 px-3 rounded-full text-xs font-semibold transition-all border cursor-pointer ${
            voiceTransform
              ? 'bg-blue-600 text-white border-blue-500 shadow-md shadow-blue-500/20'
              : 'bg-white/5 text-slate-200 border-white/10 hover:bg-white/10'
          }`}
        >
          <Sliders className="h-3.5 w-3.5 inline mr-1.5" />
          {voiceTransform ? 'Disable Voice Transform' : 'Enable Voice Transform'}
        </button>

        <div className="mt-3 space-y-1">
          <label className="text-[10px] uppercase font-semibold text-slate-400 block mb-1">
            Voice Model:
          </label>
          {[
            { id: 'robotic-vocoder', label: 'Robotic Vocoder (440Hz)' },
            { id: 'deep-pitch-neural', label: 'Deep Pitch Shift (Granular)' },
            { id: 'synthetic-clone', label: 'AI Voice Clone' },
          ].map((p) => (
            <button
              key={p.id}
              onClick={() => onSelectVoicePreset(p.id as any)}
              className={`w-full py-1.5 px-2.5 rounded-lg text-left text-xs transition-colors cursor-pointer ${
                voicePreset === p.id
                  ? 'bg-blue-600/20 text-blue-400 font-semibold border border-blue-500/30'
                  : 'text-slate-400 hover:bg-white/5 hover:text-slate-200'
              }`}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      {/* Combined Attack & Reset Actions */}
      <div className="mt-4 space-y-2">
        <button
          onClick={onActivateCombined}
          className="w-full py-2.5 px-4 rounded-full bg-red-600 hover:bg-red-700 text-white text-xs font-semibold shadow-md shadow-red-600/30 transition-all flex items-center justify-center gap-2 active:scale-95 cursor-pointer"
        >
          <Flame className="h-4 w-4" />
          <span>Full Attack (Face + Voice)</span>
        </button>

        <button
          onClick={onReset}
          className="w-full py-2 px-4 rounded-full bg-white/5 border border-white/10 text-slate-300 text-xs font-semibold hover:bg-white/10 transition-colors flex items-center justify-center gap-2 cursor-pointer"
        >
          <RotateCcw className="h-3.5 w-3.5" />
          <span>Reset to Clean Feed</span>
        </button>
      </div>

      <div className="mt-auto pt-4 text-[10px] text-slate-400 text-center">
        Role: <span className="text-emerald-400 font-semibold">Tester (Verified)</span>
      </div>
    </div>
  );
};
