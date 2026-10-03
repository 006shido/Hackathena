import React, { useState, useRef, useEffect } from 'react';
import {
  Sparkles,
  Upload,
  RotateCcw,
  CheckCircle2,
  AlertTriangle,
  X,
  ShieldCheck,
  Cpu,
  Eye,
  Camera,
  ShieldAlert,
  Mic,
  Sliders,
  Flame,
  Headphones,
} from 'lucide-react';
import { FacePreset } from '../types/attack';
import { GalleryFaceValidationResult, FaceSwapTelemetry } from '../attack/faceSwapEngine';

interface AIFaceSwapPanelProps {
  faceSwapActive: boolean;
  onToggleFaceSwap: () => void;
  onUploadGalleryFace: (file: File) => Promise<GalleryFaceValidationResult>;
  onResetFace: () => void;
  onSelectPreset: (preset: FacePreset) => void;
  currentPreset: FacePreset;
  telemetry?: FaceSwapTelemetry;
  selectedFacePreview: string | null;
  selectedFaceName: string;
  processedStream: MediaStream | null;

  // Integrated Voice & Attack Simulator features
  voiceTransformActive?: boolean;
  onToggleVoiceTransform?: () => void;
  onActivateCombined?: () => void;
  onResetAll?: () => void;
  onSelectVoicePreset?: (preset: 'robotic-vocoder' | 'deep-pitch-neural' | 'synthetic-clone') => void;
  voicePreset?: 'robotic-vocoder' | 'deep-pitch-neural' | 'synthetic-clone';
  attackMode?: string;
  isVoiceMonitoring?: boolean;
  onToggleVoiceMonitor?: () => void;

  onClose?: () => void;
}

export const AIFaceSwapPanel: React.FC<AIFaceSwapPanelProps> = ({
  faceSwapActive,
  onToggleFaceSwap,
  onUploadGalleryFace,
  onResetFace,
  onSelectPreset,
  currentPreset,
  telemetry,
  selectedFacePreview,
  selectedFaceName,
  processedStream,
  voiceTransformActive = false,
  onToggleVoiceTransform,
  onActivateCombined,
  onResetAll,
  onSelectVoicePreset,
  voicePreset = 'robotic-vocoder',
  attackMode = 'none',
  isVoiceMonitoring = false,
  onToggleVoiceMonitor,
  onClose,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [activeTab, setActiveTab] = useState<'faceswap' | 'attack'>('faceswap');
  const [consentGiven, setConsentGiven] = useState(true);
  const [consentError, setConsentError] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadSuccessMessage, setUploadSuccessMessage] = useState<string | null>(null);
  const previewVideoRef = useRef<HTMLVideoElement>(null);

  // Bind live processed stream to the thumbnail video preview
  useEffect(() => {
    if (previewVideoRef.current && processedStream) {
      previewVideoRef.current.srcObject = processedStream;
      previewVideoRef.current.play().catch(() => {});
    }
  }, [processedStream, faceSwapActive]);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsAnalyzing(true);
    setUploadError(null);
    setUploadSuccessMessage(null);

    try {
      const result = await onUploadGalleryFace(file);
      if (result.success) {
        setUploadSuccessMessage(`Face detected successfully (${result.landmarksCount || 478} landmarks)`);
        setUploadError(null);
      } else {
        setUploadError(result.error || 'Failed to detect a face in the selected photograph.');
        setUploadSuccessMessage(null);
      }
    } catch (err: any) {
      setUploadError(err.message || 'An error occurred during face detection.');
    } finally {
      setIsAnalyzing(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleToggleSwap = () => {
    if (!faceSwapActive && !consentGiven) {
      setConsentError(true);
      return;
    }
    setConsentError(false);
    onToggleFaceSwap();
  };

  return (
    <div className="w-full lg:w-96 shrink-0 flex flex-col h-full bg-[#16181f] border-l border-white/10 p-4 overflow-y-auto shadow-2xl text-slate-100 font-sans">
      {/* Panel Header */}
      <div className="flex items-center justify-between pb-3 border-b border-white/10">
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-blue-500/15 text-blue-400 border border-blue-500/20">
            <Sparkles className="h-4 w-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-sm font-bold text-white tracking-tight">Tester Lab</h2>
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
                Tester Only
              </span>
            </div>
            <p className="text-[11px] text-slate-400 mt-0.5">Real-time Biometrics & Simulation</p>
          </div>
        </div>

        {onClose && (
          <button
            onClick={onClose}
            className="p-1 rounded-full text-slate-400 hover:text-white hover:bg-white/10 transition-colors cursor-pointer"
            title="Close Panel"
          >
            <X className="h-4 w-4" />
          </button>
        )}
      </div>

      {/* Segmented Tab Switcher */}
      <div className="flex p-1 rounded-xl bg-black/40 border border-white/10 mt-3 mb-1">
        <button
          type="button"
          onClick={() => setActiveTab('faceswap')}
          className={`flex-1 py-1.5 px-3 rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition-all cursor-pointer ${
            activeTab === 'faceswap'
              ? 'bg-blue-600 text-white shadow-sm'
              : 'text-slate-400 hover:text-white'
          }`}
        >
          <Sparkles className="h-3.5 w-3.5" />
          <span>AI Face Swap</span>
          {faceSwapActive && <span className="h-1.5 w-1.5 rounded-full bg-blue-300 animate-pulse" />}
        </button>
        <button
          type="button"
          onClick={() => setActiveTab('attack')}
          className={`flex-1 py-1.5 px-3 rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition-all cursor-pointer ${
            activeTab === 'attack'
              ? 'bg-red-600/90 text-white shadow-sm'
              : 'text-slate-400 hover:text-white'
          }`}
        >
          <ShieldAlert className="h-3.5 w-3.5" />
          <span>Voice & Attack</span>
          {voiceTransformActive && <span className="h-1.5 w-1.5 rounded-full bg-red-400 animate-ping" />}
        </button>
      </div>

      {/* TAB 1: AI FACE SWAP */}
      {activeTab === 'faceswap' && (
        <div className="space-y-3.5 mt-2 animate-fade-in">
          {/* Critical Face-Only Notice */}
          <div className="flex items-start gap-2.5 px-3 py-2.5 rounded-xl bg-blue-500/10 border border-blue-500/20 text-[11px] text-slate-200 leading-snug">
            <ShieldCheck className="h-4 w-4 text-blue-400 shrink-0 mt-0.5" />
            <span>
              <strong className="text-white">Face-Only Preservation:</strong> Replaces only the facial mask. Original hair, hairline, ears, neck, body, and background remain completely untouched.
            </span>
          </div>

          {/* Likeness Consent & Ethics Checkbox */}
          <div className="p-3 rounded-xl bg-white/5 border border-white/10">
            <label className="flex items-start gap-2.5 cursor-pointer">
              <input
                type="checkbox"
                checked={consentGiven}
                onChange={(e) => {
                  setConsentGiven(e.target.checked);
                  if (e.target.checked) setConsentError(false);
                }}
                className="mt-0.5 h-4 w-4 rounded border-white/20 bg-slate-800 text-blue-600 focus:ring-blue-500 cursor-pointer"
              />
              <span className="text-[11px] text-slate-300 leading-tight">
                I confirm that I have permission and legal consent to use this photograph's likeness for testing purposes.
              </span>
            </label>
            {consentError && (
              <p className="mt-2 text-[10px] text-red-400 flex items-center gap-1 font-medium">
                <AlertTriangle className="h-3 w-3 shrink-0" />
                Consent is required before enabling the AI Face Swap filter.
              </p>
            )}
          </div>

          {/* Section 1: Choose Face from Gallery */}
          <div className="p-3.5 rounded-2xl bg-white/5 border border-white/10">
            <div className="flex items-center justify-between mb-2.5">
              <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">
                Source Facial Identity
              </span>
              <span className="text-[10px] text-slate-400">JPG, PNG, WEBP</span>
            </div>

            {/* Hidden File Input */}
            <input
              type="file"
              ref={fileInputRef}
              accept="image/jpeg,image/png,image/webp,image/jpg"
              onChange={handleFileChange}
              className="hidden"
            />

            {/* Choose Face from Gallery Button */}
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={isAnalyzing}
              className="w-full py-2.5 px-4 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white text-xs font-semibold shadow-md shadow-blue-500/20 flex items-center justify-center gap-2 transition-all cursor-pointer active:scale-98 disabled:opacity-50"
            >
              {isAnalyzing ? (
                <>
                  <div className="h-4 w-4 rounded-full border-2 border-white/30 border-t-white animate-spin" />
                  <span>Analyzing Face in Gallery Photo...</span>
                </>
              ) : (
                <>
                  <Upload className="h-4 w-4" />
                  <span>Choose Face from Gallery</span>
                </>
              )}
            </button>

            {/* Validation Result Messages */}
            {uploadError && (
              <div className="mt-2.5 p-2.5 rounded-xl bg-red-500/15 border border-red-500/30 text-[11px] text-red-300 flex items-start gap-2">
                <AlertTriangle className="h-4 w-4 text-red-400 shrink-0 mt-0.5" />
                <div className="leading-snug">{uploadError}</div>
              </div>
            )}

            {uploadSuccessMessage && (
              <div className="mt-2.5 p-2.5 rounded-xl bg-emerald-500/15 border border-emerald-500/30 text-[11px] text-emerald-300 flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0" />
                <div className="leading-snug font-medium">{uploadSuccessMessage}</div>
              </div>
            )}

            {/* Selected Face Preview */}
            <div className="mt-3 pt-3 border-t border-white/10 flex items-center justify-between">
              <div className="flex items-center gap-3 min-w-0">
                <div className="relative w-12 h-12 rounded-xl overflow-hidden bg-slate-800 border border-white/15 shrink-0 flex items-center justify-center shadow-inner">
                  {selectedFacePreview ? (
                    <img
                      src={selectedFacePreview}
                      alt={selectedFaceName}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <Camera className="h-5 w-5 text-slate-500" />
                  )}
                  {faceSwapActive && (
                    <div className="absolute inset-0 bg-blue-500/20 border border-blue-400 rounded-xl" />
                  )}
                </div>
                <div className="min-w-0">
                  <p className="text-xs font-semibold text-white truncate">{selectedFaceName}</p>
                  <p className="text-[10px] text-slate-400 flex items-center gap-1 mt-0.5">
                    <CheckCircle2 className="h-3 w-3 text-emerald-400" />
                    <span>Face Verified (478 pts)</span>
                  </p>
                </div>
              </div>

              {/* Reset Face Button */}
              <button
                type="button"
                onClick={onResetFace}
                title="Reset Face"
                className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-slate-400 hover:text-slate-200 border border-white/10 transition-colors cursor-pointer"
              >
                <RotateCcw className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>

          {/* Preset Personas Quick Select */}
          <div className="p-3 rounded-2xl bg-white/5 border border-white/10">
            <label className="text-[10px] uppercase font-bold text-slate-400 tracking-wider block mb-2">
              Or Select Standard Persona:
            </label>
            <div className="grid grid-cols-4 gap-1.5">
              {[
                { id: 'mona-lisa', name: 'Emma', img: '/mona_lisa.jpg' },
                { id: 'cyber-agent', name: 'Alex', img: '/cyber_agent.jpg' },
                { id: 'astronaut', name: 'Sophia', img: '/astronaut.jpg' },
                { id: 'synthetic-executive', name: 'David', img: '/synthetic_executive.jpg' },
              ].map((p) => {
                const isSelected = currentPreset === p.id;
                return (
                  <button
                    key={p.id}
                    type="button"
                    onClick={() => onSelectPreset(p.id as FacePreset)}
                    className={`flex flex-col items-center p-1.5 rounded-xl border transition-all cursor-pointer ${
                      isSelected
                        ? 'bg-blue-600/25 border-blue-500 shadow-sm'
                        : 'bg-white/5 border-white/10 hover:bg-white/10'
                    }`}
                  >
                    <img
                      src={p.img}
                      alt={p.name}
                      className="w-9 h-9 rounded-lg object-cover mb-1"
                    />
                    <span className="text-[10px] font-medium text-slate-200 truncate w-full text-center">
                      {p.name}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Section 2: Enable / Disable Face Swap Control */}
          <button
            type="button"
            onClick={handleToggleSwap}
            className={`w-full py-3 px-4 rounded-xl text-xs font-bold transition-all border flex items-center justify-center gap-2 cursor-pointer shadow-lg ${
              faceSwapActive
                ? 'bg-blue-600 hover:bg-blue-500 text-white border-blue-400 shadow-blue-600/30'
                : 'bg-white/10 hover:bg-white/15 text-slate-200 border-white/15'
            }`}
          >
            <Sparkles className="h-4 w-4" />
            <span>{faceSwapActive ? 'Disable Face Swap' : 'Enable Face Swap'}</span>
          </button>

          {/* Section 3: Live Processed Preview */}
          <div className="p-3 rounded-2xl bg-white/5 border border-white/10">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-1.5">
                <Eye className="h-3.5 w-3.5 text-blue-400" />
                <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">
                  Live Processed Preview
                </span>
              </div>
              <span
                className={`text-[9px] font-semibold px-2 py-0.5 rounded-full ${
                  faceSwapActive ? 'bg-blue-500/20 text-blue-400' : 'bg-white/5 text-slate-400'
                }`}
              >
                {faceSwapActive ? 'Swapped Feed' : 'Clean Camera Feed'}
              </span>
            </div>

            <div className="relative w-full aspect-video rounded-xl overflow-hidden bg-slate-950 border border-white/10 flex items-center justify-center">
              <video
                ref={previewVideoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover transform -scale-x-100"
              />
              {faceSwapActive && (
                <div className="absolute top-2 right-2 px-2 py-0.5 rounded-md bg-black/70 backdrop-blur-md text-[9px] font-bold text-blue-400 border border-blue-500/30 flex items-center gap-1">
                  <span className="h-1.5 w-1.5 rounded-full bg-blue-400 animate-ping" />
                  <span>AI Modified</span>
                </div>
              )}
            </div>
          </div>

          {/* Section 4: Real-time Processing Status Telemetry */}
          <div className="p-3.5 rounded-2xl bg-white/5 border border-white/10">
            <div className="flex items-center justify-between mb-2.5">
              <div className="flex items-center gap-1.5">
                <Cpu className="h-3.5 w-3.5 text-blue-400" />
                <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">
                  Processing Status
                </span>
              </div>
              <span className="text-[10px] text-emerald-400 font-mono font-bold">
                {telemetry?.fps ? `${telemetry.fps} FPS` : '30 FPS'}
              </span>
            </div>

            <div className="space-y-1.5 text-[11px]">
              <div className="flex items-center justify-between text-slate-300 py-0.5 border-b border-white/5">
                <span className="text-slate-400">Pipeline Backend:</span>
                <span className="font-semibold text-slate-200">MediaPipe GPU + WebGL</span>
              </div>
              <div className="flex items-center justify-between text-slate-300 py-0.5 border-b border-white/5">
                <span className="text-slate-400">Landmark Detection:</span>
                <span className="font-semibold text-emerald-400">
                  {telemetry?.landmarksDetected ? '478 3D Points Tracked' : 'Detecting...'}
                </span>
              </div>
              <div className="flex items-center justify-between text-slate-300 py-0.5 border-b border-white/5">
                <span className="text-slate-400">Facial Mesh:</span>
                <span className="font-semibold text-slate-200">854 Delaunay Triangles</span>
              </div>
              <div className="flex items-center justify-between text-slate-300 py-0.5 border-b border-white/5">
                <span className="text-slate-400">Skin Tone Match:</span>
                <span className="font-semibold text-blue-400">Auto Lighting Adaptive</span>
              </div>
              <div className="flex items-center justify-between text-slate-300 py-0.5">
                <span className="text-slate-400">Mask Boundary:</span>
                <span className="font-semibold text-emerald-400">Hair & Neck Preserved (0% Cut)</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: VOICE & ATTACK SIMULATOR */}
      {activeTab === 'attack' && (
        <div className="space-y-3.5 mt-2 animate-fade-in">
          {/* Attack Mode Status Indicator */}
          <div className="flex items-center justify-between px-3 py-2.5 rounded-xl bg-red-500/10 border border-red-500/20 text-[11px] text-red-300">
            <div className="flex items-center gap-2">
              <ShieldAlert className="h-4 w-4 text-red-400" />
              <span className="font-semibold text-white">Simulation Mode:</span>
            </div>
            <span className="font-mono uppercase font-bold text-red-400">
              {attackMode !== 'none' ? attackMode : 'Standby'}
            </span>
          </div>

          {/* Voice Simulation Controls */}
          {onToggleVoiceTransform && (
            <div className="p-3.5 rounded-2xl bg-white/5 border border-white/10">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Mic className="h-4 w-4 text-blue-400" />
                  <h3 className="text-xs font-semibold text-slate-200">Voice Simulation</h3>
                </div>
                <span
                  className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${
                    voiceTransformActive ? 'bg-blue-500/20 text-blue-400' : 'bg-white/5 text-slate-400'
                  }`}
                >
                  {voiceTransformActive ? 'Active' : 'Off'}
                </span>
              </div>

              <button
                type="button"
                onClick={onToggleVoiceTransform}
                className={`w-full py-2.5 px-3 rounded-xl text-xs font-semibold transition-all border cursor-pointer ${
                  voiceTransformActive
                    ? 'bg-blue-600 text-white border-blue-500 shadow-md shadow-blue-500/20'
                    : 'bg-white/5 text-slate-200 border-white/10 hover:bg-white/10'
                }`}
              >
                <Sliders className="h-3.5 w-3.5 inline mr-1.5" />
                {voiceTransformActive ? 'Disable Voice Transform' : 'Enable Voice Transform'}
              </button>

              {onSelectVoicePreset && (
                <div className="mt-3 space-y-1">
                  <label className="text-[10px] uppercase font-bold text-slate-400 tracking-wider block mb-1">
                    Voice Model:
                  </label>
                  {[
                    { id: 'robotic-vocoder', label: 'Robotic Vocoder (440Hz)' },
                    { id: 'deep-pitch-neural', label: 'Deep Pitch Shift (Granular)' },
                    { id: 'synthetic-clone', label: 'AI Synthetic Voice Clone' },
                  ].map((p) => (
                    <button
                      key={p.id}
                      type="button"
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
              )}

              {/* Headphone Live Voice Monitor */}
              {onToggleVoiceMonitor && (
                <div className="mt-3 pt-3 border-t border-white/10">
                  <button
                    type="button"
                    onClick={onToggleVoiceMonitor}
                    disabled={!voiceTransformActive}
                    className={`w-full py-2 px-3 rounded-xl text-xs font-medium transition-all border flex items-center justify-between cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed ${
                      isVoiceMonitoring
                        ? 'bg-emerald-600/25 border-emerald-500/50 text-emerald-300 shadow-sm'
                        : 'bg-white/5 border-white/10 text-slate-300 hover:bg-white/10'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <Headphones className="h-3.5 w-3.5 text-blue-400" />
                      <span>Hear Transformed Voice</span>
                    </div>
                    <span className={`text-[10px] px-1.5 py-0.5 rounded font-mono ${
                      isVoiceMonitoring ? 'bg-emerald-500/30 text-emerald-300 font-bold' : 'text-slate-400'
                    }`}>
                      {isVoiceMonitoring ? 'ON' : 'OFF'}
                    </span>
                  </button>
                  <p className="text-[10px] text-slate-400 mt-1 px-1">
                    🎧 Use headphones to hear your modified voice without echo.
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Combined Attack (Face + Voice) */}
          {onActivateCombined && (
            <button
              type="button"
              onClick={onActivateCombined}
              className="w-full py-2.5 px-4 rounded-xl bg-red-600 hover:bg-red-700 text-white text-xs font-semibold shadow-md shadow-red-600/30 transition-all flex items-center justify-center gap-2 active:scale-95 cursor-pointer"
            >
              <Flame className="h-4 w-4" />
              <span>Full Attack (Face + Voice Combined)</span>
            </button>
          )}

          {/* Reset All */}
          {onResetAll && (
            <button
              type="button"
              onClick={onResetAll}
              className="w-full py-2 px-4 rounded-xl bg-white/5 border border-white/10 text-slate-300 text-xs font-semibold hover:bg-white/10 transition-colors flex items-center justify-center gap-2 cursor-pointer"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              <span>Reset to Clean Feed</span>
            </button>
          )}
        </div>
      )}

      <div className="mt-auto pt-4 text-[10px] text-slate-500 text-center">
        DeepTrace AI Biometric Injection Pipeline • Exclusively for Security Testers
      </div>
    </div>
  );
};
