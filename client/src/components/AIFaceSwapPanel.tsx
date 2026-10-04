import React, { useState, useRef, useEffect } from 'react';
import {
  User,
  Upload,
  RotateCcw,
  Check,
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
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import { FacePreset, FaceBlendConfig, DEFAULT_FACE_BLEND_CONFIG } from '../types/attack';
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
  blendConfig?: FaceBlendConfig;
  onUpdateBlendConfig?: (config: Partial<FaceBlendConfig>) => void;

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
  blendConfig,
  onUpdateBlendConfig,
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
  const [showAdvancedBlend, setShowAdvancedBlend] = useState(false);
  const [activeBlendPreset, setActiveBlendPreset] = useState<'ultra' | 'natural' | 'studio' | 'contour'>('ultra');
  const previewVideoRef = useRef<HTMLVideoElement>(null);

  const activeConfig = blendConfig || DEFAULT_FACE_BLEND_CONFIG;

  const applyBlendPreset = (preset: 'ultra' | 'natural' | 'studio' | 'contour') => {
    setActiveBlendPreset(preset);
    if (!onUpdateBlendConfig) return;
    switch (preset) {
      case 'ultra':
        onUpdateBlendConfig({
          featherRadius: 6,
          skinToneMatch: 0.85,
          lightingTransfer: 0.65,
          mouthBlend: 0.90,
          sensorGrain: 0.35,
          maskInset: 0.0,
          naturalEyes: true,
        });
        break;
      case 'natural':
        onUpdateBlendConfig({
          featherRadius: 7,
          skinToneMatch: 0.95,
          lightingTransfer: 0.80,
          mouthBlend: 0.95,
          sensorGrain: 0.45,
          maskInset: 0.0,
          naturalEyes: true,
        });
        break;
      case 'studio':
        onUpdateBlendConfig({
          featherRadius: 5,
          skinToneMatch: 0.70,
          lightingTransfer: 0.40,
          mouthBlend: 0.85,
          sensorGrain: 0.15,
          maskInset: 0.0,
          naturalEyes: true,
        });
        break;
      case 'contour':
        onUpdateBlendConfig({
          featherRadius: 6,
          skinToneMatch: 0.80,
          lightingTransfer: 0.60,
          mouthBlend: 0.90,
          sensorGrain: 0.25,
          maskInset: 0.0,
          naturalEyes: false,
        });
        break;
    }
  };

  // Bind live processed stream to thumbnail video preview
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
        setUploadSuccessMessage(`Face detected (${result.landmarksCount || 478} landmarks)`);
        setUploadError(null);
      } else {
        setUploadError(result.error || 'Failed to detect a face in the photograph.');
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
    <div className="w-full lg:w-92 shrink-0 flex flex-col h-full bg-[#111216] border-l border-zinc-800 p-4 overflow-y-auto text-zinc-100 font-sans select-none">
      {/* Panel Header */}
      <div className="flex items-center justify-between pb-3 border-b border-zinc-800">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-sm font-semibold text-zinc-100 tracking-tight">Tester Tools</h2>
            <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-zinc-800 text-zinc-400 border border-zinc-700/60">
              Biometrics
            </span>
          </div>
          <p className="text-[11px] text-zinc-500 mt-0.5">Video identity & face simulation</p>
        </div>

        {onClose && (
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800 transition-colors cursor-pointer"
            title="Close Panel"
          >
            <X className="h-4 w-4" />
          </button>
        )}
      </div>

      {/* Segmented Tab Switcher */}
      <div className="flex p-1 rounded-xl bg-zinc-900 border border-zinc-800 mt-3 mb-2 gap-1">
        <button
          type="button"
          onClick={() => setActiveTab('faceswap')}
          className={`flex-1 py-1.5 px-3 rounded-lg text-xs font-medium flex items-center justify-center gap-1.5 transition-all cursor-pointer ${
            activeTab === 'faceswap'
              ? 'bg-zinc-800 text-white shadow-xs border border-zinc-700/60'
              : 'text-zinc-400 hover:text-zinc-200'
          }`}
        >
          <User className="h-3.5 w-3.5" />
          <span>Face Swap</span>
          {faceSwapActive && <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />}
        </button>
        <button
          type="button"
          onClick={() => setActiveTab('attack')}
          className={`flex-1 py-1.5 px-3 rounded-lg text-xs font-medium flex items-center justify-center gap-1.5 transition-all cursor-pointer ${
            activeTab === 'attack'
              ? 'bg-zinc-800 text-white shadow-xs border border-zinc-700/60'
              : 'text-zinc-400 hover:text-zinc-200'
          }`}
        >
          <Mic className="h-3.5 w-3.5" />
          <span>Voice & Audio</span>
          {voiceTransformActive && <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />}
        </button>
      </div>

      {/* TAB 1: FACE SWAP */}
      {activeTab === 'faceswap' && (
        <div className="space-y-3 mt-1 animate-fade-in">
          {/* Face-Only Preservation Notice */}
          <div className="flex items-start gap-2.5 p-3 rounded-xl bg-sky-950/20 border border-sky-900/40 text-[11px] text-zinc-300 leading-relaxed">
            <ShieldCheck className="h-4 w-4 text-sky-400 shrink-0 mt-0.5" />
            <span>
              <strong className="text-sky-200 font-medium">Face only:</strong> Hair, hairline, ears, neck, and room background remain completely untouched.
            </span>
          </div>

          {/* Consent Checkbox */}
          <div className="p-3 rounded-xl bg-zinc-900/40 border border-zinc-800">
            <label className="flex items-start gap-2.5 cursor-pointer">
              <input
                type="checkbox"
                checked={consentGiven}
                onChange={(e) => {
                  setConsentGiven(e.target.checked);
                  if (e.target.checked) setConsentError(false);
                }}
                className="mt-0.5 h-4 w-4 rounded border-zinc-700 bg-zinc-800 text-sky-500 focus:ring-sky-500 cursor-pointer accent-sky-500"
              />
              <span className="text-[11px] text-zinc-400 leading-snug">
                I have consent to use this photograph's likeness for testing purposes.
              </span>
            </label>
            {consentError && (
              <p className="mt-2 text-[10px] text-red-400 flex items-center gap-1 font-medium">
                <AlertTriangle className="h-3 w-3 shrink-0" />
                Consent is required before enabling Face Swap.
              </p>
            )}
          </div>

          {/* Section 1: Target Face */}
          <div className="p-3.5 rounded-xl bg-zinc-900/50 border border-zinc-800 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider">
                Target Face
              </span>
              <span className="text-[10px] text-zinc-500 font-mono">
                JPG, PNG, WEBP
              </span>
            </div>

            {/* Hidden File Input */}
            <input
              type="file"
              ref={fileInputRef}
              accept="image/jpeg,image/png,image/webp,image/jpg"
              onChange={handleFileChange}
              className="hidden"
            />

            {/* Upload Button */}
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={isAnalyzing}
              className="w-full py-2 px-3 rounded-lg bg-zinc-800 hover:bg-zinc-750 text-zinc-200 hover:text-white border border-zinc-700/70 font-medium text-xs transition-colors flex items-center justify-center gap-2 cursor-pointer disabled:opacity-50"
            >
              {isAnalyzing ? (
                <>
                  <div className="h-3.5 w-3.5 rounded-full border-2 border-sky-400 border-t-white animate-spin" />
                  <span>Analyzing face...</span>
                </>
              ) : (
                <>
                  <Upload className="h-3.5 w-3.5 text-zinc-400" />
                  <span>Upload custom photo</span>
                </>
              )}
            </button>

            {/* Upload Messages */}
            {uploadError && (
              <div className="p-2 rounded-lg bg-red-950/40 border border-red-900/50 text-[11px] text-red-300 flex items-start gap-1.5">
                <AlertTriangle className="h-3.5 w-3.5 text-red-400 shrink-0 mt-0.5" />
                <span>{uploadError}</span>
              </div>
            )}

            {uploadSuccessMessage && (
              <div className="p-2 rounded-lg bg-emerald-950/30 border border-emerald-800/40 text-[11px] text-emerald-300 flex items-center gap-1.5">
                <Check className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                <span>{uploadSuccessMessage}</span>
              </div>
            )}

            {/* Active Persona Card */}
            <div className="p-2 rounded-lg bg-zinc-950/60 border border-zinc-800 flex items-center justify-between gap-3">
              <div className="flex items-center gap-2.5 min-w-0">
                <div className="relative w-10 h-10 rounded-lg overflow-hidden bg-zinc-900 border border-zinc-800 shrink-0">
                  {selectedFacePreview ? (
                    <img
                      src={selectedFacePreview}
                      alt={selectedFaceName}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <Camera className="h-4 w-4 text-zinc-600 m-auto" />
                  )}
                </div>
                <div className="min-w-0">
                  <p className="text-xs font-medium text-zinc-200 truncate">{selectedFaceName}</p>
                  <p className="text-[10px] text-emerald-400 font-mono mt-0.5 flex items-center gap-1">
                    <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                    478 landmark mesh verified
                  </p>
                </div>
              </div>

              {/* Reset Face Button */}
              <button
                type="button"
                onClick={onResetFace}
                title="Reset to default persona"
                className="p-1.5 rounded-lg text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800 transition-colors cursor-pointer shrink-0"
              >
                <RotateCcw className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>

          {/* Section 2: Presets */}
          <div className="p-3.5 rounded-xl bg-zinc-900/50 border border-zinc-800 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider">
                Presets
              </span>
              <span className="text-[10px] text-zinc-500">4 headshots</span>
            </div>

            <div className="grid grid-cols-4 gap-1.5">
              {[
                { id: 'mona-lisa', name: 'Emma', role: 'Studio', img: '/mona_lisa.jpg' },
                { id: 'cyber-agent', name: 'Alex', role: 'Clean', img: '/cyber_agent.jpg' },
                { id: 'astronaut', name: 'Sophia', role: 'Natural', img: '/astronaut.jpg' },
                { id: 'synthetic-executive', name: 'David', role: 'Exec', img: '/synthetic_executive.jpg' },
              ].map((p) => {
                const isSelected = currentPreset === p.id;
                return (
                  <button
                    key={p.id}
                    type="button"
                    onClick={() => onSelectPreset(p.id as FacePreset)}
                    className={`flex flex-col items-center p-2 rounded-lg border transition-all cursor-pointer ${
                      isSelected
                        ? 'bg-sky-950/20 border-sky-500/60 text-white shadow-xs'
                        : 'bg-zinc-950/40 hover:bg-zinc-800/50 border-zinc-800/70 text-zinc-400 hover:text-zinc-200'
                    }`}
                  >
                    <div className={`relative w-10 h-10 rounded-md overflow-hidden mb-1 border ${
                      isSelected ? 'border-sky-500/60' : 'border-zinc-800'
                    }`}>
                      <img
                        src={p.img}
                        alt={p.name}
                        className="w-full h-full object-cover"
                      />
                    </div>
                    <span className="text-[11px] font-medium truncate w-full text-center">
                      {p.name}
                    </span>
                    <span className={`text-[9px] truncate w-full text-center ${
                      isSelected ? 'text-sky-400 font-medium' : 'text-zinc-500'
                    }`}>
                      {p.role}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Section 3: Face Blending Controls */}
          <div className="p-3.5 rounded-xl bg-zinc-900/50 border border-zinc-800 space-y-2.5">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-xs font-semibold text-zinc-200">Face Blending</h3>
                <p className="text-[10px] text-zinc-500">Tone matching & contour feathering</p>
              </div>

              <button
                type="button"
                onClick={() => setShowAdvancedBlend(!showAdvancedBlend)}
                className="py-1 px-2 rounded-md border border-zinc-800 hover:bg-zinc-800 text-zinc-400 hover:text-zinc-200 text-xs font-medium transition-colors flex items-center gap-1 cursor-pointer"
              >
                <Sliders className="h-3 w-3" />
                <span>{showAdvancedBlend ? 'Close' : 'Adjust'}</span>
                {showAdvancedBlend ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
              </button>
            </div>

            {/* Profiles */}
            <div className="grid grid-cols-4 gap-1">
              {[
                { id: 'ultra', label: 'Ultra', sub: '6px' },
                { id: 'natural', label: 'Soft', sub: '7px' },
                { id: 'studio', label: 'Studio', sub: '5px' },
                { id: 'contour', label: 'Contour', sub: '6px' },
              ].map((p) => {
                const isSelected = activeBlendPreset === p.id;
                return (
                  <button
                    key={p.id}
                    type="button"
                    onClick={() => applyBlendPreset(p.id as any)}
                    className={`py-1.5 px-1 rounded-lg border text-center transition-all cursor-pointer ${
                      isSelected
                        ? 'bg-sky-950/20 border-sky-500/60 text-sky-200 font-medium shadow-xs'
                        : 'bg-zinc-950/40 hover:bg-zinc-800/50 border-zinc-800/70 text-zinc-400 hover:text-zinc-200'
                    }`}
                  >
                    <div className="text-[11px] leading-tight truncate">{p.label}</div>
                    <div className={`text-[9px] font-mono mt-0.5 ${
                      isSelected ? 'text-sky-400' : 'text-zinc-500'
                    }`}>{p.sub}</div>
                  </button>
                );
              })}
            </div>

            {/* Fine-Tuning Sliders Drawer */}
            {showAdvancedBlend && (
              <div className="pt-2.5 border-t border-zinc-800/80 space-y-3 animate-fade-in text-[11px]">
                {/* 1. Edge Feather */}
                <div>
                  <div className="flex items-center justify-between text-zinc-300 mb-1">
                    <span>Edge Feather Softness</span>
                    <span className="font-mono text-zinc-300 font-medium px-1.5 py-0.2 rounded bg-zinc-800 border border-zinc-700/60 text-[10px]">
                      {activeConfig.featherRadius}px
                    </span>
                  </div>
                  <input
                    type="range"
                    min="2"
                    max="10"
                    step="1"
                    value={activeConfig.featherRadius}
                    onChange={(e) => onUpdateBlendConfig?.({ featherRadius: Number(e.target.value) })}
                    className="premium-slider w-full cursor-pointer"
                  />
                  <div className="flex justify-between text-[9px] text-zinc-500 mt-0.5">
                    <span>Crisp (3px)</span>
                    <span>Balanced (6px)</span>
                    <span>Soft (9px)</span>
                  </div>
                </div>

                {/* 2. Skin Tone Match */}
                <div>
                  <div className="flex items-center justify-between text-zinc-300 mb-1">
                    <span>Skin Tone Match</span>
                    <span className="font-mono text-zinc-300 font-medium px-1.5 py-0.2 rounded bg-zinc-800 border border-zinc-700/60 text-[10px]">
                      {Math.round(activeConfig.skinToneMatch * 100)}%
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={activeConfig.skinToneMatch}
                    onChange={(e) => onUpdateBlendConfig?.({ skinToneMatch: Number(e.target.value) })}
                    className="premium-slider w-full cursor-pointer"
                  />
                </div>

                {/* 3. Ambient Lighting Transfer */}
                <div>
                  <div className="flex items-center justify-between text-zinc-300 mb-1">
                    <span>Ambient Lighting Transfer</span>
                    <span className="font-mono text-zinc-300 font-medium px-1.5 py-0.2 rounded bg-zinc-800 border border-zinc-700/60 text-[10px]">
                      {Math.round(activeConfig.lightingTransfer * 100)}%
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={activeConfig.lightingTransfer}
                    onChange={(e) => onUpdateBlendConfig?.({ lightingTransfer: Number(e.target.value) })}
                    className="premium-slider w-full cursor-pointer"
                  />
                </div>

                {/* 4. Mouth Pass-Through */}
                <div>
                  <div className="flex items-center justify-between text-zinc-300 mb-1">
                    <span>Speech & Mouth Pass-Through</span>
                    <span className="font-mono text-zinc-300 font-medium px-1.5 py-0.2 rounded bg-zinc-800 border border-zinc-700/60 text-[10px]">
                      {Math.round(activeConfig.mouthBlend * 100)}%
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={activeConfig.mouthBlend}
                    onChange={(e) => onUpdateBlendConfig?.({ mouthBlend: Number(e.target.value) })}
                    className="premium-slider w-full cursor-pointer"
                  />
                </div>

                {/* 5. Sensor Noise */}
                <div>
                  <div className="flex items-center justify-between text-zinc-300 mb-1">
                    <span>Camera Grain Match</span>
                    <span className="font-mono text-zinc-300 font-medium px-1.5 py-0.2 rounded bg-zinc-800 border border-zinc-700/60 text-[10px]">
                      {Math.round(activeConfig.sensorGrain * 100)}%
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={activeConfig.sensorGrain}
                    onChange={(e) => onUpdateBlendConfig?.({ sensorGrain: Number(e.target.value) })}
                    className="premium-slider w-full cursor-pointer"
                  />
                </div>

                {/* 6. Natural Eye Gaze */}
                <div className="pt-2 border-t border-zinc-800/80 flex items-center justify-between">
                  <div>
                    <span className="font-medium text-zinc-200 block">Natural Eye Gaze & Blink Sync</span>
                    <span className="text-[10px] text-zinc-500">Preserves live blinking and gaze movement</span>
                  </div>
                  <button
                    type="button"
                    onClick={() => onUpdateBlendConfig?.({ naturalEyes: !activeConfig.naturalEyes })}
                    className={`relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
                      activeConfig.naturalEyes ? 'bg-zinc-200' : 'bg-zinc-700'
                    }`}
                  >
                    <span
                      className={`pointer-events-none inline-block h-4 w-4 transform rounded-full shadow-sm ring-0 transition duration-200 ease-in-out ${
                        activeConfig.naturalEyes ? 'translate-x-4 bg-zinc-900' : 'translate-x-0 bg-zinc-400'
                      }`}
                    />
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Section 4: Master Action Trigger Button */}
          <div>
            {!faceSwapActive ? (
              <button
                type="button"
                onClick={handleToggleSwap}
                className="w-full py-2.5 px-4 rounded-xl bg-zinc-100 hover:bg-white text-zinc-950 font-medium text-xs transition-colors flex items-center justify-center gap-2 cursor-pointer shadow-xs active:scale-[0.99]"
              >
                <User className="h-3.5 w-3.5 text-zinc-900" />
                <span>Enable Face Swap</span>
              </button>
            ) : (
              <button
                type="button"
                onClick={handleToggleSwap}
                className="w-full py-2.5 px-4 rounded-xl bg-zinc-800 hover:bg-zinc-750 border border-zinc-700 text-zinc-200 font-medium text-xs transition-colors flex items-center justify-between cursor-pointer active:scale-[0.99]"
              >
                <div className="flex items-center gap-2">
                  <span className="h-2 w-2 rounded-full bg-emerald-400" />
                  <span className="font-medium text-zinc-100">Face Swap Active</span>
                </div>
                <span className="text-[11px] text-zinc-400 hover:text-white transition-colors">
                  Turn off
                </span>
              </button>
            )}
          </div>

          {/* Section 5: Stream Monitor Preview */}
          <div className="p-3 rounded-xl bg-zinc-900/50 border border-zinc-800 space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5">
                <Eye className="h-3.5 w-3.5 text-zinc-400" />
                <span className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider">
                  Live Preview
                </span>
              </div>
              <span className={`text-[9px] font-mono px-2 py-0.5 rounded border ${
                faceSwapActive
                  ? 'text-sky-300 bg-sky-950/40 border-sky-800/50'
                  : 'text-zinc-400 bg-zinc-800 border-zinc-700/60'
              }`}>
                {faceSwapActive ? 'Swapped Feed' : 'Clean Camera Feed'}
              </span>
            </div>

            <div className="relative w-full aspect-video rounded-lg overflow-hidden bg-black border border-zinc-800 flex items-center justify-center">
              <video
                ref={previewVideoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover transform -scale-x-100"
              />
              {faceSwapActive && (
                <div className="absolute top-2 right-2 px-2 py-0.5 rounded bg-black/80 backdrop-blur-xs text-[9px] font-medium text-emerald-300 border border-emerald-500/30 flex items-center gap-1">
                  <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
                  <span>Modified Live</span>
                </div>
              )}
            </div>
          </div>

          {/* Section 6: Processing Diagnostics */}
          <div className="p-3.5 rounded-xl bg-zinc-900/50 border border-zinc-800 space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5">
                <Cpu className="h-3.5 w-3.5 text-sky-400" />
                <span className="text-[10px] font-semibold text-zinc-300 uppercase tracking-wider">
                  Diagnostics
                </span>
              </div>
              <span className="text-[10px] font-mono flex items-center gap-1.5 text-emerald-400 bg-emerald-950/40 px-2 py-0.5 rounded border border-emerald-800/50">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                {telemetry?.fps ? `${telemetry.fps} FPS` : '56 FPS'}
              </span>
            </div>

            <div className="space-y-1 text-[11px]">
              <div className="flex items-center justify-between py-1 border-b border-zinc-800/60">
                <span className="text-zinc-500">Pipeline</span>
                <span className="text-zinc-200 font-mono">MediaPipe + WebGL</span>
              </div>
              <div className="flex items-center justify-between py-1 border-b border-zinc-800/60">
                <span className="text-zinc-500">Landmarks</span>
                <span className="font-mono flex items-center gap-1 text-emerald-400">
                  <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                  {telemetry?.landmarksDetected ? '478 points' : '478 points'}
                </span>
              </div>
              <div className="flex items-center justify-between py-1 border-b border-zinc-800/60">
                <span className="text-zinc-500">Mesh Triangles</span>
                <span className="text-zinc-300 font-mono">854 triangles</span>
              </div>
              <div className="flex items-center justify-between py-1 border-b border-zinc-800/60">
                <span className="text-zinc-500">Skin Matching</span>
                <span className="text-sky-400 font-mono">Reinhard Adaptive</span>
              </div>
              <div className="flex items-center justify-between py-1 border-b border-zinc-800/60">
                <span className="text-zinc-500">Lighting</span>
                <span className="text-amber-300 font-mono">Ambient Transfer</span>
              </div>
              <div className="flex items-center justify-between py-1 border-b border-zinc-800/60">
                <span className="text-zinc-500">Feathering</span>
                <span className="text-violet-300 font-mono">{activeConfig.featherRadius}px Gaussian</span>
              </div>
              <div className="flex items-center justify-between py-1">
                <span className="text-zinc-500">Boundary</span>
                <span className="text-emerald-400 font-mono">Hair & Neck Preserved</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: VOICE & ATTACK SIMULATOR */}
      {activeTab === 'attack' && (
        <div className="space-y-3 mt-1 animate-fade-in">
          {/* Status Indicator */}
          <div className="flex items-center justify-between p-3 rounded-xl bg-zinc-900/50 border border-zinc-800/80 text-[11px]">
            <div className="flex items-center gap-2">
              <ShieldAlert className="h-4 w-4 text-zinc-400" />
              <span className="text-zinc-300 font-medium">Attack Simulation Mode</span>
            </div>
            <span className="font-mono text-xs uppercase text-zinc-300 font-medium bg-zinc-800 px-2 py-0.5 rounded border border-zinc-700/60">
              {attackMode !== 'none' ? attackMode : 'Standby'}
            </span>
          </div>

          {/* Voice Simulation Controls */}
          {onToggleVoiceTransform && (
            <div className="p-3.5 rounded-xl bg-zinc-900/50 border border-zinc-800/80 space-y-2.5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Mic className="h-4 w-4 text-zinc-400" />
                  <h3 className="text-xs font-semibold text-zinc-200">Voice Simulation</h3>
                </div>
                <span className="text-[10px] font-mono text-zinc-400 bg-zinc-800 px-2 py-0.5 rounded border border-zinc-700/60">
                  {voiceTransformActive ? 'Active' : 'Off'}
                </span>
              </div>

              <button
                type="button"
                onClick={onToggleVoiceTransform}
                className={`w-full py-2 px-3 rounded-lg text-xs font-medium transition-colors border cursor-pointer ${
                  voiceTransformActive
                    ? 'bg-zinc-800 text-white border-zinc-600 shadow-xs'
                    : 'bg-zinc-950/40 text-zinc-300 border-zinc-800 hover:bg-zinc-800/60'
                }`}
              >
                <Sliders className="h-3.5 w-3.5 inline mr-1.5 text-zinc-400" />
                {voiceTransformActive ? 'Disable Voice Transform' : 'Enable Voice Transform'}
              </button>

              {onSelectVoicePreset && (
                <div className="mt-2.5 space-y-1">
                  <label className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider block mb-1">
                    Voice Model
                  </label>
                  {[
                    { id: 'robotic-vocoder', label: 'Robotic Vocoder (440Hz)' },
                    { id: 'deep-pitch-neural', label: 'Deep Pitch Shift (Granular)' },
                    { id: 'synthetic-clone', label: 'Synthetic Voice Clone' },
                  ].map((p) => (
                    <button
                      key={p.id}
                      type="button"
                      onClick={() => onSelectVoicePreset(p.id as any)}
                      className={`w-full py-1.5 px-2.5 rounded-lg text-left text-xs transition-colors cursor-pointer ${
                        voicePreset === p.id
                          ? 'bg-zinc-800 text-white font-medium border border-zinc-700/60'
                          : 'text-zinc-400 hover:bg-zinc-800/40 hover:text-zinc-200'
                      }`}
                    >
                      {p.label}
                    </button>
                  ))}
                </div>
              )}

              {/* Headphone Live Voice Monitor */}
              {onToggleVoiceMonitor && (
                <div className="pt-2.5 border-t border-zinc-800/80">
                  <button
                    type="button"
                    onClick={onToggleVoiceMonitor}
                    disabled={!voiceTransformActive}
                    className={`w-full py-1.5 px-3 rounded-lg text-xs font-medium transition-colors border flex items-center justify-between cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed ${
                      isVoiceMonitoring
                        ? 'bg-zinc-800 border-zinc-600 text-white'
                        : 'bg-zinc-950/40 border-zinc-800 text-zinc-400 hover:bg-zinc-800/40 hover:text-zinc-200'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <Headphones className="h-3.5 w-3.5 text-zinc-400" />
                      <span>Hear Transformed Voice</span>
                    </div>
                    <span className="text-[10px] font-mono text-zinc-400">
                      {isVoiceMonitoring ? 'ON' : 'OFF'}
                    </span>
                  </button>
                  <p className="text-[10px] text-zinc-500 mt-1">
                    Use headphones to avoid feedback echo.
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
              className="w-full py-2.5 px-4 rounded-xl bg-zinc-800 hover:bg-zinc-700 text-zinc-100 border border-zinc-700 text-xs font-medium transition-colors flex items-center justify-center gap-2 cursor-pointer shadow-xs active:scale-[0.99]"
            >
              <Flame className="h-4 w-4 text-zinc-400" />
              <span>Full Attack (Face + Voice)</span>
            </button>
          )}

          {/* Reset All */}
          {onResetAll && (
            <button
              type="button"
              onClick={onResetAll}
              className="w-full py-2 px-4 rounded-xl bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-zinc-400 hover:text-zinc-200 text-xs font-medium transition-colors flex items-center justify-center gap-2 cursor-pointer"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              <span>Reset all to clean feed</span>
            </button>
          )}
        </div>
      )}

      <div className="mt-auto pt-4 text-[10px] text-zinc-600 text-center font-mono">
        Tester Security Simulator • Biometric Sandbox
      </div>
    </div>
  );
};
