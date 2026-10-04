import React, { useState, useEffect, useRef } from 'react';
import {
  Upload,
  Cpu,
  ArrowRight,
  AlertTriangle,
  CheckCircle,
  RefreshCw,
  Clock,
  Layers,
  Sparkles,
  ArrowLeft,
  Activity,
  Image as ImageIcon,
  ShieldCheck,
  Check
} from 'lucide-react';

interface InferenceMetadata {
  model: string;
  device: string;
  gpu: string;
  latency_ms: number;
  network_latency_ms: number;
  landmark_error: number | null;
  face_detected: boolean;
  mask_mean: number;
  identity_gain: number;
  A_cosine_source_composite: number;
  B_cosine_target_composite: number;
  C_cosine_source_target: number;
  D_cosine_source_swap: number;
}

interface InferenceResponse {
  status: string;
  swapped_image: string;
  mask_image: string;
  aligned_source_image: string;
  comparison_grid_image: string;
  metadata: InferenceMetadata;
}

interface SamplePair {
  name: string;
  description: string;
  sourceUrl: string;
  targetUrl: string;
  expectedGain: string;
}

const PRESET_PAIRS: SamplePair[] = [
  {
    name: 'Pair 1',
    description: 'CelebA 197935 → 098180',
    sourceUrl: '/samples/celeba/197935.jpg',
    targetUrl: '/samples/celeba/098180.jpg',
    expectedGain: '+0.1177',
  },
  {
    name: 'Pair 2',
    description: 'CelebA 081968 → 037827',
    sourceUrl: '/samples/celeba/081968.jpg',
    targetUrl: '/samples/celeba/037827.jpg',
    expectedGain: '+0.1771',
  },
  {
    name: 'Pair 3',
    description: 'CelebA 202283 → 169194',
    sourceUrl: '/samples/celeba/202283.jpg',
    targetUrl: '/samples/celeba/169194.jpg',
    expectedGain: '+0.2154',
  },
];

interface Phase6GTestPanelProps {
  onBack?: () => void;
}

export const Phase6GTestPanel: React.FC<Phase6GTestPanelProps> = ({ onBack }) => {
  // Image files & previews
  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const [sourcePreview, setSourcePreview] = useState<string | null>(null);
  const [targetFile, setTargetFile] = useState<File | null>(null);
  const [targetPreview, setTargetPreview] = useState<string | null>(null);

  // Active preset label
  const [activePreset, setActivePreset] = useState<string | null>(null);

  // Execution state
  const [loading, setLoading] = useState(false);
  const [loadingPhase, setLoadingPhase] = useState<string>('');
  const [elapsedMs, setElapsedMs] = useState<number>(0);
  const [error, setError] = useState<{ message: string; status?: number } | null>(null);

  // Result state
  const [result, setResult] = useState<InferenceResponse | null>(null);
  const [browserRoundtripMs, setBrowserRoundtripMs] = useState<number | null>(null);

  // Service health
  const [serviceHealth, setServiceHealth] = useState<{ ready: boolean; gpu?: string; error?: string } | null>(null);

  const sourceInputRef = useRef<HTMLInputElement>(null);
  const targetInputRef = useRef<HTMLInputElement>(null);
  const timerRef = useRef<number | null>(null);

  // Check health on mount
  useEffect(() => {
    checkHealth();
  }, []);

  const checkHealth = async () => {
    try {
      const res = await fetch('/api/ml/health');
      if (res.ok) {
        const data = await res.json();
        setServiceHealth({ ready: data.ready, gpu: data.gpu });
      } else {
        setServiceHealth({ ready: false, error: `HTTP ${res.status}` });
      }
    } catch (err: any) {
      setServiceHealth({ ready: false, error: err.message || 'Connection failed' });
    }
  };

  // Helper to load sample image
  const loadPresetPair = async (pair: SamplePair) => {
    try {
      setError(null);
      setActivePreset(pair.name);

      const [srcBlob, tgtBlob] = await Promise.all([
        fetch(pair.sourceUrl).then((r) => r.blob()),
        fetch(pair.targetUrl).then((r) => r.blob()),
      ]);

      const srcName = pair.sourceUrl.split('/').pop() || 'source.jpg';
      const tgtName = pair.targetUrl.split('/').pop() || 'target.jpg';

      const sFile = new File([srcBlob], srcName, { type: 'image/jpeg' });
      const tFile = new File([tgtBlob], tgtName, { type: 'image/jpeg' });

      setSourceFile(sFile);
      setSourcePreview(URL.createObjectURL(srcBlob));
      setTargetFile(tFile);
      setTargetPreview(URL.createObjectURL(tgtBlob));
    } catch (err: any) {
      setError({ message: `Failed to load sample pair: ${err.message}` });
    }
  };

  const handleSourceUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setActivePreset(null);
      setSourceFile(file);
      setSourcePreview(URL.createObjectURL(file));
      setError(null);
    }
  };

  const handleTargetUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setActivePreset(null);
      setTargetFile(file);
      setTargetPreview(URL.createObjectURL(file));
      setError(null);
    }
  };

  // Execute inference via Express proxy
  const handleRunInference = async () => {
    if (!sourceFile || !targetFile) {
      setError({ message: 'Please select both source and target images.' });
      return;
    }

    setLoading(true);
    setLoadingPhase('Transmitting to Express gateway...');
    setError(null);
    setResult(null);
    setBrowserRoundtripMs(null);

    const tStart = performance.now();
    setElapsedMs(0);
    timerRef.current = window.setInterval(() => {
      setElapsedMs(Math.round(performance.now() - tStart));
    }, 50);

    try {
      const formData = new FormData();
      formData.append('source', sourceFile, sourceFile.name);
      formData.append('target', targetFile, targetFile.name);

      setLoadingPhase('Running Phase 6G CUDA neural pipeline on RTX 5060...');

      const response = await fetch('/api/ml/face-swap', {
        method: 'POST',
        body: formData,
      });

      const roundtrip = Math.round(performance.now() - tStart);
      setBrowserRoundtripMs(roundtrip);

      if (response.status === 429) {
        setError({
          status: 429,
          message: 'GPU inference busy — please wait and try again.',
        });
        return;
      }

      if (response.status === 503) {
        setError({
          status: 503,
          message: 'ML service unavailable. Ensure Python FastAPI service is running on 127.0.0.1:8000.',
        });
        return;
      }

      if (!response.ok) {
        let errDetail = `HTTP ${response.status} error`;
        try {
          const errJson = await response.json();
          errDetail = errJson.error || errJson.detail || errDetail;
        } catch {
          // ignore
        }
        setError({
          status: response.status,
          message: errDetail,
        });
        return;
      }

      const data: InferenceResponse = await response.json();
      setResult(data);
    } catch (err: any) {
      setError({
        message: `Network error reaching Express: ${err.message || 'Failed to fetch'}`,
      });
    } finally {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
      setLoading(false);
      setLoadingPhase('');
    }
  };

  return (
    <div className="min-h-[100dvh] w-full bg-[#f8fafc] text-slate-900 flex flex-col font-sans animate-fade-in">
      {/* Top Header */}
      <header className="w-full border-b border-slate-200/80 bg-white sticky top-0 z-30">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            {onBack && (
              <button
                onClick={onBack}
                className="p-2 rounded-xl text-slate-500 hover:text-slate-800 hover:bg-slate-100 transition-colors cursor-pointer"
                title="Back to Dashboard"
              >
                <ArrowLeft className="h-4 w-4" />
              </button>
            )}
            <div className="flex items-center gap-2">
              <div className="h-8 w-8 rounded-lg bg-blue-600 flex items-center justify-center text-white shadow-xs">
                <Cpu className="h-4 w-4" />
              </div>
              <div>
                <h1 className="text-base sm:text-lg font-bold text-slate-900 leading-tight">
                  Phase 6G Neural Face Swap Test
                </h1>
                <p className="text-[11px] text-slate-500 hidden sm:block">
                  Express Gateway (:5001) → FastAPI (:8000) → RTX 5060 CUDA
                </p>
              </div>
            </div>
          </div>

          {/* Health Badge */}
          <div className="flex items-center gap-2">
            {serviceHealth?.ready ? (
              <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-50 border border-emerald-200 text-emerald-700 text-xs font-medium">
                <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
                <span className="hidden sm:inline">{serviceHealth.gpu || 'RTX 5060'} Ready</span>
                <span className="sm:hidden">Online</span>
              </div>
            ) : (
              <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-amber-50 border border-amber-200 text-amber-700 text-xs font-medium">
                <span className="h-2 w-2 rounded-full bg-amber-500" />
                <span>ML Offline ({serviceHealth?.error || 'Checking...'})</span>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main Body */}
      <main className="flex-1 w-full max-w-6xl mx-auto px-4 sm:px-6 py-6 sm:py-8 space-y-6">
        {/* Preset Selector Card */}
        <div className="p-4 sm:p-5 rounded-2xl bg-white border border-slate-200/80 shadow-2xs space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs sm:text-sm font-semibold text-slate-800">
              <Sparkles className="h-4 w-4 text-blue-600" />
              <span>Genuine CelebA Cross-ID Test Pairs:</span>
            </div>
            <span className="text-[11px] text-slate-400">1-Click Fast Validation</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
            {PRESET_PAIRS.map((pair) => (
              <button
                key={pair.name}
                type="button"
                onClick={() => loadPresetPair(pair)}
                className={`p-3 rounded-xl border text-left transition-all cursor-pointer flex flex-col justify-between ${
                  activePreset === pair.name
                    ? 'border-blue-500 bg-blue-50/60 ring-2 ring-blue-500/20'
                    : 'border-slate-200 hover:border-slate-300 hover:bg-slate-50'
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-bold text-slate-900">{pair.name}</span>
                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-blue-100/70 text-blue-700 font-semibold">
                    Exp: {pair.expectedGain}
                  </span>
                </div>
                <span className="text-[11px] text-slate-500 font-mono truncate">{pair.description}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Input Pair Selection Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6">
          {/* Source Image Selector */}
          <div className="p-5 rounded-2xl bg-white border border-slate-200/80 shadow-2xs flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-500">
                  1. Source Image (Identity)
                </span>
                {sourceFile && (
                  <span className="text-[11px] font-mono text-emerald-600 font-medium truncate max-w-[160px]">
                    {sourceFile.name}
                  </span>
                )}
              </div>

              {sourcePreview ? (
                <div className="relative aspect-square max-w-[220px] mx-auto rounded-xl overflow-hidden border border-slate-200 bg-slate-100 shadow-2xs group">
                  <img src={sourcePreview} alt="Source Preview" className="w-full h-full object-cover" />
                  <div className="absolute inset-0 bg-slate-900/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                    <button
                      type="button"
                      onClick={() => sourceInputRef.current?.click()}
                      className="px-3 py-1.5 rounded-lg bg-white/95 text-slate-800 text-xs font-medium shadow cursor-pointer"
                    >
                      Change Photo
                    </button>
                  </div>
                </div>
              ) : (
                <div
                  onClick={() => sourceInputRef.current?.click()}
                  className="aspect-square max-w-[220px] mx-auto rounded-xl border-2 border-dashed border-slate-200 hover:border-blue-400 bg-slate-50/50 hover:bg-blue-50/20 transition-all flex flex-col items-center justify-center p-4 text-center cursor-pointer"
                >
                  <Upload className="h-7 w-7 text-slate-400 mb-2" />
                  <span className="text-xs font-semibold text-slate-700 mb-0.5">Upload Source Face</span>
                  <span className="text-[11px] text-slate-400">JPG, PNG, or WEBP</span>
                </div>
              )}
            </div>

            <input
              ref={sourceInputRef}
              type="file"
              accept="image/png, image/jpeg, image/jpg, image/webp"
              onChange={handleSourceUpload}
              className="hidden"
            />

            <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
              <span>Extracts 512-D ArcFace embedding + Delaunay warp</span>
              <button
                type="button"
                onClick={() => sourceInputRef.current?.click()}
                className="text-blue-600 hover:underline font-medium cursor-pointer"
              >
                Browse...
              </button>
            </div>
          </div>

          {/* Target Image Selector */}
          <div className="p-5 rounded-2xl bg-white border border-slate-200/80 shadow-2xs flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-500">
                  2. Target Image (Pose & Context)
                </span>
                {targetFile && (
                  <span className="text-[11px] font-mono text-emerald-600 font-medium truncate max-w-[160px]">
                    {targetFile.name}
                  </span>
                )}
              </div>

              {targetPreview ? (
                <div className="relative aspect-square max-w-[220px] mx-auto rounded-xl overflow-hidden border border-slate-200 bg-slate-100 shadow-2xs group">
                  <img src={targetPreview} alt="Target Preview" className="w-full h-full object-cover" />
                  <div className="absolute inset-0 bg-slate-900/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                    <button
                      type="button"
                      onClick={() => targetInputRef.current?.click()}
                      className="px-3 py-1.5 rounded-lg bg-white/95 text-slate-800 text-xs font-medium shadow cursor-pointer"
                    >
                      Change Photo
                    </button>
                  </div>
                </div>
              ) : (
                <div
                  onClick={() => targetInputRef.current?.click()}
                  className="aspect-square max-w-[220px] mx-auto rounded-xl border-2 border-dashed border-slate-200 hover:border-blue-400 bg-slate-50/50 hover:bg-blue-50/20 transition-all flex flex-col items-center justify-center p-4 text-center cursor-pointer"
                >
                  <Upload className="h-7 w-7 text-slate-400 mb-2" />
                  <span className="text-xs font-semibold text-slate-700 mb-0.5">Upload Target Face</span>
                  <span className="text-[11px] text-slate-400">JPG, PNG, or WEBP</span>
                </div>
              )}
            </div>

            <input
              ref={targetInputRef}
              type="file"
              accept="image/png, image/jpeg, image/jpg, image/webp"
              onChange={handleTargetUpload}
              className="hidden"
            />

            <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
              <span>Provides MediaPipe landmarks & background</span>
              <button
                type="button"
                onClick={() => targetInputRef.current?.click()}
                className="text-blue-600 hover:underline font-medium cursor-pointer"
              >
                Browse...
              </button>
            </div>
          </div>
        </div>

        {/* Action Button & Execution Status */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4 p-4 sm:p-5 rounded-2xl bg-white border border-slate-200/80 shadow-2xs">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-xl bg-slate-100 flex items-center justify-center text-slate-600 shrink-0">
              <Activity className="h-5 w-5" />
            </div>
            <div>
              <div className="text-xs sm:text-sm font-bold text-slate-900">
                Execute Phase 6G Neural Swap
              </div>
              <div className="text-[11px] text-slate-500">
                POST /api/ml/face-swap → FastAPI :8000 → PyTorch CUDA
              </div>
            </div>
          </div>

          <button
            type="button"
            disabled={!sourceFile || !targetFile || loading}
            onClick={handleRunInference}
            className="w-full sm:w-auto px-6 py-3 rounded-xl bg-blue-600 hover:bg-blue-700 disabled:opacity-40 text-white font-semibold text-xs sm:text-sm shadow-xs transition-all flex items-center justify-center gap-2 cursor-pointer disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <RefreshCw className="h-4 w-4 animate-spin" />
                <span>Running Test ({elapsedMs} ms)...</span>
              </>
            ) : (
              <>
                <span>Run Phase 6G Test</span>
                <ArrowRight className="h-4 w-4" />
              </>
            )}
          </button>
        </div>

        {/* Loading Indicator Banner */}
        {loading && (
          <div className="p-4 rounded-xl bg-blue-50/80 border border-blue-200 text-blue-900 text-xs sm:text-sm flex items-center gap-3 animate-pulse">
            <RefreshCw className="h-5 w-5 text-blue-600 animate-spin shrink-0" />
            <div className="flex-1">
              <span className="font-semibold">{loadingPhase}</span>
              <span className="ml-2 font-mono text-blue-600">[{elapsedMs} ms elapsed]</span>
            </div>
          </div>
        )}

        {/* Error Alert Banner */}
        {error && (
          <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-900 text-xs sm:text-sm flex items-start gap-3 animate-fade-in">
            <AlertTriangle className="h-5 w-5 text-rose-600 shrink-0 mt-0.5" />
            <div className="flex-1">
              <div className="font-bold flex items-center gap-2">
                <span>Inference Error</span>
                {error.status && (
                  <span className="px-1.5 py-0.2 rounded bg-rose-200 text-rose-800 font-mono text-[10px]">
                    HTTP {error.status}
                  </span>
                )}
              </div>
              <p className="mt-0.5 text-xs text-rose-700 leading-relaxed font-medium">
                {error.message}
              </p>
            </div>
          </div>
        )}

        {/* Inference Results Section */}
        {result && (
          <div className="space-y-6 animate-fade-in-up">
            {/* Visual Images Grid */}
            <div className="p-5 rounded-2xl bg-white border border-slate-200/80 shadow-2xs space-y-4">
              <div className="flex items-center justify-between pb-3 border-b border-slate-100">
                <div className="flex items-center gap-2">
                  <Layers className="h-4 w-4 text-blue-600" />
                  <span className="text-sm font-bold text-slate-900">Phase 6G Generated Visual Outputs</span>
                </div>
                <div className="flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-emerald-50 border border-emerald-200 text-emerald-700 text-xs font-semibold">
                  <Check className="h-3 w-3 stroke-[3]" />
                  <span>Real Phase 6G Inference</span>
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                {/* 1. Swapped Composite */}
                <div className="flex flex-col items-center p-3 rounded-xl bg-slate-50 border border-slate-200/80">
                  <span className="text-[11px] font-bold text-slate-700 uppercase tracking-wider mb-2 text-center">
                    Composite Output (I_comp)
                  </span>
                  <div className="aspect-square w-full rounded-lg overflow-hidden border border-slate-300 bg-white shadow-2xs">
                    <img
                      src={result.swapped_image}
                      alt="Composite Face Swap"
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <span className="text-[10px] text-slate-400 mt-2 font-mono">128 × 128 (Composite)</span>
                </div>

                {/* 2. Predicted Mask */}
                <div className="flex flex-col items-center p-3 rounded-xl bg-slate-50 border border-slate-200/80">
                  <span className="text-[11px] font-bold text-slate-700 uppercase tracking-wider mb-2 text-center">
                    Predicted Mask (M_pred)
                  </span>
                  <div className="aspect-square w-full rounded-lg overflow-hidden border border-slate-300 bg-white shadow-2xs">
                    <img
                      src={result.mask_image}
                      alt="Predicted Blending Mask"
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <span className="text-[10px] text-slate-400 mt-2 font-mono">
                    Coverage: {(result.metadata.mask_mean * 100).toFixed(1)}%
                  </span>
                </div>

                {/* 3. 6D Aligned Source */}
                <div className="flex flex-col items-center p-3 rounded-xl bg-slate-50 border border-slate-200/80">
                  <span className="text-[11px] font-bold text-slate-700 uppercase tracking-wider mb-2 text-center">
                    6D Aligned Source (L_src)
                  </span>
                  <div className="aspect-square w-full rounded-lg overflow-hidden border border-slate-300 bg-white shadow-2xs">
                    <img
                      src={result.aligned_source_image}
                      alt="6D Aligned Source Warp"
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <span className="text-[10px] text-slate-400 mt-2 font-mono">Piecewise-Affine Remap</span>
                </div>

                {/* 4. Full Diagnostic Strip Preview */}
                <div className="flex flex-col items-center p-3 rounded-xl bg-slate-50 border border-slate-200/80">
                  <span className="text-[11px] font-bold text-slate-700 uppercase tracking-wider mb-2 text-center">
                    Comparison Grid Strip
                  </span>
                  <div className="aspect-square w-full rounded-lg overflow-hidden border border-slate-300 bg-white shadow-2xs flex items-center justify-center p-1">
                    <img
                      src={result.comparison_grid_image}
                      alt="Full 6-column comparison grid"
                      className="w-full h-auto object-contain"
                    />
                  </div>
                  <span className="text-[10px] text-slate-400 mt-2 font-mono">6-Channel Grid</span>
                </div>
              </div>
            </div>

            {/* Diagnostic Metrics Table Card */}
            <div className="p-5 rounded-2xl bg-white border border-slate-200/80 shadow-2xs space-y-4">
              <div className="flex items-center justify-between pb-3 border-b border-slate-100">
                <span className="text-sm font-bold text-slate-900">Diagnostic Performance & Identity Metrics</span>
                <span className="text-xs font-mono text-slate-500">
                  {result.metadata.model.toUpperCase()} • {result.metadata.device}
                </span>
              </div>

              {/* Highlight Stats Row */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {/* Identity Gain */}
                <div className="p-3.5 rounded-xl bg-blue-50/70 border border-blue-200/80 text-center">
                  <div className="text-[10px] uppercase font-bold tracking-wider text-blue-600 mb-0.5">
                    Identity Gain (A - C)
                  </div>
                  <div className="text-xl sm:text-2xl font-black text-blue-700 font-mono">
                    {result.metadata.identity_gain > 0 ? '+' : ''}
                    {result.metadata.identity_gain.toFixed(4)}
                  </div>
                  <div className="text-[10px] text-blue-500 mt-0.5">Source Transfer Gain</div>
                </div>

                {/* Browser Roundtrip */}
                <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 text-center">
                  <div className="text-[10px] uppercase font-bold tracking-wider text-slate-500 mb-0.5">
                    Browser Roundtrip
                  </div>
                  <div className="text-xl sm:text-2xl font-bold text-slate-800 font-mono">
                    {browserRoundtripMs} <span className="text-xs font-normal">ms</span>
                  </div>
                  <div className="text-[10px] text-slate-400 mt-0.5">React → Express → GPU → React</div>
                </div>

                {/* Pipeline Latency */}
                <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 text-center">
                  <div className="text-[10px] uppercase font-bold tracking-wider text-slate-500 mb-0.5">
                    Pipeline Latency
                  </div>
                  <div className="text-xl sm:text-2xl font-bold text-slate-800 font-mono">
                    {result.metadata.latency_ms.toFixed(1)} <span className="text-xs font-normal">ms</span>
                  </div>
                  <div className="text-[10px] text-slate-400 mt-0.5">
                    Net GPU: {result.metadata.network_latency_ms.toFixed(1)} ms
                  </div>
                </div>

                {/* Face Redetection */}
                <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 text-center">
                  <div className="text-[10px] uppercase font-bold tracking-wider text-slate-500 mb-0.5">
                    Face Redetection
                  </div>
                  <div className="text-lg sm:text-xl font-bold text-emerald-600 flex items-center justify-center gap-1">
                    <ShieldCheck className="h-5 w-5" />
                    <span>{result.metadata.face_detected ? '100% PASS' : 'FAIL'}</span>
                  </div>
                  <div className="text-[10px] text-slate-400 mt-0.5">
                    LM Error: {result.metadata.landmark_error?.toFixed(2) || 'N/A'} px
                  </div>
                </div>
              </div>

              {/* Full ArcFace Cosine Similarities Table */}
              <div className="mt-4 pt-3 border-t border-slate-100">
                <span className="text-xs font-bold text-slate-700 block mb-2.5">
                  ArcFace Cosine Similarity Decomposition:
                </span>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs font-mono">
                  <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200">
                    <span className="text-[10px] text-slate-500 block">A: Source ↔ Composite</span>
                    <span className="font-bold text-slate-900">
                      {result.metadata.A_cosine_source_composite.toFixed(4)}
                    </span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200">
                    <span className="text-[10px] text-slate-500 block">B: Target ↔ Composite</span>
                    <span className="font-bold text-slate-900">
                      {result.metadata.B_cosine_target_composite.toFixed(4)}
                    </span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200">
                    <span className="text-[10px] text-slate-500 block">C: Source ↔ Target</span>
                    <span className="font-bold text-slate-900">
                      {result.metadata.C_cosine_source_target.toFixed(4)}
                    </span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200">
                    <span className="text-[10px] text-slate-500 block">D: Source ↔ Raw Swap</span>
                    <span className="font-bold text-slate-900">
                      {result.metadata.D_cosine_source_swap.toFixed(4)}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};
