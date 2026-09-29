export type DetectionStatus = 'idle' | 'listening' | 'analyzing' | 'human' | 'suspicious' | 'deepfake';

export interface VoiceAcousticMetrics {
  rmsEnergy: number; // 0.0 - 1.0 (speech activity)
  spectralCentroidHz: number; // Hz (center frequency)
  spectralRolloffHz: number; // 85% energy cutoff
  spectralFlatness: number; // 0.0 - 1.0 (noise vs tone)
  harmonicPeakRatio: number; // ratio of max peak to median
  carrierHarmonicDetected: boolean; // vocoder carrier detected
  bandpassResonanceDetected: boolean; // narrow peak around 1.2-2.4kHz
  jitterVariance: number; // pitch stability
}

export interface VoiceDetectionState {
  isAnalyzing: boolean;
  hasAudio: boolean;
  isVoiceActive: boolean;
  anomalyScore: number; // 0 - 100
  confidence: number; // 0 - 100
  status: DetectionStatus;
  metrics: VoiceAcousticMetrics;
  detectedAnomalies: string[];
  frequencyData: Uint8Array;
  timeDomainData: Uint8Array;
  timestamp: number;
}
