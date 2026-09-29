import { useEffect, useRef, useState } from 'react';
import { VoiceDetectionState } from '../types/detection';
import { VoiceDeepfakeDetector } from '../detection/voiceDeepfakeDetector';

const INITIAL_STATE: VoiceDetectionState = {
  isAnalyzing: false,
  hasAudio: false,
  isVoiceActive: false,
  anomalyScore: 0,
  confidence: 0,
  status: 'idle',
  metrics: {
    rmsEnergy: 0,
    spectralCentroidHz: 0,
    spectralRolloffHz: 0,
    spectralFlatness: 0,
    harmonicPeakRatio: 0,
    carrierHarmonicDetected: false,
    bandpassResonanceDetected: false,
    jitterVariance: 0,
  },
  detectedAnomalies: [],
  frequencyData: new Uint8Array(512),
  timeDomainData: new Uint8Array(512),
  timestamp: Date.now(),
};

export function useVoiceDetection(
  remoteStream: MediaStream | null,
  isPeerVoiceAttacking: boolean = false,
  enabled: boolean = true
) {
  const [detectionState, setDetectionState] = useState<VoiceDetectionState>(INITIAL_STATE);
  const detectorRef = useRef<VoiceDeepfakeDetector | null>(null);

  // Sync peer attack telemetry whenever it changes
  useEffect(() => {
    if (detectorRef.current) {
      detectorRef.current.setPeerAttackTelemetry(isPeerVoiceAttacking);
    }
  }, [isPeerVoiceAttacking]);

  useEffect(() => {
    if (!enabled || !remoteStream) {
      if (detectorRef.current) {
        detectorRef.current.stop();
        detectorRef.current = null;
      }
      setDetectionState(INITIAL_STATE);
      return;
    }

    const detector = new VoiceDeepfakeDetector((newState) => {
      setDetectionState(newState);
    });

    detector.setPeerAttackTelemetry(isPeerVoiceAttacking);
    detector.start(remoteStream);
    detectorRef.current = detector;

    return () => {
      detector.stop();
      detectorRef.current = null;
    };
  }, [remoteStream, enabled, isPeerVoiceAttacking]);

  return detectionState;
}
