import { useEffect,useRef,useState } from 'react';
import { VoiceDeepfakeDetector } from '../detection/voiceDeepfakeDetector';
import { VoiceDetectionState } from '../types/detection';

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

/**
 * Experimental acoustic anomaly monitoring of incoming audio.
 * Analyzes the incoming WebRTC MediaStream purely via digital signal processing
 * and hand-written acoustic rules; no trained classifier is used here.
 * Completely independent of any sender signals or telemetry.
 */
export function useVoiceDetection(
  remoteStream: MediaStream | null,
  enabled: boolean = true
) {
  const [detectionState, setDetectionState] = useState<VoiceDetectionState>(INITIAL_STATE);
  const detectorRef = useRef<VoiceDeepfakeDetector | null>(null);
  const [previousInput, setPreviousInput] = useState({ remoteStream, enabled });
  if (previousInput.remoteStream !== remoteStream || previousInput.enabled !== enabled) {
    setPreviousInput({ remoteStream, enabled });
    setDetectionState(INITIAL_STATE);
  }

  useEffect(() => {
    if (!enabled || !remoteStream) {
      if (detectorRef.current) {
        detectorRef.current.stop();
        detectorRef.current = null;
      }
      return;
    }

    const detector = new VoiceDeepfakeDetector((newState) => {
      setDetectionState(newState);
    });

    detector.start(remoteStream);
    detectorRef.current = detector;

    return () => {
      detector.stop();
      detectorRef.current = null;
    };
  }, [remoteStream, enabled]);

  return detectionState;
}
