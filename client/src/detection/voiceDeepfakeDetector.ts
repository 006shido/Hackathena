import { DetectionStatus,VoiceAcousticMetrics,VoiceDetectionState } from '../types/detection';

interface FrameAnomalyScores {
  carrierScore: number;
  resonanceScore: number;
  combScore: number;
  cutoffScore: number;
  jitterScore: number;
  anomalies: string[];
}

export class VoiceDeepfakeDetector {
  private audioCtx: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private sourceNode: MediaStreamAudioSourceNode | null = null;
  private isRunning = false;
  private animFrameId: number | null = null;
  private smoothedScore = 0;

  // Adaptive ambient noise floor tracking (rejects fan hum, AC, and room noise)
  private ambientNoiseFloor = 0.015;

  // Multi-frame pitch tracking for micro-jitter
  private pitchPeriodHistory: number[] = [];

  // Multi-frame rolling buffer (10 frames ~300ms) for high-accuracy temporal stability
  private frameHistory: FrameAnomalyScores[] = [];

  // Speech Activity Hold & UI Throttling
  private lastSpeechTimestamp = 0;
  private isVoiceActiveHeld = false;
  private lastUiUpdateTime = 0;
  private cachedAnomalies: string[] = [];

  private onStateUpdate?: (state: VoiceDetectionState) => void;

  constructor(onUpdate?: (state: VoiceDetectionState) => void) {
    this.onStateUpdate = onUpdate;
  }

  public start(remoteStream: MediaStream) {
    this.stop();

    const audioTracks = remoteStream.getAudioTracks();
    if (audioTracks.length === 0) {
      console.warn('[VoiceDetector] No audio tracks found in stream.');
      return;
    }

    try {
      const AudioContextClass =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      this.audioCtx = new AudioContextClass();

      if (this.audioCtx.state === 'suspended') {
        this.audioCtx.resume().catch(() => {
          // Will resume automatically upon user gesture
        });
      }

      this.sourceNode = this.audioCtx.createMediaStreamSource(remoteStream);
      this.analyser = this.audioCtx.createAnalyser();
      this.analyser.fftSize = 1024;
      this.analyser.smoothingTimeConstant = 0.82;

      // Connect source to analyser ONLY - DO NOT connect to destination
      this.sourceNode.connect(this.analyser);

      this.isRunning = true;
      this.runAnalysisLoop();
    } catch (err) {
      console.error('[VoiceDetector] Failed to initialize AudioContext:', err);
    }
  }

  public stop() {
    this.isRunning = false;
    if (this.animFrameId !== null) {
      cancelAnimationFrame(this.animFrameId);
      this.animFrameId = null;
    }
    if (this.sourceNode) {
      try {
        this.sourceNode.disconnect();
      } catch {
        // ignore
      }
      this.sourceNode = null;
    }
    if (this.audioCtx && this.audioCtx.state !== 'closed') {
      try {
        this.audioCtx.close();
      } catch {
        // ignore
      }
      this.audioCtx = null;
    }
    this.analyser = null;
    this.smoothedScore = 0;
    this.pitchPeriodHistory = [];
    this.frameHistory = [];
    this.lastSpeechTimestamp = 0;
    this.isVoiceActiveHeld = false;
    this.cachedAnomalies = [];
    this.ambientNoiseFloor = 0.015;
  }

  private runAnalysisLoop = () => {
    if (!this.isRunning || !this.analyser || !this.audioCtx) return;

    const now = Date.now();
    const bufferLength = this.analyser.frequencyBinCount; // 512 bins
    const freqData = new Uint8Array(bufferLength);
    const timeData = new Uint8Array(bufferLength);

    this.analyser.getByteFrequencyData(freqData);
    this.analyser.getByteTimeDomainData(timeData);

    const sampleRate = this.audioCtx.sampleRate || 48000;
    const binWidth = sampleRate / (bufferLength * 2); // ~46.875 Hz per bin

    // -------------------------------------------------------------
    // FEATURE 1: RMS Energy & Adaptive Noise Floor (Fan/Hiss Rejection)
    // -------------------------------------------------------------
    let sumSquares = 0;
    for (let i = 0; i < timeData.length; i++) {
      const val = (timeData[i] - 128) / 128;
      sumSquares += val * val;
    }
    const rms = Math.sqrt(sumSquares / timeData.length);

    // Adapt noise floor slowly to constant ambient room/fan sound
    if (rms < this.ambientNoiseFloor) {
      this.ambientNoiseFloor = this.ambientNoiseFloor * 0.9 + rms * 0.1;
    } else if (rms < 0.035) {
      this.ambientNoiseFloor = this.ambientNoiseFloor * 0.996 + rms * 0.004;
    }

    // -------------------------------------------------------------
    // FEATURE 2: Spectral Flatness (Wiener Entropy in speech band)
    // Fan/white noise has high flatness (>0.30); human vowels have low flatness (<0.15)
    // -------------------------------------------------------------
    const speechBins = Math.min(bufferLength, 128); // 0 to ~6 kHz
    let sumLog = 0;
    let sumLinear = 0;
    for (let i = 1; i < speechBins; i++) {
      const mag = Math.max(freqData[i] / 255, 0.0001);
      sumLog += Math.log(mag);
      sumLinear += mag;
    }
    const geometricMean = Math.exp(sumLog / (speechBins - 1));
    const arithmeticMean = sumLinear / (speechBins - 1);
    const spectralFlatness = arithmeticMean > 0 ? Math.min(1, geometricMean / arithmeticMean) : 0;

    // Distinguish real speech from fan/ambient noise:
    // Speech requires energy above noise floor, absolute minimum 0.024, AND not flat pink noise
    const isAboveNoise = rms > Math.max(0.024, this.ambientNoiseFloor * 2.2);
    const isNonFan = spectralFlatness < 0.30 || rms > 0.06;
    const instantSpeech = isAboveNoise && isNonFan;

    // VAD Hold filter (500ms hangover to bridge syllables)
    if (instantSpeech) {
      this.lastSpeechTimestamp = now;
      this.isVoiceActiveHeld = true;
    } else if (now - this.lastSpeechTimestamp > 500) {
      this.isVoiceActiveHeld = false;
    }

    const hasAudio = this.isVoiceActiveHeld;

    // -------------------------------------------------------------
    // FEATURE 3: Spectral Moments (Centroid & Rolloff)
    // -------------------------------------------------------------
    let weightedSum = 0;
    let totalMagnitude = 0;
    for (let i = 0; i < bufferLength; i++) {
      const mag = freqData[i];
      weightedSum += i * binWidth * mag;
      totalMagnitude += mag;
    }
    const spectralCentroid = totalMagnitude > 0 ? Math.round(weightedSum / totalMagnitude) : 0;

    let cumulativeMag = 0;
    const thresholdMag = totalMagnitude * 0.85;
    let rolloffHz = 0;
    for (let i = 0; i < bufferLength; i++) {
      cumulativeMag += freqData[i];
      if (cumulativeMag >= thresholdMag) {
        rolloffHz = Math.round(i * binWidth);
        break;
      }
    }

    // -------------------------------------------------------------
    // FEATURE 4: Sub-Harmonic Carrier Oscillation Detection
    // 50Hz/60Hz mains hum is low volume (<40) and steady.
    // Ring modulator carrier is loud (>80) and active during speech.
    // -------------------------------------------------------------
    const carrierBin1 = freqData[1] || 0; // ~47 Hz
    const carrierBin2 = freqData[2] || 0; // ~94 Hz
    const neighborAverage =
      ((freqData[4] || 0) + (freqData[5] || 0) + (freqData[6] || 0) + (freqData[7] || 0)) / 4;

    const maxCarrierBin = Math.max(carrierBin1, carrierBin2);
    const carrierSpikeRatio = neighborAverage > 0 ? maxCarrierBin / neighborAverage : 1;

    // Must be speech-active, loud carrier (>75), and sharp spike over neighbors
    const carrierHarmonicDetected =
      instantSpeech &&
      maxCarrierBin > 75 &&
      carrierSpikeRatio > 2.2 &&
      (maxCarrierBin - neighborAverage) > 35;

    let carrierScore = 0;
    if (carrierHarmonicDetected) {
      carrierScore = Math.min(100, Math.round(45 + (carrierSpikeRatio - 2.2) * 30));
    }

    // -------------------------------------------------------------
    // FEATURE 5: Resonant Formant Peaking / High-Q Bandpass Dome
    // Checks for unnatural Q=3 to Q=5 filter domes at 1200Hz or 2400Hz
    // -------------------------------------------------------------
    // 1200Hz bandpass peak (~bin 26)
    const bin1200 = Math.round(1200 / binWidth);
    let peak1200 = 0;
    for (let i = bin1200 - 3; i <= bin1200 + 3; i++) {
      if (freqData[i] > peak1200) peak1200 = freqData[i];
    }
    let surround1200 = 0;
    for (let i = bin1200 - 12; i <= bin1200 + 12; i++) {
      surround1200 += freqData[i] || 0;
    }
    const avgSurround1200 = surround1200 / 25;
    const resonance1200Ratio = avgSurround1200 > 0 ? peak1200 / avgSurround1200 : 1;

    // 2400Hz peaking peak (~bin 51)
    const bin2400 = Math.round(2400 / binWidth);
    let peak2400 = 0;
    for (let i = bin2400 - 3; i <= bin2400 + 3; i++) {
      if (freqData[i] > peak2400) peak2400 = freqData[i];
    }
    let surround2400 = 0;
    for (let i = bin2400 - 12; i <= bin2400 + 12; i++) {
      surround2400 += freqData[i] || 0;
    }
    const avgSurround2400 = surround2400 / 25;
    const resonance2400Ratio = avgSurround2400 > 0 ? peak2400 / avgSurround2400 : 1;

    const maxResonanceRatio = Math.max(resonance1200Ratio, resonance2400Ratio);
    const bandpassResonanceDetected =
      instantSpeech &&
      maxResonanceRatio > 2.4 &&
      Math.max(peak1200, peak2400) > 85;

    let resonanceScore = 0;
    if (bandpassResonanceDetected) {
      resonanceScore = Math.min(100, Math.round(40 + (maxResonanceRatio - 2.4) * 35));
    }

    // -------------------------------------------------------------
    // FEATURE 6: True Comb Filter Regularity (Sharp Alternating Peaks)
    // Fan noise is a smooth curve (0 peaks). Ring-mod vocoder has 6-15 sharp peaks!
    // -------------------------------------------------------------
    let sharpCombPeaks = 0;
    let totalPeakDepth = 0;

    if (instantSpeech) {
      // Inspect speech band from bin 4 to 55 (200Hz to 2.5kHz)
      for (let i = 4; i < 55; i++) {
        const cur = freqData[i] || 0;
        const left = freqData[i - 1] || 0;
        const right = freqData[i + 1] || 0;
        // Peak must rise sharply above both immediate neighbors and have solid volume
        if (cur > left + 14 && cur > right + 14 && cur > 60) {
          sharpCombPeaks++;
          totalPeakDepth += (cur - left) + (cur - right);
        }
      }
    }

    const avgPeakDepth = sharpCombPeaks > 0 ? totalPeakDepth / sharpCombPeaks : 0;
    const combDetected = instantSpeech && sharpCombPeaks >= 5 && avgPeakDepth > 32;

    let combScore = 0;
    if (combDetected) {
      combScore = Math.min(100, Math.round(45 + (sharpCombPeaks - 5) * 9));
    }

    // -------------------------------------------------------------
    // FEATURE 7: Brickwall Lowpass Cutoff (< 2.2 kHz)
    // Deep-pitch neural filters cut off steeply at 1800Hz with zero HF energy
    // -------------------------------------------------------------
    let lowBandEnergy = 0;
    for (let i = 2; i < Math.floor(2500 / binWidth); i++) {
      lowBandEnergy += freqData[i] || 0;
    }
    let highBandEnergy = 0;
    for (let i = Math.floor(3500 / binWidth); i < speechBins; i++) {
      highBandEnergy += freqData[i] || 0;
    }
    const hfToLfRatio = lowBandEnergy > 0 ? highBandEnergy / lowBandEnergy : 0.5;

    const cutoffDetected =
      instantSpeech &&
      lowBandEnergy > 800 &&
      hfToLfRatio < 0.025 &&
      rolloffHz < 2200;

    let cutoffScore = 0;
    if (cutoffDetected) {
      cutoffScore = Math.min(100, Math.round(50 + (0.025 - hfToLfRatio) * 1600));
    }

    // -------------------------------------------------------------
    // FEATURE 8: Time-Domain Pitch Period Autocorrelation & Jitter
    // -------------------------------------------------------------
    let bestCorrelation = 0;
    let currentPitchPeriod = 0;
    const minLag = Math.floor(sampleRate / 400); // 400 Hz
    const maxLag = Math.min(Math.floor(sampleRate / 60), timeData.length - 256 + 1);

    if (instantSpeech) {
      for (let lag = minLag; lag < maxLag; lag += 2) {
        let corr = 0;
        for (let i = 0; i < 256; i++) {
          corr += (timeData[i] - 128) * (timeData[i + lag] - 128);
        }
        if (corr > bestCorrelation) {
          bestCorrelation = corr;
          currentPitchPeriod = lag;
        }
      }
    }

    let jitterVariance = 0.015; // Natural human baseline (~1.5%)
    if (currentPitchPeriod > 0) {
      this.pitchPeriodHistory.push(currentPitchPeriod);
      if (this.pitchPeriodHistory.length > 10) {
        this.pitchPeriodHistory.shift();
      }

      if (this.pitchPeriodHistory.length >= 5) {
        let sumDelta = 0;
        for (let p = 1; p < this.pitchPeriodHistory.length; p++) {
          sumDelta += Math.abs(this.pitchPeriodHistory[p] - this.pitchPeriodHistory[p - 1]);
        }
        jitterVariance = sumDelta / (this.pitchPeriodHistory.length * currentPitchPeriod);
      }
    }

    const zeroJitterDetected =
      instantSpeech &&
      jitterVariance < 0.0025 &&
      bestCorrelation > 150000;

    let jitterScore = 0;
    if (zeroJitterDetected) {
      jitterScore = Math.min(100, Math.round(45 + (0.0025 - jitterVariance) * 15000));
    }

    // -------------------------------------------------------------
    // CURRENT FRAME ANOMALIES COMPILATION
    // -------------------------------------------------------------
    const frameAnomalies: string[] = [];
    if (carrierHarmonicDetected) frameAnomalies.push('Carrier Ring-Modulation (65Hz / Sub)');
    if (bandpassResonanceDetected) frameAnomalies.push(`Resonant Formant Peak (${resonance1200Ratio > resonance2400Ratio ? '1.2kHz' : '2.4kHz'})`);
    if (combDetected) frameAnomalies.push('Harmonic Comb Filter Regularity');
    if (cutoffDetected) frameAnomalies.push('Lowpass Bandwidth Cliff (< 2.2kHz)');
    if (zeroJitterDetected) frameAnomalies.push('Mechanical Pitch Lock (Zero Jitter)');

    // Push to 10-frame rolling window for temporal stability
    if (instantSpeech) {
      this.frameHistory.push({
        carrierScore,
        resonanceScore,
        combScore,
        cutoffScore,
        jitterScore,
        anomalies: frameAnomalies,
      });
      if (this.frameHistory.length > 10) {
        this.frameHistory.shift();
      }
    } else if (!hasAudio) {
      this.frameHistory = [];
    }

    // -------------------------------------------------------------
    // MULTI-FRAME TEMPORAL ENSEMBLE CLASSIFIER
    // Requires CO-OCCURRENCE of at least 2 distinct acoustic anomalies
    // or 1 extremely strong sustained anomaly (>75) during speech.
    // Fan/ambient noise produces 0.
    // -------------------------------------------------------------
    let aggregatedAnomalyScore = 0;

    if (hasAudio && this.frameHistory.length > 0) {
      let avgCarrier = 0;
      let avgResonance = 0;
      let avgComb = 0;
      let avgCutoff = 0;
      let avgJitter = 0;

      for (const f of this.frameHistory) {
        avgCarrier += f.carrierScore;
        avgResonance += f.resonanceScore;
        avgComb += f.combScore;
        avgCutoff += f.cutoffScore;
        avgJitter += f.jitterScore;
      }
      const count = this.frameHistory.length;
      avgCarrier /= count;
      avgResonance /= count;
      avgComb /= count;
      avgCutoff /= count;
      avgJitter /= count;

      const primaryIndicators = [avgCarrier, avgResonance, avgComb, avgCutoff, avgJitter];
      // Only indicators >= 35 count as substantial
      const strongIndicators = primaryIndicators.filter((s) => s >= 35);

      if (strongIndicators.length >= 2) {
        // Confirmed deepfake: multiple distinct acoustic physical anomalies confirmed co-occurring!
        const sorted = [...strongIndicators].sort((a, b) => b - a);
        aggregatedAnomalyScore = Math.min(98, Math.round(sorted[0] * 0.7 + sorted[1] * 0.3));
      } else if (strongIndicators.length === 1 && strongIndicators[0] > 70) {
        // Single isolated anomaly: cautious score only, does not trigger critical threat
        aggregatedAnomalyScore = Math.min(48, Math.round(strongIndicators[0] * 0.55));
      } else {
        // Genuine human speech: natural vocal timbre, normal baseline
        aggregatedAnomalyScore = Math.min(15, Math.round(rms * 25));
      }

      // Collect all confirmed anomalies
      const anomalySet = new Set<string>();
      for (const f of this.frameHistory) {
        for (const a of f.anomalies) anomalySet.add(a);
      }
      if (anomalySet.size > 0 && strongIndicators.length >= 1) {
        this.cachedAnomalies = Array.from(anomalySet);
      } else {
        this.cachedAnomalies = [];
      }
    } else {
      aggregatedAnomalyScore = 0;
      this.cachedAnomalies = [];
    }

    // Smooth the display score with asymmetric alpha (fast attack on confirmed deepfake, smooth decay)
    const targetScore = aggregatedAnomalyScore;
    const isIncreasing = targetScore > this.smoothedScore;
    const smoothAlpha = isIncreasing ? 0.28 : 0.08;
    this.smoothedScore = this.smoothedScore * (1 - smoothAlpha) + targetScore * smoothAlpha;
    const finalScore = Math.round(this.smoothedScore);

    // Classification Status:
    // Requires confirmed speech (hasAudio) and high thresholds to eliminate false positives
    let status: DetectionStatus = 'idle';
    if (!hasAudio) {
      status = 'listening';
    } else if (finalScore >= 60) {
      status = 'deepfake';
    } else if (finalScore >= 30) {
      status = 'suspicious';
    } else {
      status = 'human';
    }

    const confidence = hasAudio
      ? Math.min(99, Math.max(78, 70 + Math.round(rms * 80) + (this.cachedAnomalies.length > 0 ? 12 : 0)))
      : 0;

    let maxPeak = 0;
    for (let i = 0; i < speechBins; i++) {
      if (freqData[i] > maxPeak) maxPeak = freqData[i];
    }
    const avgMagnitude = totalMagnitude / (bufferLength || 1);
    const harmonicPeakRatio = avgMagnitude > 0 ? maxPeak / avgMagnitude : 0;

    const metrics: VoiceAcousticMetrics = {
      rmsEnergy: Math.round(rms * 1000) / 1000,
      spectralCentroidHz: spectralCentroid,
      spectralRolloffHz: rolloffHz,
      spectralFlatness: Math.round(spectralFlatness * 100) / 100,
      harmonicPeakRatio: Math.round(harmonicPeakRatio * 10) / 10,
      carrierHarmonicDetected,
      bandpassResonanceDetected,
      jitterVariance: Math.round(jitterVariance * 10000) / 10000,
    };

    const state: VoiceDetectionState = {
      isAnalyzing: true,
      hasAudio,
      isVoiceActive: hasAudio,
      anomalyScore: finalScore,
      confidence,
      status,
      metrics,
      detectedAnomalies: this.cachedAnomalies,
      frequencyData: freqData,
      timeDomainData: timeData,
      timestamp: now,
    };

    // Throttle React state dispatch to 15 FPS (every 66ms)
    if (now - this.lastUiUpdateTime >= 66) {
      this.lastUiUpdateTime = now;
      if (this.onStateUpdate) {
        this.onStateUpdate(state);
      }
    }

    this.animFrameId = requestAnimationFrame(this.runAnalysisLoop);
  };
}
