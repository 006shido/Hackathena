export type VoicePreset = 'robotic-vocoder' | 'deep-pitch-neural' | 'synthetic-clone';

export class VoiceTransformationPipeline {
  private audioCtx: AudioContext | null = null;
  private sourceNode: MediaStreamAudioSourceNode | null = null;
  private destinationNode: MediaStreamAudioDestinationNode | null = null;
  private carrierOsc: OscillatorNode | null = null;
  private ringModGain: GainNode | null = null;
  private filterNode: BiquadFilterNode | null = null;
  private waveShaper: WaveShaperNode | null = null;
  private outputGain: GainNode | null = null;

  private active = false;
  private originalTrack: MediaStreamTrack | null = null;
  private outputStream: MediaStream | null = null;
  private preset: VoicePreset = 'robotic-vocoder';

  public start(originalTrack: MediaStreamTrack, preset: VoicePreset = 'robotic-vocoder'): MediaStreamTrack {
    this.originalTrack = originalTrack;
    this.preset = preset;
    this.active = true;

    // Create or resume AudioContext
    const AudioContextClass = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    this.audioCtx = new AudioContextClass();

    if (this.audioCtx.state === 'suspended') {
      this.audioCtx.resume();
    }

    const inputStream = new MediaStream([originalTrack]);
    this.sourceNode = this.audioCtx.createMediaStreamSource(inputStream);
    this.destinationNode = this.audioCtx.createMediaStreamDestination();

    this.buildAudioGraph();

    this.outputStream = this.destinationNode.stream;
    return this.outputStream.getAudioTracks()[0];
  }

  private buildAudioGraph() {
    if (!this.audioCtx || !this.sourceNode || !this.destinationNode) return;

    // Clean up any existing nodes
    this.cleanupNodes();

    const ctx = this.audioCtx;

    // Master output gain
    this.outputGain = ctx.createGain();
    this.outputGain.gain.value = 1.0;
    this.outputGain.connect(this.destinationNode);

    if (this.preset === 'robotic-vocoder') {
      // Ring modulation + bandpass for synthetic robotic voice
      this.filterNode = ctx.createBiquadFilter();
      this.filterNode.type = 'bandpass';
      this.filterNode.frequency.value = 1200;
      this.filterNode.Q.value = 3.0;

      // Ring modulator carrier oscillator (65Hz drone)
      this.carrierOsc = ctx.createOscillator();
      this.carrierOsc.type = 'sawtooth';
      this.carrierOsc.frequency.value = 65;

      this.ringModGain = ctx.createGain();
      this.ringModGain.gain.value = 0.0; // modulated by input

      // Input modulates carrier gain
      this.sourceNode.connect(this.filterNode);
      this.filterNode.connect(this.ringModGain.gain);
      this.carrierOsc.connect(this.ringModGain);

      // Mix wet + subtle dry
      const dryGain = ctx.createGain();
      dryGain.gain.value = 0.3;
      this.sourceNode.connect(dryGain);
      dryGain.connect(this.outputGain);

      const wetGain = ctx.createGain();
      wetGain.gain.value = 1.2;
      this.ringModGain.connect(wetGain);
      wetGain.connect(this.outputGain);

      this.carrierOsc.start();

    } else if (this.preset === 'deep-pitch-neural') {
      // Deep pitch formant shifting with low-pass & saturation
      this.filterNode = ctx.createBiquadFilter();
      this.filterNode.type = 'lowshelf';
      this.filterNode.frequency.value = 400;
      this.filterNode.gain.value = 14;

      const highCut = ctx.createBiquadFilter();
      highCut.type = 'lowpass';
      highCut.frequency.value = 1800;

      // Soft distortion curve
      this.waveShaper = ctx.createWaveShaper();
      this.waveShaper.curve = this.makeDistortionCurve(20) as any;
      this.waveShaper.oversample = '4x';

      this.sourceNode
        .connect(this.filterNode)
        .connect(highCut)
        .connect(this.waveShaper)
        .connect(this.outputGain);

    } else {
      // Synthetic AI clone preset: formant resonances + subtle jitter
      this.filterNode = ctx.createBiquadFilter();
      this.filterNode.type = 'peaking';
      this.filterNode.frequency.value = 2400;
      this.filterNode.Q.value = 5.0;
      this.filterNode.gain.value = 8;

      const subOsc = ctx.createOscillator();
      subOsc.type = 'sine';
      subOsc.frequency.value = 45;

      const amGain = ctx.createGain();
      amGain.gain.value = 0.4;
      subOsc.connect(amGain.gain);

      this.sourceNode.connect(this.filterNode);
      this.filterNode.connect(amGain);
      amGain.connect(this.outputGain);

      subOsc.start();
      this.carrierOsc = subOsc;
    }
  }

  private makeDistortionCurve(amount = 20): Float32Array {
    const k = typeof amount === 'number' ? amount : 20;
    const nSamples = 44100;
    const curve = new Float32Array(nSamples);
    const deg = Math.PI / 180;
    for (let i = 0; i < nSamples; ++i) {
      const x = (i * 2) / nSamples - 1;
      curve[i] = ((3 + k) * x * 20 * deg) / (Math.PI + k * Math.abs(x));
    }
    return curve;
  }

  public setPreset(preset: VoicePreset) {
    this.preset = preset;
    if (this.active) {
      this.buildAudioGraph();
    }
  }

  public isActive(): boolean {
    return this.active;
  }

  private cleanupNodes() {
    if (this.carrierOsc) {
      try {
        this.carrierOsc.stop();
        this.carrierOsc.disconnect();
      } catch {
        // Ignored
      }
      this.carrierOsc = null;
    }
    if (this.ringModGain) {
      this.ringModGain.disconnect();
      this.ringModGain = null;
    }
    if (this.filterNode) {
      this.filterNode.disconnect();
      this.filterNode = null;
    }
    if (this.waveShaper) {
      this.waveShaper.disconnect();
      this.waveShaper = null;
    }
    if (this.outputGain) {
      this.outputGain.disconnect();
      this.outputGain = null;
    }
  }

  public stop(): void {
    this.active = false;
    this.cleanupNodes();

    if (this.sourceNode) {
      this.sourceNode.disconnect();
      this.sourceNode = null;
    }

    if (this.outputStream) {
      this.outputStream.getTracks().forEach((track) => track.stop());
      this.outputStream = null;
    }

    if (this.audioCtx && this.audioCtx.state !== 'closed') {
      try {
        this.audioCtx.close();
      } catch {
        // Ignored
      }
      this.audioCtx = null;
    }

    this.originalTrack = null;
  }
}
