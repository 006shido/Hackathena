export type VoicePreset = 'robotic-vocoder' | 'deep-pitch-neural' | 'synthetic-clone';

export class VoiceTransformationPipeline {
  private audioCtx: AudioContext | null = null;
  private sourceNode: MediaStreamAudioSourceNode | null = null;
  private destinationNode: MediaStreamAudioDestinationNode | null = null;
  private carrierOsc: OscillatorNode | null = null;
  private ringModGain: GainNode | null = null;
  private filterNode: BiquadFilterNode | null = null;
  private secondaryFilter: BiquadFilterNode | null = null;
  private waveShaper: WaveShaperNode | null = null;
  private outputGain: GainNode | null = null;
  private monitorGain: GainNode | null = null;

  private active = false;
  private isMonitoring = false;
  private monitorVolume = 0.85;
  private originalTrack: MediaStreamTrack | null = null;
  private outputStream: MediaStream | null = null;
  private preset: VoicePreset = 'robotic-vocoder';

  public async start(originalTrack: MediaStreamTrack, preset: VoicePreset = 'robotic-vocoder'): Promise<MediaStreamTrack> {
    this.originalTrack = originalTrack;
    this.preset = preset;
    this.active = true;

    // Create or reuse AudioContext with vendor prefixes
    const AudioContextClass =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;

    if (!this.audioCtx || this.audioCtx.state === 'closed') {
      this.audioCtx = new AudioContextClass();
    }

    // Modern browsers require explicit resume within user gesture or after init
    if (this.audioCtx.state === 'suspended') {
      try {
        await this.audioCtx.resume();
        console.log('[VoicePipeline] AudioContext resumed successfully in state:', this.audioCtx.state);
      } catch (err) {
        console.warn('[VoicePipeline] AudioContext resume error:', err);
      }
    }

    // Auto-resume if browser suspends context (e.g., tab backgrounding or mobile policy)
    this.audioCtx.onstatechange = () => {
      if (this.audioCtx?.state === 'suspended' && this.active) {
        this.audioCtx.resume().catch(() => {});
      }
    };

    const inputStream = new MediaStream([originalTrack]);
    this.sourceNode = this.audioCtx.createMediaStreamSource(inputStream);
    this.destinationNode = this.audioCtx.createMediaStreamDestination();

    this.buildAudioGraph();

    this.outputStream = this.destinationNode.stream;
    const transformedTrack = this.outputStream.getAudioTracks()[0];
    transformedTrack.enabled = true;
    return transformedTrack;
  }

  public setMonitoring(enabled: boolean, volume = 0.85) {
    this.isMonitoring = enabled;
    this.monitorVolume = volume;
    if (this.monitorGain && this.audioCtx) {
      this.monitorGain.gain.setValueAtTime(
        enabled ? this.monitorVolume : 0,
        this.audioCtx.currentTime
      );
    }
  }

  public getMonitoring(): boolean {
    return this.isMonitoring;
  }

  private buildAudioGraph() {
    if (!this.audioCtx || !this.sourceNode || !this.destinationNode) return;

    this.cleanupNodes();
    const ctx = this.audioCtx;

    // Master output gain node connected to WebRTC destination
    this.outputGain = ctx.createGain();
    this.outputGain.gain.value = 1.0;
    this.outputGain.connect(this.destinationNode);

    // Dedicated local monitor gain node routed to tester's headphones
    this.monitorGain = ctx.createGain();
    this.monitorGain.gain.setValueAtTime(
      this.isMonitoring ? this.monitorVolume : 0,
      ctx.currentTime
    );
    this.outputGain.connect(this.monitorGain);
    this.monitorGain.connect(ctx.destination);

    // Preamp input booster to ensure robust signal
    const preamp = ctx.createGain();
    preamp.gain.value = 1.8;
    this.sourceNode.connect(preamp);

    if (this.preset === 'robotic-vocoder') {
      // 1. Robotic Vocoder Preset (Crisp metallic ring modulation)
      this.filterNode = ctx.createBiquadFilter();
      this.filterNode.type = 'bandpass';
      this.filterNode.frequency.value = 1350;
      this.filterNode.Q.value = 2.2;
      preamp.connect(this.filterNode);

      // Ring modulator carrier oscillator (160 Hz sawtooth drone)
      this.carrierOsc = ctx.createOscillator();
      this.carrierOsc.type = 'sawtooth';
      this.carrierOsc.frequency.value = 160;

      this.ringModGain = ctx.createGain();
      this.ringModGain.gain.value = 0.0;

      // Voice signal modulates carrier gain
      this.filterNode.connect(this.ringModGain.gain);
      this.carrierOsc.connect(this.ringModGain);

      // Mix wet (modulated robot voice) + subtle dry (clarity)
      const dryGain = ctx.createGain();
      dryGain.gain.value = 0.25;
      this.sourceNode.connect(dryGain);
      dryGain.connect(this.outputGain);

      const wetGain = ctx.createGain();
      wetGain.gain.value = 1.6;
      this.ringModGain.connect(wetGain);
      wetGain.connect(this.outputGain);

      this.carrierOsc.start();

    } else if (this.preset === 'deep-pitch-neural') {
      // 2. Deep Pitch Neural Preset (Heavy bass resonance + pitch shift effect)
      this.filterNode = ctx.createBiquadFilter();
      this.filterNode.type = 'lowshelf';
      this.filterNode.frequency.value = 220;
      this.filterNode.gain.value = 18;

      this.secondaryFilter = ctx.createBiquadFilter();
      this.secondaryFilter.type = 'peaking';
      this.secondaryFilter.frequency.value = 500;
      this.secondaryFilter.Q.value = 1.8;
      this.secondaryFilter.gain.value = 8;

      const highCut = ctx.createBiquadFilter();
      highCut.type = 'lowpass';
      highCut.frequency.value = 1700;
      highCut.Q.value = 1.5;

      // Saturation wave shaper
      this.waveShaper = ctx.createWaveShaper();
      this.waveShaper.curve = this.makeDistortionCurve(28) as any;
      this.waveShaper.oversample = '4x';

      const postGain = ctx.createGain();
      postGain.gain.value = 1.5;

      preamp
        .connect(this.filterNode)
        .connect(this.secondaryFilter)
        .connect(highCut)
        .connect(this.waveShaper)
        .connect(postGain)
        .connect(this.outputGain);

    } else {
      // 3. AI Synthetic Clone Preset (Synthetic resonance + subtle jitter)
      this.filterNode = ctx.createBiquadFilter();
      this.filterNode.type = 'peaking';
      this.filterNode.frequency.value = 1600;
      this.filterNode.Q.value = 4.0;
      this.filterNode.gain.value = 10;

      this.secondaryFilter = ctx.createBiquadFilter();
      this.secondaryFilter.type = 'peaking';
      this.secondaryFilter.frequency.value = 3200;
      this.secondaryFilter.Q.value = 3.5;
      this.secondaryFilter.gain.value = 8;

      const subOsc = ctx.createOscillator();
      subOsc.type = 'sine';
      subOsc.frequency.value = 38;

      const amGain = ctx.createGain();
      amGain.gain.value = 0.35;
      subOsc.connect(amGain.gain);

      preamp.connect(this.filterNode);
      this.filterNode.connect(this.secondaryFilter);
      this.secondaryFilter.connect(amGain);
      amGain.connect(this.outputGain);

      // Mix natural dry signal
      const dryGain = ctx.createGain();
      dryGain.gain.value = 0.35;
      this.sourceNode.connect(dryGain);
      dryGain.connect(this.outputGain);

      subOsc.start();
      this.carrierOsc = subOsc;
    }
  }

  private makeDistortionCurve(amount = 25): Float32Array {
    const k = typeof amount === 'number' ? amount : 25;
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
      try { this.ringModGain.disconnect(); } catch {}
      this.ringModGain = null;
    }
    if (this.filterNode) {
      try { this.filterNode.disconnect(); } catch {}
      this.filterNode = null;
    }
    if (this.secondaryFilter) {
      try { this.secondaryFilter.disconnect(); } catch {}
      this.secondaryFilter = null;
    }
    if (this.waveShaper) {
      try { this.waveShaper.disconnect(); } catch {}
      this.waveShaper = null;
    }
    if (this.outputGain) {
      try { this.outputGain.disconnect(); } catch {}
      this.outputGain = null;
    }
    if (this.monitorGain) {
      try { this.monitorGain.disconnect(); } catch {}
      this.monitorGain = null;
    }
  }

  public stop(): void {
    this.active = false;
    this.cleanupNodes();

    if (this.sourceNode) {
      try { this.sourceNode.disconnect(); } catch {}
      this.sourceNode = null;
    }

    if (this.outputStream) {
      this.outputStream.getTracks().forEach((track) => {
        try {
          track.enabled = false;
          track.stop();
        } catch (e) {}
      });
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
