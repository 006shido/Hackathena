/** Per-call acoustic reference. Detects spectral changes, not speaker authenticity. */
export class VoiceReference {
  private capture: number[][] | null = null;
  private reference: { mean: number[]; deviation: number[] } | null = null;
  private recent: number[][] = [];
  get state(): 'none' | 'capturing' | 'ready' { return this.capture ? 'capturing' : this.reference ? 'ready' : 'none'; }
  get progress(): number { return this.capture ? Math.round(100 * this.capture.length / 40) : this.reference ? 100 : 0; }
  start() { this.capture = []; this.reference = null; this.recent = []; }
  clearRecent() { this.recent = []; }
  observe(features: number[]): number | null {
    if (!features.length || !features.every(Number.isFinite)) return null;
    if (this.capture) {
      this.capture.push(features);
      if (this.capture.length >= 40) {
        const samples = this.capture;
        const mean = features.map((_, i) => samples.reduce((sum, row) => sum + row[i], 0) / samples.length);
        const deviation = mean.map((value, i) => Math.sqrt(samples.reduce((sum, row) => sum + (row[i] - value) ** 2, 0) / samples.length));
        this.reference = { mean, deviation }; this.capture = null;
      }
      return null;
    }
    if (!this.reference) return null;
    this.recent.push(features); if (this.recent.length > 20) this.recent.shift();
    if (this.recent.length < 15) return null;
    const mean = features.map((_, i) => this.recent.reduce((sum, row) => sum + row[i], 0) / this.recent.length);
    // Compare window means, not individual phonemes. Allow approximately two
    // standard errors plus a 3 dB minimum margin for normal variation.
    const distances = mean.map((value, i) => Math.max(0, Math.abs(value - this.reference!.mean[i]) - Math.max(3, this.reference!.deviation[i] * 2 * Math.sqrt(1 / this.recent.length + 1 / 40))));
    const distance = Math.sqrt(distances.reduce((sum, value) => sum + value * value, 0) / distances.length);
    return Math.round(Math.min(100, distance * 12));
  }
}

export function normalizedSpeechBands(db: Float32Array, sampleRate: number, fftSize: number): number[] {
  const edges = [150, 300, 600, 1000, 1600, 2500, 4000, 6000];
  const powers = edges.slice(0, -1).map((edge, index) => {
    let sum = 0;
    for (let bin = Math.ceil(edge * fftSize / sampleRate); bin < Math.min(db.length, Math.ceil(edges[index + 1] * fftSize / sampleRate)); bin++) {
      if (Number.isFinite(db[bin])) sum += 10 ** (db[bin] / 10);
    }
    return sum;
  });
  const total = powers.reduce((sum, value) => sum + value, 0);
  return powers.map(value => 10 * Math.log10(Math.max(1e-8, value / Math.max(total, 1e-12))));
}
