/** Acoustic evidence strength, never a probability of spoofing. */
export function aggregateAcousticEvidence(indicators: number[]): number {
  const sorted = indicators.filter(Number.isFinite).map(value => Math.max(0, Math.min(100, value))).sort((a, b) => b - a);
  const first = sorted[0] ?? 0;
  const second = sorted[1] ?? 0;
  // One feature is ambiguous; retain its evidence without calling it conclusive.
  return Math.round(second >= 35 ? Math.min(98, first * 0.7 + second * 0.3) : Math.min(48, first * 0.55));
}

export function estimatePitchPeriod(samples: Uint8Array, sampleRate: number): number {
  let best = 0.8;
  let period = 0;
  const maxLag = Math.min(Math.floor(sampleRate / 60), samples.length - 128);
  for (let lag = Math.floor(sampleRate / 400); lag <= maxLag; lag += 2) {
    const length = Math.min(512, samples.length - lag);
    let cross = 0, left = 0, right = 0;
    for (let i = 0; i < length; i++) {
      const a = samples[i] - 128, b = samples[i + lag] - 128;
      cross += a * b; left += a * a; right += b * b;
    }
    const correlation = left > 0 && right > 0 ? cross / Math.sqrt(left * right) : 0;
    if (correlation > best) { best = correlation; period = lag; }
  }
  return period;
}
