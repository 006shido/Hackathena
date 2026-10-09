import assert from 'node:assert/strict';
import { aggregateAcousticEvidence, estimatePitchPeriod } from '../../client/src/detection/acousticScore.ts';

assert.equal(aggregateAcousticEvidence([0, 0, 0]), 0);
assert.equal(aggregateAcousticEvidence([50, 0, 0]), 28);
assert.equal(aggregateAcousticEvidence([90, 0]), 48);
assert.equal(aggregateAcousticEvidence([90, 60]), 81);
assert.equal(aggregateAcousticEvidence([NaN, Infinity, 0]), 0);
// Regression: low male pitch previously accessed beyond the 512-sample buffer.
for (const rate of [44100, 48000]) {
  const samples = Uint8Array.from({ length: 1024 }, (_, i) => Math.round(128 + 60 * Math.sin(2 * Math.PI * 80 * i / rate)));
  const period = estimatePitchPeriod(samples, rate);
  assert.ok(Math.abs(rate / period - 80) < 1, `80 Hz at ${rate}: ${period}`);
}
assert.equal(estimatePitchPeriod(new Uint8Array(1024).fill(128), 48000), 0);
assert.equal(estimatePitchPeriod(new Uint8Array(64).fill(128), 48000), 0);
console.log('Acoustic score and bounded pitch regressions passed.');
