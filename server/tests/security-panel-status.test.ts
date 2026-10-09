import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { SecurityPanel } from '../../client/src/components/SecurityPanel';

const requireClient = createRequire(new URL('../../client/package.json', import.meta.url));
const React = requireClient('react');
const { renderToStaticMarkup } = requireClient('react-dom/server');
const render = (active: boolean, faceSwap: boolean, voiceTransform: boolean) =>
  renderToStaticMarkup(React.createElement(SecurityPanel, {
    peerAttackState: { active, faceSwap, voiceTransform, mode: 'combined' },
  }));

const both = render(true, true, true);
assert.match(both, /Video: Face change active/);
assert.match(both, /Audio: Voice transform active/);
assert.match(both, /Tester-reported changes/);
assert.doesNotMatch(both, /Face Landmark Analysis/);
const stopped = render(false, true, true);
assert.match(stopped, /Video: No face change reported/);
assert.match(stopped, /Audio: No voice transform reported/);
const voiceOnly = render(true, false, true);
assert.match(voiceOnly, /Video: No face change reported/);
assert.match(voiceOnly, /Audio: Voice transform active/);
console.log('PASS: receiver distinguishes reported video/audio changes from audio analysis.');

const waiting = renderToStaticMarkup(React.createElement(SecurityPanel, {
  hasIncomingPeer: false,
  voiceDetection: { hasAudio: false, isVoiceActive: false, confidence: 0, anomalyScore: 0,
    metrics: { carrierHarmonicDetected: false, bandpassResonanceDetected: false, harmonicPeakRatio: 0 } },
}));
assert.match(waiting, /Waiting for a peer to join/);
assert.match(waiting, /Independent face anomaly score unavailable/);
assert.match(waiting, /No speech measurement/);
assert.doesNotMatch(waiting, />0%</);
assert.doesNotMatch(waiting, /Clean|Natural|Nominal|Baseline/);
console.log('PASS: no-peer state does not fabricate measured audio or face scores.');

const measured = renderToStaticMarkup(React.createElement(SecurityPanel, {
  peerAttackState: { active: false, faceSwap: false, voiceTransform: false, mode: 'combined' },
  mediaDetection: { video: { status: 'measured', score: .73 }, audio: { status: 'measured', score: .21 } },
}));
assert.match(measured, /Model output: 73.0 \/ 100/);
assert.match(measured, /Model output: 21.0 \/ 100/);
assert.match(measured, /does not certify real or fake/);
assert.match(measured, /Video: No face change reported/);
console.log('PASS: model readings are independent of sender controls and explicitly experimental.');
