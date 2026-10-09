import assert from 'node:assert/strict';

const drawn: string[] = [];
let cameraStops = 0, derivedStops = 0, bitmapCloses = 0, sessionCounter = 0;
let delayPlay = false;
const playResolvers: Array<() => void> = [];
const track = { getSettings: () => ({ width: 640, height: 480 }), stop: () => cameraStops++ };
const context = { drawImage: (image: any) => { if (image.id) drawn.push(image.id); } };
(globalThis as any).localStorage = { getItem: () => 'test-token' };
(globalThis as any).window = { setTimeout, clearTimeout };
(globalThis as any).MediaStream = class {
  constructor(private tracks: any[]) {}
  getVideoTracks() { return this.tracks; }
};
(globalThis as any).document = { createElement: (tag: string) => tag === 'video' ? {
  play: async () => { if (delayPlay) await new Promise<void>(resolve => playResolvers.push(resolve)); }, pause: () => {}, srcObject: null,
} : {
  width: 0, height: 0, getContext: () => context,
  toBlob: (callback: any) => callback(new Blob(['frame'])),
  captureStream: () => ({ getTracks: () => [{ stop: () => derivedStops++ }], getVideoTracks: () => [{}] }),
} };
(globalThis as any).createImageBitmap = async (blob: Blob) => ({ id: await blob.text(), close: () => bitmapCloses++ });
const deleted: string[] = [];
(globalThis as any).fetch = async (path: string, options: any) => {
  if (options.method === 'DELETE') { deleted.push(path); return new Response('{}'); }
  if (path.endsWith('/sessions')) return new Response(JSON.stringify({ session_id: String(++sessionCounter).padStart(32, '0') }));
  // Deliberately deliver a stale response despite cancellation, exercising the
  // generation check rather than assuming the browser always aborts promptly.
  await new Promise(resolve => setTimeout(resolve, 75));
  return new Response(path.split('/')[5], { headers: { 'x-face-detected': 'true' } });
};

const { NeuralVideoPreview } = await import('../../client/src/attack/neuralVideoPreview.ts');
const preview = new NeuralVideoPreview();
try {
  await preview.start(track as any, new Blob(['source-a']));
  const oldId = String(1).padStart(32, '0');
  await preview.setSource(new Blob(['source-b']));
  await new Promise(resolve => setTimeout(resolve, 240));
  assert(!drawn.includes(oldId), 'Old identity was drawn after source change.');
  assert(drawn.includes(String(2).padStart(32, '0')), 'Current source frames were never drawn.');
  assert(bitmapCloses > 0, 'Decoded bitmaps were not released.');
  preview.stop();
  const afterStop = drawn.length;
  await new Promise(resolve => setTimeout(resolve, 150));
  assert.equal(drawn.length, afterStop, 'A late frame was drawn after stopping.');
  assert.equal(cameraStops, 0, 'Stopping preview stopped the original camera.');
  assert.equal(derivedStops, 1);
  assert(deleted.length >= 2, 'Source-change and stop sessions were not released.');
  delayPlay = true;
  const overlapping = new NeuralVideoPreview();
  const first = overlapping.start(track as any, new Blob(['old-start'])).then(
    value => ({ value, error: null }), error => ({ value: null, error }));
  const second = overlapping.start(track as any, new Blob(['new-start']));
  assert.equal(playResolvers.length, 2);
  playResolvers[0]();
  assert((await first).error instanceof Error, 'Superseded startup was allowed to finish.');
  playResolvers[1]();
  await second;
  assert(overlapping.ready, 'Cancelled older startup stopped the newer preview.');
  overlapping.stop();delayPlay = false;
  assert.equal(cameraStops, 0);
  assert.equal(derivedStops, 2, 'Overlapping starts created an extra derived stream.');
  console.log('PASS: stale identity rejection, bitmap cleanup, stopped-frame rejection and camera preservation.');
} finally { preview.stop(); }
