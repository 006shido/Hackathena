/**
 * Automated Integration Test Suite for Express ML Face Swap Proxy (/api/ml/face-swap)
 *
 * Verifies:
 *   1. GET /api/ml/health -> 200 OK
 *   2. POST /api/ml/face-swap with 3 genuine CelebA cross-ID pairs
 *      - Pair 1: 004831.jpg -> 004865.jpg
 *      - Pair 2: 004931.jpg -> 004842.jpg
 *      - Pair 3: 004893.jpg -> 004925.jpg
 *   3. Error Handling:
 *      - Missing source
 *      - Missing target
 *      - Invalid Content-Type
 *      - Invalid file extension (.txt)
 *   4. Concurrency:
 *      - 429 Too Many Requests propagation on simultaneous GPU inference
 */

import fs from 'fs';
import path from 'path';
import assert from 'assert';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '../../');

const EXPRESS_BASE = process.env.TEST_APP_URL || 'http://127.0.0.1:5001';

const PAIRS = [
  {
    name: 'Pair 1 (004831 -> 004865)',
    source: path.join(REPO_ROOT, 'ml/data/celeba/img_align_celeba/004831.jpg'),
    target: path.join(REPO_ROOT, 'ml/data/celeba/img_align_celeba/004865.jpg'),
    expectedGain: 1.0636,
  },
  {
    name: 'Pair 2 (004931 -> 004842)',
    source: path.join(REPO_ROOT, 'ml/data/celeba/img_align_celeba/004931.jpg'),
    target: path.join(REPO_ROOT, 'ml/data/celeba/img_align_celeba/004842.jpg'),
    expectedGain: 0.7785,
  },
  {
    name: 'Pair 3 (004893 -> 004925)',
    source: path.join(REPO_ROOT, 'ml/data/celeba/img_align_celeba/004893.jpg'),
    target: path.join(REPO_ROOT, 'ml/data/celeba/img_align_celeba/004925.jpg'),
    expectedGain: 0.6639,
  },
];

interface InferenceResult {
  pair: string;
  roundtripMs: number;
  pipelineLatencyMs: number;
  networkLatencyMs: number;
  expressOverheadMs: number;
  landmarkErrorPx: number;
  identityGain: number;
  faceDetected: boolean;
}

async function testHealth() {
  console.log('\n--- Test 1: GET /api/ml/health ---');
  const res = await fetch(`${EXPRESS_BASE}/api/ml/health`);
  assert.strictEqual(res.status, 200, `Expected 200, got ${res.status}`);
  const data = await res.json();
  console.log('Health Response:', data);
  assert.strictEqual(data.status, 'ok');
  assert.strictEqual(data.model, 'fullres');
  assert.strictEqual(data.checkpoint_sha256, '156e922187b56c3d4fcddc8bec48bc5fd85813f78739a6801b09dcb4e3d5a4d9', 'Golden metrics require the current fullres checkpoint');
  assert.strictEqual(data.ready, true);
  console.log('GET /api/ml/health PASSED!');
}

async function testInferencePair(pair: typeof PAIRS[0], index: number): Promise<InferenceResult> {
  console.log(`\n--- Test 2.${index + 1}: POST /api/ml/face-swap: ${pair.name} ---`);
  assert(fs.existsSync(pair.source), `Source file not found: ${pair.source}`);
  assert(fs.existsSync(pair.target), `Target file not found: ${pair.target}`);

  const srcBuffer = fs.readFileSync(pair.source);
  const tgtBuffer = fs.readFileSync(pair.target);

  const form = new FormData();
  form.append('source', new Blob([srcBuffer], { type: 'image/jpeg' }), path.basename(pair.source));
  form.append('target', new Blob([tgtBuffer], { type: 'image/jpeg' }), path.basename(pair.target));

  const t0 = performance.now();
  const res = await fetch(`${EXPRESS_BASE}/api/ml/face-swap`, {
    method: 'POST',
    body: form,
  });
  const t1 = performance.now();
  const roundtripMs = Math.round(t1 - t0);

  console.log(`HTTP Status: ${res.status} (Roundtrip: ${roundtripMs} ms)`);
  assert.strictEqual(res.status, 200, `Inference request failed with HTTP ${res.status}`);

  const data = await res.json();
  assert.strictEqual(data.status, 'ok');
  assert(typeof data.swapped_image === 'string' && data.swapped_image.startsWith('data:image/png;base64,'));
  assert(typeof data.mask_image === 'string' && data.mask_image.startsWith('data:image/png;base64,'));
  assert(data.metadata, 'Metadata missing from response');

  const meta = data.metadata;
  const pipelineLatencyMs = meta.latency_ms;
  const networkLatencyMs = meta.network_latency_ms;
  const expressOverheadMs = Math.round(roundtripMs - pipelineLatencyMs);

  console.log(`  Model:              ${meta.model}`);
  console.log(`  Device:             ${meta.device} (${meta.gpu})`);
  console.log(`  Pipeline Latency:   ${pipelineLatencyMs} ms`);
  console.log(`  Net GPU Latency:    ${networkLatencyMs} ms`);
  console.log(`  Express Overhead:   ${expressOverheadMs} ms`);
  console.log(`  Landmark Error:     ${meta.landmark_error} px`);
  console.log(`  Face Redetected:    ${meta.face_detected}`);
  console.log(`  Identity Gain (A-C):${meta.identity_gain > 0 ? '+' : ''}${meta.identity_gain}`);
  console.log(`  Expected Gain:      +${pair.expectedGain}`);

  assert.strictEqual(meta.face_detected, true, 'Face was not redetected on composite output');
  assert(Math.abs(meta.identity_gain - pair.expectedGain) < 0.0001, `Identity gain diverged significantly: got ${meta.identity_gain}, expected ${pair.expectedGain}`);

  console.log(`${pair.name} PASSED!`);

  return {
    pair: pair.name,
    roundtripMs,
    pipelineLatencyMs,
    networkLatencyMs,
    expressOverheadMs,
    landmarkErrorPx: meta.landmark_error,
    identityGain: meta.identity_gain,
    faceDetected: meta.face_detected,
  };
}

async function testErrorHandling() {
  console.log('\n--- Test 3: Error Handling & Validation ---');

  // 3.1 Missing target
  console.log('Testing missing target file...');
  const formMissingTarget = new FormData();
  formMissingTarget.append('source', new Blob([Buffer.from('fake-data')], { type: 'image/jpeg' }), 'src.jpg');
  const resMissingTarget = await fetch(`${EXPRESS_BASE}/api/ml/face-swap`, {
    method: 'POST',
    body: formMissingTarget,
  });
  console.log(`Missing target status: ${resMissingTarget.status}`);
  assert.strictEqual(resMissingTarget.status, 400);
  const jsonMissingTarget = await resMissingTarget.json();
  console.log('Missing target error:', jsonMissingTarget);
  assert(jsonMissingTarget.error, 'Error message expected');

  // 3.2 Missing source
  console.log('Testing missing source file...');
  const formMissingSource = new FormData();
  formMissingSource.append('target', new Blob([Buffer.from('fake-data')], { type: 'image/jpeg' }), 'tgt.jpg');
  const resMissingSource = await fetch(`${EXPRESS_BASE}/api/ml/face-swap`, {
    method: 'POST',
    body: formMissingSource,
  });
  console.log(`Missing source status: ${resMissingSource.status}`);
  assert.strictEqual(resMissingSource.status, 400);
  const jsonMissingSource = await resMissingSource.json();
  console.log('Missing source error:', jsonMissingSource);
  assert(jsonMissingSource.error, 'Error message expected');

  // 3.3 Invalid Content-Type
  console.log('Testing invalid Content-Type (application/json)...');
  const resInvalidType = await fetch(`${EXPRESS_BASE}/api/ml/face-swap`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ hello: 'world' }),
  });
  console.log(`Invalid Content-Type status: ${resInvalidType.status}`);
  assert.strictEqual(resInvalidType.status, 400);
  const jsonInvalidType = await resInvalidType.json();
  console.log('Invalid Content-Type error:', jsonInvalidType);

  // 3.4 Invalid File Extension (.txt)
  console.log('Testing invalid file extension (.txt)...');
  const formInvalidExt = new FormData();
  formInvalidExt.append('source', new Blob([Buffer.from('test text file')], { type: 'text/plain' }), 'test.txt');
  formInvalidExt.append('target', new Blob([Buffer.from('test text file')], { type: 'text/plain' }), 'test.txt');
  const resInvalidExt = await fetch(`${EXPRESS_BASE}/api/ml/face-swap`, {
    method: 'POST',
    body: formInvalidExt,
  });
  console.log(`Invalid extension status: ${resInvalidExt.status}`);
  assert.strictEqual(resInvalidExt.status, 400);
  const jsonInvalidExt = await resInvalidExt.json();
  console.log('Invalid extension error:', jsonInvalidExt);

  // 3.5 FastAPI Unavailable (connection refused -> 503)
  console.log('Testing FastAPI service unavailable (ECONNREFUSED -> 503)...');
  const expressModule = (await import('express')).default;
  const testApp = expressModule();
  process.env.ML_SERVICE_PORT = '49999';
  const { mlRouter: disconnectedMlRouter } = await import(`../src/ml.js?bust=${Date.now()}`);
  testApp.use('/api/ml', disconnectedMlRouter);
  const testServer = testApp.listen(5098);

  try {
    const formUnavail = new FormData();
    formUnavail.append('source', new Blob([Buffer.from('fake-img')], { type: 'image/jpeg' }), 'src.jpg');
    formUnavail.append('target', new Blob([Buffer.from('fake-img')], { type: 'image/jpeg' }), 'tgt.jpg');
    const resUnavail = await fetch('http://localhost:5098/api/ml/face-swap', {
      method: 'POST',
      body: formUnavail,
    });
    console.log(`Unavailable status: ${resUnavail.status}`);
    assert.strictEqual(resUnavail.status, 503);
    const jsonUnavail = await resUnavail.json();
    console.log('Unavailable error payload:', jsonUnavail);
    assert(jsonUnavail.error && jsonUnavail.error.includes('Unavailable'), 'Expected 503 service unavailable error message');
  } finally {
    testServer.close();
    process.env.ML_SERVICE_PORT = '8000';
  }

  console.log('All error handling tests PASSED!');
}

async function testConcurrency429() {
  console.log('\n--- Test 4: Concurrency & HTTP 429 Propagation ---');
  const pair = PAIRS[0];
  const srcBuffer = fs.readFileSync(pair.source);
  const tgtBuffer = fs.readFileSync(pair.target);

  const makeRequest = async (id: number) => {
    const form = new FormData();
    form.append('source', new Blob([srcBuffer], { type: 'image/jpeg' }), path.basename(pair.source));
    form.append('target', new Blob([tgtBuffer], { type: 'image/jpeg' }), path.basename(pair.target));
    const res = await fetch(`${EXPRESS_BASE}/api/ml/face-swap`, {
      method: 'POST',
      body: form,
    });
    const json = await res.json();
    return { id, status: res.status, json };
  };

  console.log('Launching 2 simultaneous requests to test GPU worker concurrency limit...');
  const req1Promise = makeRequest(1);
  await new Promise((r) => setTimeout(r, 25)); // slight delay so req1 grabs lock
  const req2Promise = makeRequest(2);

  const [res1, res2] = await Promise.all([req1Promise, req2Promise]);

  console.log(`Request 1 HTTP Status: ${res1.status}`);
  console.log(`Request 2 HTTP Status: ${res2.status}`);

  // One request should succeed (200) and the colliding request should receive 429
  const statuses = [res1.status, res2.status];
  assert(statuses.includes(200), 'At least one request must succeed (HTTP 200)');
  assert(statuses.includes(429), 'The concurrent overlapping request must be rejected with HTTP 429');

  const rejected = res1.status === 429 ? res1 : res2;
  console.log('429 Rejection payload:', rejected.json);
  assert(rejected.json.error, 'Rejection payload must contain error description');

  console.log('Concurrency 429 propagation PASSED!');
}

async function main() {
  console.log('================================================================');
  console.log('PHASE 6G EXPRESS ML FACE-SWAP INTEGRATION TEST SUITE');
  console.log('Target:', EXPRESS_BASE);
  console.log('================================================================');

  await testHealth();

  const results: InferenceResult[] = [];
  for (let i = 0; i < PAIRS.length; i++) {
    const res = await testInferencePair(PAIRS[i], i);
    results.push(res);
  }

  await testErrorHandling();
  await testConcurrency429();

  console.log('\n================================================================');
  console.log('ALL EXPRESS INTEGRATION TESTS COMPLETED SUCCESSFULLY!');
  console.log('================================================================');
  console.log(
    `${'Pair'.padEnd(30)} | ${'Roundtrip'.padStart(10)} | ${'Pipeline'.padStart(10)} | ${'Express'.padStart(10)} | ${'LM Error'.padStart(10)} | ${'Gain A-C'.padStart(10)} | Face OK`
  );
  console.log('-'.repeat(100));
  for (const r of results) {
    console.log(
      `${r.pair.padEnd(30)} | ${`${r.roundtripMs}ms`.padStart(10)} | ${`${r.pipelineLatencyMs}ms`.padStart(10)} | ${`${r.expressOverheadMs}ms`.padStart(10)} | ${`${r.landmarkErrorPx.toFixed(2)}px`.padStart(10)} | ${`${r.identityGain > 0 ? '+' : ''}${r.identityGain.toFixed(4)}`.padStart(10)} | ${r.faceDetected}`
    );
  }
  console.log('-'.repeat(100));
}

main().catch((err) => {
  console.error('\nTEST SUITE FAILED:', err);
  process.exit(1);
});
