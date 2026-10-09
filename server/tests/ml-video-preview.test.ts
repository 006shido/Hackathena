import assert from 'node:assert/strict';
import http from 'node:http';
import { once } from 'node:events';
import express from 'express';
import { generateToken } from '../src/auth.js';

const seen: string[] = [];
const upstream = http.createServer((request, response) => {
  seen.push(String(request.headers['x-ml-owner']));
  response.writeHead(200, { 'content-type': 'image/jpeg', 'x-face-detected': 'true', 'x-face-changed': 'false', 'x-video-backend': 'research-reference' });
  response.end(Buffer.from([0xff, 0xd8, 0xff, 0xd9]));
});
upstream.listen(0, '127.0.0.1');
await once(upstream, 'listening');
const upstreamAddress = upstream.address() as import('node:net').AddressInfo;
process.env.ML_SERVICE_PORT = String(upstreamAddress.port);
const { mlRouter } = await import('../src/ml.js');
const app = express();
app.use('/api/ml', mlRouter);
const server = app.listen(0, '127.0.0.1');
await once(server, 'listening');
const address = server.address() as import('node:net').AddressInfo;
const base = `http://127.0.0.1:${address.port}/api/ml/video`;
try {
  const normal = generateToken({ username: 'user', role: 'user', name: 'Normal User' });
  const tester = generateToken({ username: 'tester', role: 'tester', name: 'Tester' });
  assert.equal((await fetch(`${base}/sessions`, { method: 'POST' })).status, 401);
  assert.equal((await fetch(`${base}/sessions`, { method: 'POST', headers: { Authorization: `Bearer ${normal}` } })).status, 403);
  assert.equal(seen.length, 0);
  assert.equal((await fetch(`${base}/sessions/invalid/frame`, { method: 'POST', headers: { Authorization: `Bearer ${tester}` } })).status, 400);
  const result = await fetch(`${base}/sessions/${'a'.repeat(32)}/frame`, { method: 'POST', headers: {
    Authorization: `Bearer ${tester}`, 'content-type': 'multipart/form-data; boundary=test', 'x-ml-owner': 'spoofed',
  }, body: '--test--' });
  assert.equal(result.status, 200);
  assert.equal(result.headers.get('x-face-detected'), 'true');
  assert.equal(result.headers.get('x-video-backend'), 'research-reference');
  assert.equal(result.headers.get('x-face-changed'), 'false');
  assert.deepEqual([...new Uint8Array(await result.arrayBuffer())], [0xff, 0xd8, 0xff, 0xd9]);
  assert.deepEqual(seen, ['tester']);
  console.log('PASS: role authorization, session validation, trusted owner forwarding and binary response.');
} finally {
  server.closeAllConnections(); upstream.closeAllConnections();
  server.close(); upstream.close();
}
