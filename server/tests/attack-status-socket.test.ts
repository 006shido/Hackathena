import assert from 'node:assert/strict';
import http from 'node:http';
import { once } from 'node:events';
import { spawn } from 'node:child_process';
import { createRequire } from 'node:module';
import { generateToken } from '../src/auth.js';

const requireClient = createRequire(new URL('../../client/package.json', import.meta.url));
const { io } = requireClient('socket.io-client');
const reservation = http.createServer();
reservation.listen(0, '127.0.0.1');
await once(reservation, 'listening');
const port = (reservation.address() as import('node:net').AddressInfo).port;
await new Promise<void>(resolve => reservation.close(() => resolve()));
const server = spawn(process.execPath, ['dist/index.js'], {
  cwd: new URL('../', import.meta.url), env: { ...process.env, PORT: String(port) },
  windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'],
});
let logs = '';
server.stdout.on('data', data => { logs += String(data); });
server.stderr.on('data', data => { logs += String(data); });
const sockets: any[] = [];
function event(socket: any, name: string): Promise<any> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { socket.off(name, listener); reject(new Error(`Timed out: ${name}`)); }, 5000);
    const listener = (data: any) => { clearTimeout(timer); resolve(data); };
    socket.once(name, listener);
  });
}
async function connect(name: string, role: 'tester' | 'user') {
  const socket = io(`http://127.0.0.1:${port}`, {
    autoConnect: false, transports: ['websocket'], reconnection: false,
    auth: { token: generateToken({ username: name, name, role }) },
  });
  sockets.push(socket);
  const connected = event(socket, 'connect'); socket.connect(); await connected;
  return socket;
}
async function join(socket: any, roomId: string) {
  const joined = event(socket, 'room-joined'); socket.emit('join-room', { roomId }); await joined;
}
try {
  let ready = false;
  for (let attempt = 0; attempt < 60; attempt++) {
    if (server.exitCode !== null) throw new Error(`Server stopped: ${logs}`);
    try { ready = (await fetch(`http://127.0.0.1:${port}/api/rooms/STATUS`)).ok; } catch {}
    if (ready) break;
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  assert(ready, `Server did not start: ${logs}`);
  const tester = await connect('status-tester', 'tester');
  await join(tester, 'STATUS');
  const acknowledged = event(tester, 'attack-simulation-ack');
  tester.emit('attack-simulation-update', { roomId: 'STATUS', faceSwap: true, voiceTransform: false, attackMode: 'face' });
  await acknowledged;
  const user = await connect('status-user', 'user');
  const snapshot = event(user, 'peer-attack-state');
  await join(user, 'STATUS');
  const current = await snapshot;
  assert.equal(current.faceSwap, true);
  assert.equal(current.voiceTransform, false);
  assert.equal(current.senderId, tester.id);
  const otherTester = await connect('other-tester', 'tester');
  await join(otherTester, 'OTHER');
  let unwanted = 0;
  const count = () => unwanted++;
  user.on('peer-attack-state', count);
  const denied = event(otherTester, 'attack-error');
  otherTester.emit('attack-simulation-update', { roomId: 'STATUS', faceSwap: true, voiceTransform: true, attackMode: 'combined' });
  await denied;
  const userDenied = event(user, 'attack-error');
  user.emit('attack-simulation-update', { roomId: 'STATUS', faceSwap: true, voiceTransform: true, attackMode: 'combined' });
  await userDenied;
  await new Promise(resolve => setTimeout(resolve, 100));
  assert.equal(unwanted, 0, 'An unauthorized state was delivered.');
  user.off('peer-attack-state', count);
  const voice = event(user, 'peer-attack-state');
  tester.emit('attack-simulation-update', { roomId: 'STATUS', faceSwap: false, voiceTransform: true, attackMode: 'face' });
  assert.equal((await voice).attackMode, 'voice', 'Mode did not reflect actual video/audio flags.');
  const stopped = event(user, 'peer-attack-state');
  tester.emit('attack-simulation-update', { roomId: 'STATUS', faceSwap: false, voiceTransform: false, attackMode: 'none' });
  assert.equal((await stopped).attackMode, 'none');
  const departed = event(user, 'peer-left'); tester.disconnect();
  await departed;
  console.log('PASS: authenticated late-join snapshot, room isolation, role checks, voice/stop updates and peer teardown.');
} finally {
  sockets.forEach(socket => socket.disconnect());
  server.kill();
  if (server.exitCode === null) await once(server, 'exit');
}
