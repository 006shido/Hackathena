import { io } from 'socket.io-client';

const API_URL = 'http://localhost:5001';

async function login(username, password) {
  const res = await fetch(`${API_URL}/api/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error);
  return data;
}

async function runTests() {
  console.log('--- DeepTrace Integration & Acceptance Test ---');

  // 1. Authenticate Tester
  console.log('\n[Test 1] Logging in as TESTER...');
  const testerAuth = await login('tester', 'tester123');
  console.log('✓ Tester logged in successfully. Role:', testerAuth.user.role);
  if (testerAuth.user.role !== 'tester') throw new Error('Tester role mismatch');

  // 2. Authenticate Normal User
  console.log('\n[Test 2] Logging in as USER...');
  const userAuth = await login('user', 'user123');
  console.log('✓ User logged in successfully. Role:', userAuth.user.role);
  if (userAuth.user.role !== 'user') throw new Error('User role mismatch');

  const testRoomId = 'DEMO-777';

  // 3. Connect Tester socket
  console.log('\n[Test 3] Connecting Tester Socket.IO...');
  const testerSocket = io(API_URL, {
    auth: { token: testerAuth.token },
    transports: ['websocket'],
  });

  await new Promise((resolve, reject) => {
    testerSocket.on('connect', resolve);
    testerSocket.on('connect_error', reject);
  });
  console.log('✓ Tester socket connected, ID:', testerSocket.id);

  // 4. Tester joins room
  console.log('\n[Test 4] Tester joining room', testRoomId);
  const testerJoinedPromise = new Promise((resolve) => {
    testerSocket.once('room-joined', (data) => {
      console.log('✓ Tester received room-joined:', data.roomId, 'Peers:', data.peers.length);
      resolve(data);
    });
  });
  testerSocket.emit('join-room', { roomId: testRoomId });
  await testerJoinedPromise;

  // 5. Connect User socket
  console.log('\n[Test 5] Connecting Normal User Socket.IO...');
  const userSocket = io(API_URL, {
    auth: { token: userAuth.token },
    transports: ['websocket'],
  });

  await new Promise((resolve, reject) => {
    userSocket.on('connect', resolve);
    userSocket.on('connect_error', reject);
  });
  console.log('✓ User socket connected, ID:', userSocket.id);

  // 6. User joins room -> Tester should receive peer-joined, User receives room-joined
  console.log('\n[Test 6] User joining room', testRoomId);
  const peerJoinedPromise = new Promise((resolve) => {
    testerSocket.once('peer-joined', (data) => {
      console.log('✓ Tester received peer-joined event:', data.username, `(${data.role})`);
      resolve(data);
    });
  });

  const userJoinedPromise = new Promise((resolve) => {
    userSocket.once('room-joined', (data) => {
      console.log('✓ User received room-joined event:', data.roomId, 'Peers count:', data.peers.length);
      resolve(data);
    });
  });

  userSocket.emit('join-room', { roomId: testRoomId });
  await Promise.all([peerJoinedPromise, userJoinedPromise]);

  // 7. WebRTC Offer / Answer exchange
  console.log('\n[Test 7] Testing WebRTC Offer / Answer signaling...');
  const offerReceivedPromise = new Promise((resolve) => {
    userSocket.once('webrtc-offer', (data) => {
      console.log('✓ User received webrtc-offer from', data.senderId);
      resolve(data);
    });
  });

  testerSocket.emit('webrtc-offer', {
    roomId: testRoomId,
    sdp: { type: 'offer', sdp: 'v=0\r\no=tester 123 456 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\n' },
  });
  await offerReceivedPromise;

  const answerReceivedPromise = new Promise((resolve) => {
    testerSocket.once('webrtc-answer', (data) => {
      console.log('✓ Tester received webrtc-answer from', data.senderId);
      resolve(data);
    });
  });

  userSocket.emit('webrtc-answer', {
    roomId: testRoomId,
    sdp: { type: 'answer', sdp: 'v=0\r\no=user 789 101 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\n' },
  });
  await answerReceivedPromise;

  // 8. Server-side Role Validation: Tester triggers Attack Simulation
  console.log('\n[Test 8] Tester activating Attack Simulation (Face + Voice Combined)...');
  const userAttackAlertPromise = new Promise((resolve) => {
    userSocket.once('peer-attack-state', (data) => {
      console.log('✓ User received simulated peer-attack-state:', data.attackMode, 'Face:', data.faceSwap, 'Voice:', data.voiceTransform);
      resolve(data);
    });
  });

  testerSocket.emit('attack-simulation-update', {
    roomId: testRoomId,
    faceSwap: true,
    voiceTransform: true,
    attackMode: 'combined',
  });
  await userAttackAlertPromise;

  // 9. Server-side Role Validation: Non-tester USER attempts to trigger Attack Simulation (Must be REJECTED!)
  console.log('\n[Test 9] Non-tester User attempting unauthorized Attack Simulation (Must be REJECTED)...');
  const attackRejectionPromise = new Promise((resolve) => {
    userSocket.once('attack-error', (data) => {
      console.log('✓ Server correctly rejected User attack simulation:', data.message);
      resolve(data);
    });
  });

  userSocket.emit('attack-simulation-update', {
    roomId: testRoomId,
    faceSwap: true,
    voiceTransform: false,
    attackMode: 'face',
  });
  await attackRejectionPromise;

  // 10. Room Capacity Limit (Max 2 participants): 3rd participant attempt
  console.log('\n[Test 10] 3rd participant attempting to join full room (Must be REJECTED)...');
  const thirdAuth = await login('user', 'user123');
  const thirdSocket = io(API_URL, {
    auth: { token: thirdAuth.token },
    transports: ['websocket'],
  });

  await new Promise((resolve) => thirdSocket.on('connect', resolve));
  const fullRoomPromise = new Promise((resolve) => {
    thirdSocket.once('room-error', (data) => {
      console.log('✓ 3rd participant correctly rejected with message:', `"${data.message}"`);
      resolve(data);
    });
  });

  thirdSocket.emit('join-room', { roomId: testRoomId });
  const errorData = await fullRoomPromise;
  if (errorData.message !== 'This room is full.') {
    throw new Error('Expected "This room is full." message, got: ' + errorData.message);
  }

  // 11. Participant leave & disconnect
  console.log('\n[Test 11] Participant leaving room...');
  const peerLeftPromise = new Promise((resolve) => {
    testerSocket.once('peer-left', (data) => {
      console.log('✓ Tester received peer-left notification for:', data.username);
      resolve(data);
    });
  });

  userSocket.emit('leave-room', { roomId: testRoomId });
  await peerLeftPromise;

  // Cleanup
  testerSocket.disconnect();
  userSocket.disconnect();
  thirdSocket.disconnect();

  console.log('\n==========================================');
  console.log('ALL 11 INTEGRATION & SECURITY TESTS PASSED!');
  console.log('==========================================\n');
}

runTests().catch((err) => {
  console.error('Test Failed:', err);
  process.exit(1);
});
