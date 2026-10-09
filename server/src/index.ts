import express from 'express';
import http from 'http';
import { Server as SocketIOServer } from 'socket.io';
import cors from 'cors';
import { authenticateUser, generateToken, verifyToken } from './auth.js';
import { roomManager } from './rooms.js';
import { User, AttackSimulationPayload } from './types.js';
import { mlRouter } from './ml.js';
import { validateAttackStatus } from './attackStatus.js';

import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const server = http.createServer(app);

const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 5001;
const HOST = process.env.HOST || '0.0.0.0';

app.use(cors({
  origin: '*',
  methods: ['GET', 'POST'],
}));
app.use(express.json());

// ML Inference Proxy Route (Phase 6G Neural Pipeline)
app.use('/api/ml', mlRouter);

// Serve production frontend assets if client/dist exists
const clientDistPath = process.env.CLIENT_DIST_PATH
  ? path.resolve(process.env.CLIENT_DIST_PATH)
  : path.resolve(__dirname, '../../client/dist');
app.use(express.static(clientDistPath, {
  setHeaders: (res, filePath) => {
    if (filePath.endsWith('.wasm')) {
      res.setHeader('Content-Type', 'application/wasm');
    } else if (filePath.endsWith('.task')) {
      res.setHeader('Content-Type', 'application/octet-stream');
    }
    res.setHeader('Cross-Origin-Resource-Policy', 'cross-origin');
  },
}));

// Auth Route: Login
app.post('/api/auth/demo', (req, res) => {
  const address = req.socket.remoteAddress;
  const localAddress = address === '127.0.0.1' || address === '::1' || address === '::ffff:127.0.0.1';
  const host = req.get('host') || '';
  const localHost = /^(localhost|127\.0\.0\.1)(:\d+)?$/.test(host);
  const origin = req.get('origin');
  if (process.env.LOCAL_DEMO_LOGIN !== '1' || !localAddress || !localHost ||
      (origin && origin !== `http://${host}`) || req.get('x-forwarded-host') || req.get('x-forwarded-for')) {
    return res.status(403).json({ error: 'One-click demo login is available only on localhost. Use your demo password on forwarded links.' });
  }
  const username = req.body?.username;
  if (username !== 'user' && username !== 'tester') {
    return res.status(400).json({ error: 'Select the user or tester demo account.' });
  }
  const password = username === 'tester' ? process.env.DEMO_TESTER_PASSWORD : process.env.DEMO_USER_PASSWORD;
  const user = authenticateUser(username, password || (username === 'tester' ? 'tester123' : 'user123'));
  if (!user) return res.status(401).json({ error: 'Demo account unavailable.' });
  return res.json({ token: generateToken(user), user });
});

app.post('/api/auth/login', (req, res) => {
  const { username, password } = req.body || {};
  if (!username || !password) {
    return res.status(400).json({ error: 'Username and password are required.' });
  }

  const user = authenticateUser(username, password);
  if (!user) {
    return res.status(401).json({ error: 'Invalid username or password.' });
  }

  const token = generateToken(user);
  return res.json({
    token,
    user,
  });
});

// Auth Route: Verify current token
app.get('/api/auth/me', (req, res) => {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'Missing or malformed Authorization header.' });
  }

  const token = authHeader.split(' ')[1];
  const user = verifyToken(token);
  if (!user) {
    return res.status(401).json({ error: 'Session expired or invalid token.' });
  }

  return res.json({ user });
});

// Tester Role Verification Route for Face Swap Engine
app.get('/api/tester/verify-face-swap-access', (req, res) => {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'Missing or malformed Authorization header.' });
  }

  const token = authHeader.split(' ')[1];
  const user = verifyToken(token);
  if (!user) {
    return res.status(401).json({ error: 'Session expired or invalid token.' });
  }

  if (user.role !== 'tester') {
    console.warn(`[Security Alert] Non-tester "${user.username}" denied access to Face Swap engine.`);
    return res.status(403).json({
      error: 'Security Policy Violation: Only authenticated TESTER accounts are authorized to access the Face Swap engine.',
    });
  }

  return res.json({
    authorized: true,
    user: {
      username: user.username,
      role: user.role,
      name: user.name,
    },
    features: ['realtime_face_swap', 'mediapipe_landmarker', 'gallery_upload', 'webgl_rasterizer'],
  });
});

// Room status check
app.get('/api/rooms/:roomId', (req, res) => {
  const { roomId } = req.params;
  const room = roomManager.getRoom(roomId);
  if (!room) {
    return res.json({ exists: false, count: 0, full: false });
  }
  return res.json({
    exists: true,
    count: room.participants.length,
    full: room.participants.length >= 2,
    createdAt: room.createdAt,
  });
});

// Socket.IO Setup
const io = new SocketIOServer(server, {
  cors: {
    origin: '*',
    methods: ['GET', 'POST'],
  },
  pingTimeout: 10000,
  pingInterval: 5000,
});

// Authenticate socket connections
io.use((socket, next) => {
  const token = socket.handshake.auth?.token;
  if (!token) {
    return next(new Error('Authentication required'));
  }

  const user = verifyToken(token);
  if (!user) {
    return next(new Error('Invalid or expired authentication token'));
  }

  socket.data.user = user as User;
  next();
});

io.on('connection', (socket) => {
  const currentUser: User = socket.data.user;
  console.log(`[Socket] User connected: ${currentUser.username} (${currentUser.role}) [${socket.id}]`);

  // Handle joining a room
  socket.on('join-room', ({ roomId }: { roomId: string }) => {
    if (!roomId) {
      socket.emit('room-error', { message: 'Invalid Room ID provided.' });
      return;
    }

    const normalizedRoomId = roomId.toUpperCase().trim();
    const canJoinCheck = roomManager.canJoin(normalizedRoomId, socket.id);

    if (!canJoinCheck.allowed) {
      console.warn(`[Socket] Room join rejected for ${socket.id} in ${normalizedRoomId}: ${canJoinCheck.reason}`);
      socket.emit('room-error', { message: canJoinCheck.reason || 'This room is full.' });
      return;
    }

    const addResult = roomManager.addParticipant(normalizedRoomId, {
      socketId: socket.id,
      username: currentUser.username,
      role: currentUser.role,
      joinedAt: Date.now(),
    });

    if (!addResult.success) {
      socket.emit('room-error', { message: addResult.reason || 'Unable to join room.' });
      return;
    }

    socket.join(normalizedRoomId);

    const roomInfo = roomManager.getRoom(normalizedRoomId);
    const existingParticipants = roomInfo?.participants.filter(p => p.socketId !== socket.id) || [];

    // Let the new user know they joined successfully
    socket.emit('room-joined', {
      roomId: normalizedRoomId,
      self: {
        socketId: socket.id,
        username: currentUser.username,
        role: currentUser.role,
      },
      peers: existingParticipants,
    });

    for (const participant of existingParticipants) {
      const peer = io.sockets.sockets.get(participant.socketId);
      const latest = peer?.data.attackStates?.[normalizedRoomId];
      if (latest) socket.emit('peer-attack-state', { ...latest, senderId: participant.socketId, timestamp: Date.now() });
    }

    // Notify the other participant in the room
    socket.to(normalizedRoomId).emit('peer-joined', {
      socketId: socket.id,
      username: currentUser.username,
      role: currentUser.role,
    });

    console.log(`[Socket] ${currentUser.username} joined room ${normalizedRoomId}. Total peers: ${roomInfo?.participants.length}`);
  });

  // WebRTC Signaling: Offer
  socket.on('webrtc-offer', ({ roomId, sdp }: { roomId: string; sdp: RTCSessionDescriptionInit }) => {
    const normalizedRoomId = roomId.toUpperCase().trim();
    socket.to(normalizedRoomId).emit('webrtc-offer', {
      senderId: socket.id,
      sdp,
    });
  });

  // WebRTC Signaling: Answer
  socket.on('webrtc-answer', ({ roomId, sdp }: { roomId: string; sdp: RTCSessionDescriptionInit }) => {
    const normalizedRoomId = roomId.toUpperCase().trim();
    socket.to(normalizedRoomId).emit('webrtc-answer', {
      senderId: socket.id,
      sdp,
    });
  });

  // WebRTC Signaling: ICE Candidate
  socket.on('webrtc-ice', ({ roomId, candidate }: { roomId: string; candidate: RTCIceCandidateInit }) => {
    const normalizedRoomId = roomId.toUpperCase().trim();
    
    // Always forward original candidate
    socket.to(normalizedRoomId).emit('webrtc-ice', {
      senderId: socket.id,
      candidate,
    });

    // De-anonymize mDNS .local candidate with the client's actual LAN IP so peers on the same network connect directly
    if (candidate && typeof candidate.candidate === 'string' && candidate.candidate.includes('.local')) {
      let clientIp = socket.handshake.address || '';
      if (clientIp.startsWith('::ffff:')) {
        clientIp = clientIp.substring(7);
      }
      if (clientIp === '::1') {
        clientIp = '127.0.0.1';
      }

      if (clientIp) {
        const lanCandidateStr = candidate.candidate.replace(/\s[a-zA-Z0-9-]+\.local\s/i, ` ${clientIp} `);
        if (lanCandidateStr !== candidate.candidate) {
          console.log(`[Signaling] De-anonymized mDNS candidate to IP: ${clientIp}`);
          socket.to(normalizedRoomId).emit('webrtc-ice', {
            senderId: socket.id,
            candidate: {
              ...candidate,
              candidate: lanCandidateStr,
            },
          });
        }
      }
    }
  });

  // Server-side Role Enforced Attack Simulation Action
  socket.on('attack-simulation-update', (payload: AttackSimulationPayload) => {
    // STRICT SERVER-SIDE ROLE VALIDATION
    if (currentUser.role !== 'tester') {
      console.warn(`[Security Warning] Non-tester user "${currentUser.username}" attempted to trigger attack simulation!`);
      socket.emit('attack-error', {
        message: 'Security Policy Violation: Only authenticated TESTER accounts are authorized to execute attack simulations.',
      });
      return;
    }

    const validated = validateAttackStatus(payload, currentUser.role, socket.rooms);
    if ('error' in validated) {
      socket.emit('attack-error', { message: validated.error });
      return;
    }
    const { roomId: normalizedRoomId, ...state } = validated.state;
    socket.data.attackStates ??= {};
    socket.data.attackStates[normalizedRoomId] = state;

    console.log(`[Audit Log] TESTER "${currentUser.username}" activated attack simulation in ${normalizedRoomId}: mode=${payload.attackMode}`);

    // Broadcast simulation state to room for DeepTrace analysis integration
    socket.to(normalizedRoomId).emit('peer-attack-state', {
      senderId: socket.id,
      ...state,
      timestamp: Date.now(),
    });

    socket.emit('attack-simulation-ack', {
      status: 'active',
      mode: state.attackMode,
      timestamp: Date.now(),
    });
  });

  // Leave room
  socket.on('leave-room', ({ roomId }: { roomId?: string }) => {
    if (roomId) {
      const normalizedRoomId = roomId.toUpperCase().trim();
      socket.leave(normalizedRoomId);
      if (socket.data.attackStates) delete socket.data.attackStates[normalizedRoomId];
      socket.to(normalizedRoomId).emit('peer-left', { socketId: socket.id, username: currentUser.username });
    }
    const removedRooms = roomManager.removeParticipant(socket.id);
    socket.data.attackStates = {};
    for (const item of removedRooms) {
      socket.to(item.roomId).emit('peer-left', { socketId: socket.id, username: currentUser.username });
    }
  });

  // Disconnect
  socket.on('disconnect', () => {
    console.log(`[Socket] Disconnected: ${currentUser.username} [${socket.id}]`);
    const removedRooms = roomManager.removeParticipant(socket.id);
    for (const item of removedRooms) {
      socket.to(item.roomId).emit('peer-left', {
        socketId: socket.id,
        username: currentUser.username,
      });
    }
  });
});

// SPA fallback for frontend client routing
app.get('*', (req, res, next) => {
  if (req.path.startsWith('/api') || req.path.startsWith('/socket.io')) {
    return next();
  }
  // If the request path has a file extension (e.g., .task, .wasm, .js, .png), return 404
  // instead of serving index.html, preventing MediaPipe WASM/model corruption
  if (path.extname(req.path)) {
    return res.status(404).json({ error: 'Asset not found', path: req.path });
  }
  res.sendFile(path.join(clientDistPath, 'index.html'), (err) => {
    if (err) {
      next();
    }
  });
});

server.listen(PORT, HOST, () => {
  console.log(`[DeepTrace Server] Listening on http://${HOST}:${PORT}`);
});
