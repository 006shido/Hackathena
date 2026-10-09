import { io,Socket } from 'socket.io-client';
import { AttackMode } from '../types/attack';
import { ParticipantInfo } from '../types/call';

export interface RoomJoinedData {
  roomId: string;
  self: ParticipantInfo;
  peers: ParticipantInfo[];
}

export interface PeerJoinedData {
  socketId: string;
  username: string;
  role: 'user' | 'tester';
}

export interface SignalingEvents {
  onRoomJoined?: (data: RoomJoinedData) => void;
  onPeerJoined?: (data: PeerJoinedData) => void;
  onPeerLeft?: (data: { socketId: string; username?: string }) => void;
  onOffer?: (data: { senderId: string; sdp: RTCSessionDescriptionInit }) => void;
  onAnswer?: (data: { senderId: string; sdp: RTCSessionDescriptionInit }) => void;
  onIceCandidate?: (data: { senderId: string; candidate: RTCIceCandidateInit }) => void;
  onRoomError?: (data: { message: string }) => void;
  onPeerAttackState?: (data: { senderId: string; attackMode: AttackMode; faceSwap: boolean; voiceTransform: boolean }) => void;
  onAttackError?: (data: { message: string }) => void;
  onConnectError?: (err: Error) => void;
  onDisconnect?: (reason: string) => void;
}

class SignalingService {
  private socket: Socket | null = null;
  private listeners: SignalingEvents = {};

  public connect(token: string, listeners: SignalingEvents = {}): Socket {
    this.listeners = listeners;

    if (this.socket) {
      this.socket.disconnect();
    }

    // Connect to server origin (proxied automatically by Vite in dev, or same host in prod)
    const serverUrl = window.location.origin;

    this.socket = io(serverUrl, {
      auth: { token },
      transports: ['websocket', 'polling'],
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
    });

    this.setupListeners();
    return this.socket;
  }


  private setupListeners() {
    if (!this.socket) return;

    this.socket.on('connect_error', (err) => {
      console.error('[Signaling] Connection error:', err.message);
      this.listeners.onConnectError?.(err);
    });

    this.socket.on('room-joined', (data: RoomJoinedData) => {
      this.listeners.onRoomJoined?.(data);
    });

    this.socket.on('peer-joined', (data: PeerJoinedData) => {
      this.listeners.onPeerJoined?.(data);
    });

    this.socket.on('peer-left', (data: { socketId: string; username?: string }) => {
      this.listeners.onPeerLeft?.(data);
    });

    this.socket.on('webrtc-offer', (data: { senderId: string; sdp: RTCSessionDescriptionInit }) => {
      this.listeners.onOffer?.(data);
    });

    this.socket.on('webrtc-answer', (data: { senderId: string; sdp: RTCSessionDescriptionInit }) => {
      this.listeners.onAnswer?.(data);
    });

    this.socket.on('webrtc-ice', (data: { senderId: string; candidate: RTCIceCandidateInit }) => {
      this.listeners.onIceCandidate?.(data);
    });

    this.socket.on('room-error', (data: { message: string }) => {
      console.warn('[Signaling] room-error:', data.message);
      this.listeners.onRoomError?.(data);
    });

    this.socket.on('peer-attack-state', (data: { senderId: string; attackMode: AttackMode; faceSwap: boolean; voiceTransform: boolean }) => {
      this.listeners.onPeerAttackState?.(data);
    });

    this.socket.on('attack-error', (data: { message: string }) => {
      console.error('[Signaling] attack-error:', data.message);
      this.listeners.onAttackError?.(data);
    });

    this.socket.on('disconnect', (reason: string) => {
      this.listeners.onDisconnect?.(reason);
    });
  }

  public joinRoom(roomId: string) {
    if (!this.socket?.connected) {
      console.warn('[Signaling] Socket not connected yet, queuing join');
      this.socket?.once('connect', () => {
        this.socket?.emit('join-room', { roomId });
      });
      return;
    }
    this.socket.emit('join-room', { roomId });
  }

  public sendOffer(roomId: string, sdp: RTCSessionDescriptionInit) {
    this.socket?.emit('webrtc-offer', { roomId, sdp });
  }

  public sendAnswer(roomId: string, sdp: RTCSessionDescriptionInit) {
    this.socket?.emit('webrtc-answer', { roomId, sdp });
  }

  public sendIceCandidate(roomId: string, candidate: RTCIceCandidateInit) {
    this.socket?.emit('webrtc-ice', { roomId, candidate });
  }

  public sendAttackUpdate(roomId: string, faceSwap: boolean, voiceTransform: boolean, mode: AttackMode) {
    this.socket?.emit('attack-simulation-update', {
      roomId,
      faceSwap,
      voiceTransform,
      attackMode: mode,
      timestamp: Date.now(),
    });
  }

  public leaveRoom(roomId: string) {
    this.socket?.emit('leave-room', { roomId });
  }

  public disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }

}

export const signalingService = new SignalingService();
