import { UserRole } from './auth';

export type CallStatus =
  | 'idle'
  | 'checking-devices'
  | 'ready'
  | 'joining'
  | 'waiting'
  | 'connected'
  | 'reconnecting'
  | 'failed'
  | 'ended';

export interface ParticipantInfo {
  socketId: string;
  username: string;
  role: UserRole;
  joinedAt?: number;
}

export interface DeviceState {
  micEnabled: boolean;
  cameraEnabled: boolean;
  screenShareEnabled: boolean;
}

export interface DevicePermissionState {
  camera: 'granted' | 'denied' | 'prompt' | 'unavailable';
  microphone: 'granted' | 'denied' | 'prompt' | 'unavailable';
  error?: string;
}

export interface RoomState {
  roomId: string;
  participants: ParticipantInfo[];
}
