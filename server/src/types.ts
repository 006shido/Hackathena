export type UserRole = 'user' | 'tester';

export interface User {
  username: string;
  role: UserRole;
  name: string;
}

export interface AuthResponse {
  token: string;
  user: User;
}

export interface Participant {
  socketId: string;
  username: string;
  role: UserRole;
  joinedAt: number;
}

export interface RoomInfo {
  roomId: string;
  participants: Participant[];
  createdAt: number;
}

export interface AttackSimulationPayload {
  roomId: string;
  faceSwap: boolean;
  voiceTransform: boolean;
  attackMode: 'none' | 'face' | 'voice' | 'combined';
  timestamp: number;
}
