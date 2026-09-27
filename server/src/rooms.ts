import { Participant, RoomInfo } from './types.js';

class RoomManager {
  private rooms: Map<string, {
    roomId: string;
    participants: Map<string, Participant>;
    createdAt: number;
  }> = new Map();

  public getOrCreateRoom(roomId: string) {
    const normalizedId = roomId.toUpperCase().trim();
    if (!this.rooms.has(normalizedId)) {
      this.rooms.set(normalizedId, {
        roomId: normalizedId,
        participants: new Map(),
        createdAt: Date.now(),
      });
    }
    return this.rooms.get(normalizedId)!;
  }

  public getRoom(roomId: string): RoomInfo | null {
    const normalizedId = roomId.toUpperCase().trim();
    const room = this.rooms.get(normalizedId);
    if (!room) return null;
    return {
      roomId: room.roomId,
      participants: Array.from(room.participants.values()),
      createdAt: room.createdAt,
    };
  }

  public canJoin(roomId: string, socketId: string): { allowed: boolean; reason?: string } {
    const normalizedId = roomId.toUpperCase().trim();
    const room = this.rooms.get(normalizedId);
    if (!room) {
      // Room doesn't exist yet, can be created
      return { allowed: true };
    }

    if (room.participants.has(socketId)) {
      // Already in room
      return { allowed: true };
    }

    if (room.participants.size >= 2) {
      return { allowed: false, reason: 'This room is full.' };
    }

    return { allowed: true };
  }

  public addParticipant(roomId: string, participant: Participant): { success: boolean; reason?: string } {
    const check = this.canJoin(roomId, participant.socketId);
    if (!check.allowed) {
      return { success: false, reason: check.reason };
    }

    const room = this.getOrCreateRoom(roomId);
    room.participants.set(participant.socketId, participant);
    return { success: true };
  }

  public removeParticipant(socketId: string): { roomId: string; participant: Participant }[] {
    const removed: { roomId: string; participant: Participant }[] = [];

    for (const [roomId, room] of this.rooms.entries()) {
      if (room.participants.has(socketId)) {
        const participant = room.participants.get(socketId)!;
        room.participants.delete(socketId);
        removed.push({ roomId, participant });

        // Clean up empty room
        if (room.participants.size === 0) {
          this.rooms.delete(roomId);
        }
      }
    }

    return removed;
  }

  public getOtherParticipant(roomId: string, currentSocketId: string): Participant | null {
    const normalizedId = roomId.toUpperCase().trim();
    const room = this.rooms.get(normalizedId);
    if (!room) return null;

    for (const [socketId, participant] of room.participants.entries()) {
      if (socketId !== currentSocketId) {
        return participant;
      }
    }
    return null;
  }
}

export const roomManager = new RoomManager();
