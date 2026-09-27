class RoomManager {
    rooms = new Map();
    getOrCreateRoom(roomId) {
        const normalizedId = roomId.toUpperCase().trim();
        if (!this.rooms.has(normalizedId)) {
            this.rooms.set(normalizedId, {
                roomId: normalizedId,
                participants: new Map(),
                createdAt: Date.now(),
            });
        }
        return this.rooms.get(normalizedId);
    }
    getRoom(roomId) {
        const normalizedId = roomId.toUpperCase().trim();
        const room = this.rooms.get(normalizedId);
        if (!room)
            return null;
        return {
            roomId: room.roomId,
            participants: Array.from(room.participants.values()),
            createdAt: room.createdAt,
        };
    }
    canJoin(roomId, socketId) {
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
    addParticipant(roomId, participant) {
        const check = this.canJoin(roomId, participant.socketId);
        if (!check.allowed) {
            return { success: false, reason: check.reason };
        }
        const room = this.getOrCreateRoom(roomId);
        room.participants.set(participant.socketId, participant);
        return { success: true };
    }
    removeParticipant(socketId) {
        const removed = [];
        for (const [roomId, room] of this.rooms.entries()) {
            if (room.participants.has(socketId)) {
                const participant = room.participants.get(socketId);
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
    getOtherParticipant(roomId, currentSocketId) {
        const normalizedId = roomId.toUpperCase().trim();
        const room = this.rooms.get(normalizedId);
        if (!room)
            return null;
        for (const [socketId, participant] of room.participants.entries()) {
            if (socketId !== currentSocketId) {
                return participant;
            }
        }
        return null;
    }
}
export const roomManager = new RoomManager();
