export function validateAttackStatus(payload, role, rooms) {
    if (role !== 'tester')
        return { error: 'Only tester accounts may report simulation changes.' };
    if (!payload || typeof payload !== 'object')
        return { error: 'Invalid simulation status.' };
    const input = payload;
    if (typeof input.roomId !== 'string' || typeof input.faceSwap !== 'boolean' || typeof input.voiceTransform !== 'boolean') {
        return { error: 'Invalid simulation status.' };
    }
    const roomId = input.roomId.toUpperCase().trim();
    if (!roomId || !rooms.has(roomId))
        return { error: 'Join this room before reporting simulation changes.' };
    const attackMode = input.faceSwap ? (input.voiceTransform ? 'combined' : 'face') : (input.voiceTransform ? 'voice' : 'none');
    return { state: { roomId, faceSwap: input.faceSwap, voiceTransform: input.voiceTransform, attackMode } };
}
