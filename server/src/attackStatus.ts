import type { AttackSimulationPayload } from './types.js';

export function validateAttackStatus(payload: unknown, role: string, rooms: Set<string>) {
  if (role !== 'tester') return { error: 'Only tester accounts may report simulation changes.' } as const;
  if (!payload || typeof payload !== 'object') return { error: 'Invalid simulation status.' } as const;
  const input = payload as Partial<AttackSimulationPayload>;
  if (typeof input.roomId !== 'string' || typeof input.faceSwap !== 'boolean' || typeof input.voiceTransform !== 'boolean') {
    return { error: 'Invalid simulation status.' } as const;
  }
  const roomId = input.roomId.toUpperCase().trim();
  if (!roomId || !rooms.has(roomId)) return { error: 'Join this room before reporting simulation changes.' } as const;
  const attackMode = input.faceSwap ? (input.voiceTransform ? 'combined' : 'face') : (input.voiceTransform ? 'voice' : 'none');
  return { state: { roomId, faceSwap: input.faceSwap, voiceTransform: input.voiceTransform, attackMode } } as const;
}
