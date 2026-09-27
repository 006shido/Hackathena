export type AttackMode = 'none' | 'face' | 'voice' | 'combined';

export interface AttackState {
  faceSwap: boolean;
  voiceTransform: boolean;
  mode: AttackMode;
  facePreset: 'neural-clone' | 'biometric-mask' | 'synthetic-executive';
  voicePreset: 'robotic-vocoder' | 'deep-pitch-neural' | 'synthetic-clone';
}

export interface AttackStatusUpdate {
  mode: AttackMode;
  faceSwap: boolean;
  voiceTransform: boolean;
  timestamp: number;
}
