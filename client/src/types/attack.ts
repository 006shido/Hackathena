export type AttackMode = 'none' | 'face' | 'voice' | 'combined';

export type FacePreset =
  | 'neural-clone'
  | 'synthetic-executive'
  | 'mona-lisa'
  | 'cyber-agent'
  | 'astronaut'
  | 'custom-upload'
  | 'biometric-mask'
  | 'cyber-filter';

export interface AttackState {
  faceSwap: boolean;
  voiceTransform: boolean;
  mode: AttackMode;
  facePreset: FacePreset;
  voicePreset: 'robotic-vocoder' | 'deep-pitch-neural' | 'synthetic-clone';
}


export interface AttackStatusUpdate {
  mode: AttackMode;
  faceSwap: boolean;
  voiceTransform: boolean;
  timestamp: number;
}
