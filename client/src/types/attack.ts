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

export interface FaceBlendConfig {
  featherRadius: number;
  skinToneMatch: number;
  lightingTransfer: number;
  mouthBlend: number;
  sensorGrain: number;
  maskInset: number;
  naturalEyes: boolean;
}

export const DEFAULT_FACE_BLEND_CONFIG: FaceBlendConfig = {
  featherRadius: 16,
  skinToneMatch: 0.85,
  lightingTransfer: 0.65,
  mouthBlend: 0.90,
  sensorGrain: 0.35,
  maskInset: 0.06,
  naturalEyes: true,
};
