/**
 * Real-time Face Manipulation & AI Face Swap Pipeline (Tester Exclusive)
 * 
 * Exclusively for Tester accounts to execute AI-powered face swap simulation during live WebRTC calls.
 * Features:
 * - MediaPipe Face Landmarker for real-time 478 3D facial landmark detection
 * - GPU-accelerated WebGL piecewise affine Delaunay mesh rasterizer (854 triangles)
 * - Automatic face detection and validation for device gallery photograph uploads
 * - Restrictive facial mask preserving 100% of original hair, ears, neck, body, and background
 * - Expression transfer preserving blinking, mouth opening, lip movements, and head rotation
 * - Dynamic skin-tone and lighting color matching
 * - Seamless feathered contour blending eliminating harsh edges
 * - Hidden HTML5 <canvas> stream interception via canvas.captureStream(30)
 * - Seamless non-destructive track replacement into outgoing WebRTC connection
 * - Automatic graceful degradation: unaltered webcam frame if face is lost
 */

import { FacePreset, FaceBlendConfig } from '../types/attack';
import { FaceSwapEngine, GalleryFaceValidationResult, FaceSwapTelemetry } from './faceSwapEngine';

export type { FacePreset, FaceBlendConfig, GalleryFaceValidationResult, FaceSwapTelemetry };

export interface LandmarkPoint {
  x: number;
  y: number;
  z?: number;
}

export class FaceSimulationPipeline {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private videoEl: HTMLVideoElement;
  private active = false;
  private animFrameId: number | null = null;
  private originalTrack: MediaStreamTrack | null = null;
  private outputStream: MediaStream | null = null;
  private preset: FacePreset = 'neural-clone';

  // AI Face Swap Engine (MediaPipe + WebGL GPU)
  private engine: FaceSwapEngine;

  constructor() {
    this.canvas = document.createElement('canvas');
    this.canvas.width = 1280;
    this.canvas.height = 720;
    const context = this.canvas.getContext('2d', { alpha: false });
    if (!context) {
      throw new Error('Failed to acquire 2D context for FaceSimulationPipeline');
    }
    this.ctx = context;

    this.videoEl = document.createElement('video');
    this.videoEl.autoplay = true;
    this.videoEl.muted = true;
    this.videoEl.playsInline = true;

    // Instantiate core AI face swap engine
    this.engine = new FaceSwapEngine();
  }

  /**
   * Initializes MediaPipe models on-demand for Tester
   */
  public async initializeLibraries(): Promise<void> {
    await this.engine.initializeModels();
  }

  /**
   * Device Gallery Face Upload:
   * Accepts JPG, JPEG, PNG, WEBP file.
   * Runs automated face detection with MediaPipe and validates that a face exists.
   */
  public async loadGalleryImage(file: File | Blob): Promise<GalleryFaceValidationResult> {
    const res = await this.engine.loadGalleryImage(file);
    if (res.success) {
      this.preset = 'custom-upload';
    }
    return res;
  }

  /**
   * Compatibility method for dataUrl
   */
  public setCustomAvatar(dataUrl: string): void {
    fetch(dataUrl)
      .then((res) => res.blob())
      .then((blob) => {
        this.loadGalleryImage(blob);
      })
      .catch((err) => console.warn('[FaceSimulationPipeline] Error converting dataUrl to blob:', err));
  }

  /**
   * Resets face swap back to clean default
   */
  public async resetFace(): Promise<void> {
    await this.engine.resetFace();
    this.preset = 'neural-clone';
  }

  public async setPreset(preset: FacePreset): Promise<void> {
    this.preset = preset;
    const presetMap: Record<string, { src: string; name: string }> = {
      'neural-clone': { src: '/synthetic_face_avatar.jpg', name: 'Marcus (Neural Clone)' },
      'mona-lisa': { src: '/mona_lisa.jpg', name: 'Emma (Studio Headshot)' },
      'cyber-agent': { src: '/cyber_agent.jpg', name: 'Alex (Clean Headshot)' },
      'astronaut': { src: '/astronaut.jpg', name: 'Sophia (Natural Portrait)' },
      'synthetic-executive': { src: '/synthetic_executive.jpg', name: 'David (Corporate Exec)' },
    };

    if (presetMap[preset]) {
      await this.engine.loadPresetSource(preset, presetMap[preset].src, presetMap[preset].name);
    }
  }

  public setBlendConfig(config: Partial<FaceBlendConfig>): void {
    this.engine.setBlendConfig(config);
  }

  public getBlendConfig(): FaceBlendConfig {
    return this.engine.getBlendConfig();
  }

  public getTelemetry(): FaceSwapTelemetry {
    return this.engine.getTelemetry();
  }

  public getEngine(): FaceSwapEngine {
    return this.engine;
  }

  public isActive(): boolean {
    return this.active;
  }

  /**
   * Starts the face manipulation pipeline by intercepting the webcam track,
   * routing it through the hidden canvas, and capturing the stream.
   */
  public start(originalTrack: MediaStreamTrack, preset: FacePreset = 'neural-clone'): MediaStreamTrack {
    this.originalTrack = originalTrack;
    this.preset = preset;
    this.active = true;

    // Trigger on-demand library load for Tester
    this.initializeLibraries();

    // Ensure WebGL GPU programs and buffers are healthy
    this.engine.ensureWebGL();

    // Attach original video track to internal video element if needed
    const currentStream = this.videoEl.srcObject as MediaStream | null;
    const currentTrack = currentStream?.getVideoTracks()[0];
    if (!currentTrack || currentTrack.id !== originalTrack.id) {
      this.videoEl.srcObject = new MediaStream([originalTrack]);
    }
    this.videoEl.play().catch((err) => console.warn('[FaceSimulationPipeline] Video play error:', err));

    // Match canvas dimensions to track settings
    const settings = originalTrack.getSettings();
    if (settings.width && settings.height) {
      this.canvas.width = settings.width;
      this.canvas.height = settings.height;
      this.engine.resize(settings.width, settings.height);
    }

    // Cancel existing loop before starting new one
    if (this.animFrameId !== null) {
      cancelAnimationFrame(this.animFrameId);
      this.animFrameId = null;
    }

    // Start 60 FPS requestAnimationFrame render loop
    this.renderLoop();

    // Capture transformed canvas stream at 30 fps for WebRTC
    this.outputStream = this.canvas.captureStream(30);
    const transformedTrack = this.outputStream.getVideoTracks()[0];
    return transformedTrack;
  }

  private renderLoop = () => {
    if (!this.active) return;

    const { width, height } = this.canvas;

    if (this.videoEl.readyState >= 2) {
      // 1. Process AI Face Swap with MediaPipe & WebGL Delaunay mesh
      this.engine.processVideoFrame(this.videoEl, this.ctx, width, height);

      // 2. Optional secondary HUD overlays for test modes
      if (this.preset === 'biometric-mask') {
        this.drawBiometricTelemetry(width, height);
      } else if (this.preset === 'cyber-filter') {
        this.drawCyberOverlay(width, height);
      }
    } else {
      this.ctx.fillStyle = '#16181f';
      this.ctx.fillRect(0, 0, width, height);
    }

    this.animFrameId = requestAnimationFrame(this.renderLoop);
  };

  private drawBiometricTelemetry(width: number, height: number) {
    this.ctx.save();
    this.ctx.fillStyle = 'rgba(52, 168, 83, 0.85)';
    this.ctx.font = '500 11px monospace';
    this.ctx.fillText('BIOMETRIC DELAUNAY MESH: 854 TRIANGLES', 24, height - 36);
    this.ctx.fillText('MEDIAN CONFIDENCE: 99.4%', 24, height - 20);
    this.ctx.restore();
  }

  private drawCyberOverlay(width: number, height: number) {
    this.ctx.save();
    this.ctx.strokeStyle = 'rgba(59, 130, 246, 0.45)';
    this.ctx.lineWidth = 1.5;
    this.ctx.strokeRect(width * 0.25, height * 0.2, width * 0.5, height * 0.6);
    this.ctx.restore();
  }

  /**
   * Pauses face swap rendering and halts canvas capture stream.
   * Keeps WebGL shaders, GPU buffers, and AI models in memory so re-enabling is instant.
   */
  public stop(): void {
    this.active = false;
    if (this.animFrameId !== null) {
      cancelAnimationFrame(this.animFrameId);
      this.animFrameId = null;
    }
    if (this.outputStream) {
      this.outputStream.getTracks().forEach((track) => {
        try {
          track.enabled = false;
          track.stop();
        } catch (e) {}
      });
      this.outputStream = null;
    }
    if (this.videoEl) {
      try {
        this.videoEl.pause();
      } catch (e) {}
    }
    this.engine.resetTracking();
  }

  /**
   * Permanent teardown when call ends or component unmounts.
   */
  public destroy(): void {
    this.stop();
    if (this.videoEl) {
      try {
        this.videoEl.pause();
        this.videoEl.srcObject = null;
      } catch (e) {}
    }
    this.originalTrack = null;
    this.engine.release();
  }
}
