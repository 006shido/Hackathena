export type FacePreset = 'neural-clone' | 'biometric-mask' | 'synthetic-executive';

export class FaceSimulationPipeline {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private videoEl: HTMLVideoElement;
  private active = false;
  private animFrameId: number | null = null;
  private originalTrack: MediaStreamTrack | null = null;
  private outputStream: MediaStream | null = null;
  private preset: FacePreset = 'neural-clone';
  
  private avatarImages: Map<string, HTMLImageElement> = new Map();
  private isImageLoaded = false;
  private frameCount = 0;

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

    this.preloadAvatars();
  }

  private preloadAvatars() {
    const avatar1 = new Image();
    avatar1.src = '/synthetic_face_avatar.jpg';
    avatar1.onload = () => {
      this.avatarImages.set('neural-clone', avatar1);
      this.isImageLoaded = true;
    };

    const avatar2 = new Image();
    avatar2.src = '/synthetic_executive.jpg';
    avatar2.onload = () => {
      this.avatarImages.set('synthetic-executive', avatar2);
    };
  }

  public start(originalTrack: MediaStreamTrack, preset: FacePreset = 'neural-clone'): MediaStreamTrack {
    this.originalTrack = originalTrack;
    this.preset = preset;
    this.active = true;

    // Attach original video track to internal video element
    const stream = new MediaStream([originalTrack]);
    this.videoEl.srcObject = stream;
    this.videoEl.play().catch((err) => console.warn('Video element play error:', err));

    // Match canvas dimensions to track settings if available
    const settings = originalTrack.getSettings();
    if (settings.width && settings.height) {
      this.canvas.width = settings.width;
      this.canvas.height = settings.height;
    }

    this.renderLoop();

    // Capture canvas stream at 30 fps
    this.outputStream = this.canvas.captureStream(30);
    const transformedTrack = this.outputStream.getVideoTracks()[0];
    return transformedTrack;
  }

  public setPreset(preset: FacePreset) {
    this.preset = preset;
  }

  public isActive(): boolean {
    return this.active;
  }

  private renderLoop = () => {
    if (!this.active) return;

    this.frameCount++;
    const { width, height } = this.canvas;

    // 1. Draw camera video background
    if (this.videoEl.readyState >= 2) {
      this.ctx.drawImage(this.videoEl, 0, 0, width, height);
    } else {
      this.ctx.fillStyle = '#0f172a';
      this.ctx.fillRect(0, 0, width, height);
    }

    // 2. Render Face Swap / Deepfake Synthesis Overlay
    this.renderSyntheticFace(width, height);

    this.animFrameId = requestAnimationFrame(this.renderLoop);
  };

  private renderSyntheticFace(width: number, height: number) {
    // Face region estimation (portrait center headshot)
    const faceW = width * 0.38;
    const faceH = faceW * 1.25;
    
    // Subtle realistic breathing / head micro-sway motion simulation
    const swayX = Math.sin(this.frameCount * 0.04) * 4;
    const swayY = Math.cos(this.frameCount * 0.03) * 3;

    const centerX = width * 0.5 + swayX;
    const centerY = height * 0.44 + swayY;
    const faceX = centerX - faceW / 2;
    const faceY = centerY - faceH / 2;

    const currentImg = this.preset === 'synthetic-executive'
      ? this.avatarImages.get('synthetic-executive')
      : this.avatarImages.get('neural-clone');

    this.ctx.save();

    if (currentImg && currentImg.complete && currentImg.naturalWidth > 0) {
      // Create feathered oval clipping path for seamless face blending
      this.ctx.save();
      this.ctx.beginPath();
      this.ctx.ellipse(centerX, centerY, faceW * 0.46, faceH * 0.54, 0, 0, Math.PI * 2);
      this.ctx.clip();

      // Render synthetic persona face onto the face region
      this.ctx.drawImage(currentImg, faceX, faceY, faceW, faceH);

      // Subtle neural synthetic noise / artifact simulation
      if (this.frameCount % 12 === 0) {
        this.ctx.fillStyle = 'rgba(6, 182, 212, 0.06)';
        this.ctx.fillRect(faceX, faceY, faceW, faceH);
      }

      this.ctx.restore();

      // Soft edge feathering mask ring
      const featherGrad = this.ctx.createRadialGradient(
        centerX, centerY, faceW * 0.38,
        centerX, centerY, faceW * 0.52
      );
      featherGrad.addColorStop(0, 'rgba(0,0,0,0)');
      featherGrad.addColorStop(1, 'rgba(15, 23, 42, 0.2)');
      this.ctx.fillStyle = featherGrad;
      this.ctx.beginPath();
      this.ctx.ellipse(centerX, centerY, faceW * 0.52, faceH * 0.58, 0, 0, Math.PI * 2);
      this.ctx.fill();

    } else {
      // Procedural Cyber Face Morph fallback
      this.ctx.save();
      this.ctx.beginPath();
      this.ctx.ellipse(centerX, centerY, faceW * 0.44, faceH * 0.52, 0, 0, Math.PI * 2);
      this.ctx.fillStyle = 'rgba(30, 41, 59, 0.85)';
      this.ctx.fill();

      // Simulated facial wireframe mesh
      this.ctx.strokeStyle = 'rgba(6, 182, 212, 0.6)';
      this.ctx.lineWidth = 1.5;
      this.ctx.stroke();
      this.ctx.restore();
    }

    // Biometric / Landmark overlay if preset is biometric-mask or subtle tester HUD
    if (this.preset === 'biometric-mask') {
      this.drawBiometricMesh(centerX, centerY, faceW, faceH);
    }

    // Watermark / Testing indicator HUD badge (Subtle & high-tech)
    this.drawSecurityHud(centerX, centerY, faceW, faceH);

    this.ctx.restore();
  }

  private drawBiometricMesh(cx: number, cy: number, fw: number, fh: number) {
    this.ctx.save();
    this.ctx.strokeStyle = 'rgba(16, 185, 129, 0.45)';
    this.ctx.lineWidth = 1;
    this.ctx.setLineDash([2, 3]);

    // Facial landmark points
    const landmarks = [
      { x: cx - fw * 0.2, y: cy - fh * 0.12 }, // left eye
      { x: cx + fw * 0.2, y: cy - fh * 0.12 }, // right eye
      { x: cx, y: cy + fh * 0.05 },            // nose tip
      { x: cx - fw * 0.15, y: cy + fh * 0.24 },// left mouth
      { x: cx + fw * 0.15, y: cy + fh * 0.24 },// right mouth
      { x: cx, y: cy + fh * 0.26 },            // chin
    ];

    landmarks.forEach((pt) => {
      this.ctx.beginPath();
      this.ctx.arc(pt.x, pt.y, 3, 0, Math.PI * 2);
      this.ctx.fillStyle = '#10b981';
      this.ctx.fill();
    });

    // Mesh connecting lines
    this.ctx.beginPath();
    this.ctx.moveTo(landmarks[0].x, landmarks[0].y);
    this.ctx.lineTo(landmarks[2].x, landmarks[2].y);
    this.ctx.lineTo(landmarks[1].x, landmarks[1].y);
    this.ctx.lineTo(landmarks[4].x, landmarks[4].y);
    this.ctx.lineTo(landmarks[5].x, landmarks[5].y);
    this.ctx.lineTo(landmarks[3].x, landmarks[3].y);
    this.ctx.closePath();
    this.ctx.stroke();

    this.ctx.restore();
  }

  private drawSecurityHud(cx: number, cy: number, fw: number, fh: number) {
    this.ctx.save();
    
    // Top badge: Face Swap Active
    const badgeW = 240;
    const badgeH = 26;
    const badgeX = cx - badgeW / 2;
    const badgeY = cy - fh * 0.58 - badgeH;

    this.ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
    this.ctx.strokeStyle = '#ef4444';
    this.ctx.lineWidth = 1.5;
    this.ctx.beginPath();
    this.ctx.roundRect(badgeX, badgeY, badgeW, badgeH, 4);
    this.ctx.fill();
    this.ctx.stroke();

    // Pulse red dot
    const pulseAlpha = 0.6 + Math.sin(this.frameCount * 0.1) * 0.4;
    this.ctx.fillStyle = `rgba(239, 68, 68, ${pulseAlpha})`;
    this.ctx.beginPath();
    this.ctx.arc(badgeX + 14, badgeY + 13, 5, 0, Math.PI * 2);
    this.ctx.fill();

    this.ctx.font = '600 11px "JetBrains Mono", monospace';
    this.ctx.fillStyle = '#f87171';
    this.ctx.textAlign = 'left';
    this.ctx.fillText('FACE SIMULATION: ACTIVE', badgeX + 26, badgeY + 17);

    // Subtle corner brackets around face region
    const bSize = 16;
    const left = cx - fw * 0.48;
    const right = cx + fw * 0.48;
    const top = cy - fh * 0.52;
    const bottom = cy + fh * 0.56;

    this.ctx.strokeStyle = 'rgba(239, 68, 68, 0.6)';
    this.ctx.lineWidth = 2;
    this.ctx.setLineDash([]);

    // Top-left
    this.ctx.beginPath();
    this.ctx.moveTo(left, top + bSize);
    this.ctx.lineTo(left, top);
    this.ctx.lineTo(left + bSize, top);
    this.ctx.stroke();

    // Top-right
    this.ctx.beginPath();
    this.ctx.moveTo(right - bSize, top);
    this.ctx.lineTo(right, top);
    this.ctx.lineTo(right, top + bSize);
    this.ctx.stroke();

    // Bottom-left
    this.ctx.beginPath();
    this.ctx.moveTo(left, bottom - bSize);
    this.ctx.lineTo(left, bottom);
    this.ctx.lineTo(left + bSize, bottom);
    this.ctx.stroke();

    // Bottom-right
    this.ctx.beginPath();
    this.ctx.moveTo(right - bSize, bottom);
    this.ctx.lineTo(right, bottom);
    this.ctx.lineTo(right, bottom - bSize);
    this.ctx.stroke();

    this.ctx.restore();
  }

  public stop(): void {
    this.active = false;
    if (this.animFrameId !== null) {
      cancelAnimationFrame(this.animFrameId);
      this.animFrameId = null;
    }
    if (this.outputStream) {
      this.outputStream.getTracks().forEach((track) => track.stop());
      this.outputStream = null;
    }
    if (this.videoEl.srcObject) {
      this.videoEl.srcObject = null;
    }
    this.originalTrack = null;
  }
}
