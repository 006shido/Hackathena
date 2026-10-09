/** Local fixture entry only. Production App, hooks and signaling remain unchanged. */
import React from 'react';
import { createRoot } from 'react-dom/client';
import { App } from '../src/App';
import { mediaService } from '../src/services/media';
import '../src/index.css';

const role = new URLSearchParams(location.search).get('role') === 'user' ? 'user' : 'tester';
const fixture = await (await fetch('/__acceptance/fixture')).json();
localStorage.setItem('deeptrace_auth_token', fixture[role]);
localStorage.setItem('deeptrace_auth_user', JSON.stringify({
  username: `acceptance-${role}`, name: `Acceptance ${role}`, role,
}));

const resources = new Set<() => void>();
let sharedTrack: MediaStreamTrack | null = null;
let audioContexts: AudioContext[] = [];
mediaService.getUserMedia = async (video = true, audio = true) => {
  const clip = document.createElement('video');
  clip.src = '/__acceptance/video.webm'; clip.loop = true; clip.muted = true; clip.playsInline = true;
  await clip.play();
  const canvas = document.createElement('canvas'); canvas.width = 320; canvas.height = 240;
  const context = canvas.getContext('2d')!;
  const draw = () => context.drawImage(clip, 0, 0, 320, 240);
  draw(); const timer = window.setInterval(draw, 67);
  const camera = canvas.captureStream(15);
  const ctx = new AudioContext(); audioContexts.push(ctx);
  const oscillator = ctx.createOscillator(); oscillator.frequency.value = 440;
  const gain = ctx.createGain(); gain.gain.value = 0.1;
  const destination = ctx.createMediaStreamDestination();
  oscillator.connect(gain).connect(destination); oscillator.start();
  const stream = new MediaStream([
    ...(video ? camera.getVideoTracks() : []), ...(audio ? destination.stream.getAudioTracks() : []),
  ]);
  const release = () => {
    clearInterval(timer); clip.pause(); clip.removeAttribute('src'); clip.load();
    camera.getTracks().forEach(track => track.stop());
    destination.stream.getTracks().forEach(track => track.stop());
    oscillator.stop(); void ctx.close(); resources.delete(release);
  };
  resources.add(release);
  // Track.stop() does not emit ended; observe teardown without changing its semantics.
  const watcher = window.setInterval(() => {
    if (stream.getTracks().every(track => track.readyState === 'ended')) {
      clearInterval(watcher); release();
    }
  }, 250);
  resources.add(() => clearInterval(watcher));
  return { stream, hasVideo: video, hasAudio: audio };
};
mediaService.getScreenMedia = async () => {
  const canvas = document.createElement('canvas'); canvas.width = 320; canvas.height = 240;
  const context = canvas.getContext('2d')!;
  const draw = () => { context.fillStyle = 'rgb(17,120,210)'; context.fillRect(0, 0, 320, 240); };
  draw(); const timer = setInterval(draw, 67);
  const stream = canvas.captureStream(15); sharedTrack = stream.getVideoTracks()[0];
  resources.add(() => { clearInterval(timer); stream.getTracks().forEach(track => track.stop()); });
  return stream;
};
document.addEventListener('pointerdown', () => {
  audioContexts = audioContexts.filter(ctx => ctx.state !== 'closed');
  audioContexts.forEach(ctx => void ctx.resume());
});
window.addEventListener('beforeunload', () => {
  resources.forEach(release => release());
  localStorage.removeItem('deeptrace_auth_token'); localStorage.removeItem('deeptrace_auth_user');
});
createRoot(document.getElementById('root')!).render(<App />);
const controls = document.createElement('aside');
controls.style.cssText = 'position:fixed;top:0;left:0;z-index:10000;background:#fff;color:#111;padding:4px;font:12px sans-serif';
controls.textContent = `Local ${role} fixture: recorded video / generated audio. `;
const endShare = document.createElement('button'); endShare.textContent = 'End simulated share';
endShare.onclick = () => { sharedTrack?.stop(); sharedTrack?.dispatchEvent(new Event('ended')); };
controls.append(endShare); document.body.append(controls);
