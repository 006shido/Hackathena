import { io } from 'socket.io-client';
import { NeuralVideoPreview } from '../src/attack/neuralVideoPreview';
import { VoiceTransformationPipeline } from '../src/attack/voiceTransformation';

const status=document.querySelector<HTMLPreElement>('#status')!;
const original=document.querySelector<HTMLVideoElement>('#original')!;
const remote=document.querySelector<HTMLVideoElement>('#remote')!;
const delay=(ms:number)=>new Promise(resolve=>setTimeout(resolve,ms));
function log(message:string){status.textContent+='\n'+message;}
function check(condition:unknown,message:string){if(!condition)throw new Error(message);log('PASS: '+message);}
function event(socket:ReturnType<typeof io>,name:string):Promise<any>{
  return new Promise((resolve,reject)=>{
    const timer=setTimeout(()=>{socket.off(name,listener);reject(new Error('Timed out: '+name));},10000);
    const listener=(value:any)=>{clearTimeout(timer);resolve(value);};socket.once(name,listener);
  });
}
function snapshot(video:HTMLVideoElement){
  const canvas=document.createElement('canvas');canvas.width=video.videoWidth;canvas.height=video.videoHeight;
  canvas.getContext('2d')!.drawImage(video,0,0);return canvas.toDataURL('image/png');
}

document.querySelector<HTMLButtonElement>('#start')!.onclick=async()=>{
  document.querySelector<HTMLButtonElement>('#start')!.disabled=true;
  status.textContent='Running';
  const audio=new AudioContext();await audio.resume();
  const speech=new URLSearchParams(location.search).get('audio')==='speech';
  const oscillator=audio.createOscillator();oscillator.frequency.value=440;
  const speechSource=speech ? audio.createBufferSource() : null;
  if(speechSource){speechSource.buffer=await audio.decodeAudioData(await (await fetch('/__acceptance/speech.wav')).arrayBuffer());speechSource.loop=true;}
  const gain=audio.createGain();gain.gain.value=.2;
  const destination=audio.createMediaStreamDestination();(speechSource || oscillator).connect(gain).connect(destination);(speechSource || oscillator).start();
  const rawAudio=destination.stream.getAudioTracks()[0];
  const preview=new NeuralVideoPreview();const voice=new VoiceTransformationPipeline();
  const sender=new RTCPeerConnection({iceServers:[]});const receiver=new RTCPeerConnection({iceServers:[]});
  let tester:ReturnType<typeof io>|undefined,user:ReturnType<typeof io>|undefined;
  let running=true;let rawVideo:MediaStreamTrack|undefined;let timer=0;
  const report:any={started:new Date().toISOString(),audioFixture:speech?'speech':'tone',scope:'Real browser, production preview/voice classes and production signaling; recorded video and selected audio fixture, not physical webcam/microphone or full Call UI.',checks:{},snapshots:{}};
  try{
    const fixture=await fetch('/__acceptance/fixture').then(r=>r.json());
    localStorage.setItem('deeptrace_auth_token',fixture.tester);
    original.src='/__acceptance/video.webm';await original.play();
    const canvas=document.createElement('canvas');canvas.width=320;canvas.height=240;
    const draw=()=>{if(!running)return;canvas.getContext('2d')!.drawImage(original,0,0,320,240);timer=window.setTimeout(draw,67);};draw();
    rawVideo=canvas.captureStream(15).getVideoTracks()[0];
    const source=await fetch('/__acceptance/source.jpg').then(r=>r.blob());
    const synthetic=await preview.start(rawVideo,source);
    const videoSender=sender.addTrack(synthetic,new MediaStream([synthetic]));
    const audioSender=sender.addTrack(rawAudio,new MediaStream([rawAudio]));
    const incoming=new MediaStream();remote.srcObject=incoming;
    receiver.ontrack=e=>{incoming.addTrack(e.track);void remote.play();};
    const pendingSender:RTCIceCandidateInit[]=[],pendingReceiver:RTCIceCandidateInit[]=[];
    tester=io({auth:{token:fixture.tester},autoConnect:false,reconnection:false});
    user=io({auth:{token:fixture.user},autoConnect:false,reconnection:false});
    const states:any[]=[];user.on('peer-attack-state',state=>states.push(state));
    sender.onicecandidate=e=>{if(e.candidate)tester!.emit('webrtc-ice',{roomId:'NEURAL',candidate:e.candidate.toJSON()});};
    receiver.onicecandidate=e=>{if(e.candidate)user!.emit('webrtc-ice',{roomId:'NEURAL',candidate:e.candidate.toJSON()});};
    tester.on('webrtc-ice',async({candidate})=>{if(sender.remoteDescription)await sender.addIceCandidate(candidate);else pendingSender.push(candidate);});
    user.on('webrtc-ice',async({candidate})=>{if(receiver.remoteDescription)await receiver.addIceCandidate(candidate);else pendingReceiver.push(candidate);});
    user.on('webrtc-offer',async({sdp})=>{
      await receiver.setRemoteDescription(sdp);for(const c of pendingReceiver.splice(0))await receiver.addIceCandidate(c);
      await receiver.setLocalDescription(await receiver.createAnswer());user!.emit('webrtc-answer',{roomId:'NEURAL',sdp:receiver.localDescription});
    });
    tester.on('webrtc-answer',async({sdp})=>{
      await sender.setRemoteDescription(sdp);for(const c of pendingSender.splice(0))await sender.addIceCandidate(c);
    });
    let connected=event(tester,'connect');tester.connect();await connected;
    let joined=event(tester,'room-joined');tester.emit('join-room',{roomId:'NEURAL'});await joined;
    const ack=event(tester,'attack-simulation-ack');
    tester.emit('attack-simulation-update',{roomId:'NEURAL',faceSwap:true,voiceTransform:false,attackMode:'face'});await ack;
    connected=event(user,'connect');user.connect();await connected;
    const late=event(user,'peer-attack-state');joined=event(user,'room-joined');user.emit('join-room',{roomId:'NEURAL'});await joined;
    check((await late).faceSwap,'late user receives active face status');report.checks.lateJoin=true;
    await sender.setLocalDescription(await sender.createOffer());tester.emit('webrtc-offer',{roomId:'NEURAL',sdp:sender.localDescription});
    for(let i=0;i<100 && (receiver.connectionState!=='connected'||remote.videoWidth===0);i++)await delay(100);
    check(receiver.connectionState==='connected' && remote.videoWidth===320,'remote WebRTC video connects at expected dimensions');
    await delay(4000);
    check(preview.ready && preview.fps>0,'neural frames reach the preview canvas');
    report.previewFps=preview.fps;report.snapshots['remote-a']=snapshot(remote);report.checks.video=true;
    const transformed=await voice.start(rawAudio,'robotic-vocoder');await audioSender.replaceTrack(transformed);
    const voiceStatus=event(user,'peer-attack-state');tester.emit('attack-simulation-update',{roomId:'NEURAL',faceSwap:true,voiceTransform:true,attackMode:'combined'});
    check((await voiceStatus).voiceTransform,'remote receives voice transformation status');
    const analyser=audio.createAnalyser();analyser.fftSize=2048;
    audio.createMediaStreamSource(new MediaStream(incoming.getAudioTracks())).connect(analyser);
    await delay(1500);const samples=new Float32Array(2048);analyser.getFloatTimeDomainData(samples);
    const rms=Math.sqrt(samples.reduce((sum,x)=>sum+x*x,0)/samples.length);
    check(rms>.001,'transformed audio is non-silent at the receiving peer');report.audioRms=rms;report.checks.audio=true;
    const screen=document.createElement('canvas');screen.width=320;screen.height=240;
    const screenContext=screen.getContext('2d')!;screenContext.fillStyle='rgb(17,120,210)';screenContext.fillRect(0,0,320,240);
    const screenTrack=screen.captureStream(5).getVideoTracks()[0];
    preview.stop();await videoSender.replaceTrack(screenTrack);
    const sharing=event(user,'peer-attack-state');tester.emit('attack-simulation-update',{roomId:'NEURAL',faceSwap:false,voiceTransform:true,attackMode:'voice'});await sharing;
    await delay(1000);const receivedScreen=document.createElement('canvas');receivedScreen.width=320;receivedScreen.height=240;
    const receivedContext=receivedScreen.getContext('2d')!;receivedContext.drawImage(remote,0,0);
    const color=receivedContext.getImageData(160,120,1,1).data;
    check(Math.abs(color[0]-17)<20 && Math.abs(color[1]-120)<20 && Math.abs(color[2]-210)<20,'replacement screen track reaches remote peer');
    check(rawVideo.readyState==='live','screen sharing preserves camera input');
    const resumed=await preview.start(rawVideo,source);await videoSender.replaceTrack(resumed);screenTrack.stop();
    const restored=event(user,'peer-attack-state');tester.emit('attack-simulation-update',{roomId:'NEURAL',faceSwap:true,voiceTransform:true,attackMode:'combined'});await restored;
    await delay(1000);check(preview.ready && receiver.connectionState==='connected','camera effect resumes after screen-track replacement');report.checks.screenReplacement=true;
    const second=await fetch('/__acceptance/source-b.jpg').then(r=>r.blob());await preview.setSource(second);await delay(3500);
    report.snapshots['remote-b']=snapshot(remote);
    check(preview.ready && receiver.connectionState==='connected','source switching keeps the media connection alive');report.checks.sourceSwitch=true;
    const stats:any[]=[];(await receiver.getStats()).forEach(entry=>{if(entry.type==='inbound-rtp')stats.push({kind:entry.kind,bytesReceived:entry.bytesReceived,framesDecoded:entry.framesDecoded});});
    check(stats.some(s=>s.kind==='video' && s.framesDecoded>10),'receiver decoded multiple transmitted frames');report.inboundStats=stats;
    preview.stop();voice.stop();await videoSender.replaceTrack(rawVideo);await audioSender.replaceTrack(rawAudio);
    const stopped=event(user,'peer-attack-state');tester.emit('attack-simulation-update',{roomId:'NEURAL',faceSwap:false,voiceTransform:false,attackMode:'none'});
    check((await stopped).attackMode==='none','receiver sees simulation stop');
    check(rawVideo.readyState==='live' && rawAudio.readyState==='live','stopping effects preserves original media tracks');report.checks.stop=true;
    await delay(1000);report.snapshots['remote-restored']=snapshot(remote);
    report.passed=true;log('COMPLETE: media acceptance checks passed');
  }catch(error){report.passed=false;report.error=String(error);log('FAILED: '+String(error));}
  finally{
    running=false;clearTimeout(timer);preview.stop();voice.stop();sender.close();receiver.close();tester?.disconnect();user?.disconnect();
    rawVideo?.stop();rawAudio.stop();(speechSource || oscillator).stop();await audio.close();original.pause();remote.srcObject=null;
    localStorage.removeItem('deeptrace_auth_token');
    report.finished=new Date().toISOString();
    await fetch('/__acceptance/result',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(report)});
    log('Results saved locally.');
  }
};
