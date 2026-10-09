import { VoiceTransformationPipeline, VoicePreset } from '../src/attack/voiceTransformation';
const status = document.querySelector<HTMLPreElement>('#status')!;
const delay = (ms:number) => new Promise(resolve => setTimeout(resolve,ms));
function check(value:unknown,message:string){if(!value)throw new Error(message);status.textContent += `\nPASS: ${message}`;}
function wav(chunks:Float32Array[],sampleRate:number){
  const count=chunks.reduce((sum,chunk)=>sum+chunk.length,0);
  const buffer=new ArrayBuffer(44+count*2),view=new DataView(buffer);
  const word=(offset:number,value:string)=>Array.from(value).forEach((char,i)=>view.setUint8(offset+i,char.charCodeAt(0)));
  word(0,'RIFF');view.setUint32(4,36+count*2,true);word(8,'WAVE');word(12,'fmt ');
  view.setUint32(16,16,true);view.setUint16(20,1,true);view.setUint16(22,1,true);
  view.setUint32(24,sampleRate,true);view.setUint32(28,sampleRate*2,true);view.setUint16(32,2,true);view.setUint16(34,16,true);
  word(36,'data');view.setUint32(40,count*2,true);let offset=44;
  for(const chunk of chunks)for(const sample of chunk){view.setInt16(offset,Math.round(Math.max(-1,Math.min(1,sample))*32767),true);offset+=2;}
  return new Blob([buffer],{type:'audio/wav'});
}
async function capture(ctx:AudioContext,track:MediaStreamTrack){
  const source=ctx.createMediaStreamSource(new MediaStream([track]));
  const processor=ctx.createScriptProcessor(4096,1,1),silent=ctx.createGain();silent.gain.value=0;
  const chunks:Float32Array[]=[];processor.onaudioprocess=event=>chunks.push(event.inputBuffer.getChannelData(0).slice());
  source.connect(processor).connect(silent).connect(ctx.destination);await delay(4500);
  source.disconnect();processor.disconnect();silent.disconnect();processor.onaudioprocess=null;
  let sum=0,count=0,clipped=0,peak=0;
  for(const chunk of chunks)for(const sample of chunk){sum+=sample*sample;count++;peak=Math.max(peak,Math.abs(sample));if(Math.abs(sample)>=.99)clipped++;}
  const blob=wav(chunks,ctx.sampleRate);
  const data=await new Promise<string>(resolve=>{const reader=new FileReader();reader.onload=()=>resolve(String(reader.result));reader.readAsDataURL(blob);});
  return {rms:Math.sqrt(sum/Math.max(1,count)),peak,clippedRatio:clipped/Math.max(1,count),samples:count,data};
}
document.querySelector<HTMLButtonElement>('#start')!.onclick=async()=>{
  document.querySelector<HTMLButtonElement>('#start')!.disabled=true;status.textContent='Running';
  const ctx=new AudioContext();await ctx.resume();const voice=new VoiceTransformationPipeline();
  const sender=new RTCPeerConnection({iceServers:[]}),receiver=new RTCPeerConnection({iceServers:[]});
  const remoteAudio=document.createElement('audio');remoteAudio.muted=true;remoteAudio.autoplay=true;document.body.append(remoteAudio);
  const report:any={scope:'Recorded speech, production voice pipeline and actual loopback WebRTC. Not full UI, microphone, cross-computer connectivity or speech-recognition accuracy.',measurements:{},audio:{},checks:{}};
  let speech:AudioBufferSourceNode|undefined,raw:MediaStreamTrack|undefined;
  try{
    speech=ctx.createBufferSource();speech.buffer=await ctx.decodeAudioData(await (await fetch('/__acceptance/speech.wav')).arrayBuffer());speech.loop=true;
    const destination=ctx.createMediaStreamDestination();speech.connect(destination);speech.start();raw=destination.stream.getAudioTracks()[0];
    let incoming:MediaStreamTrack|undefined;receiver.ontrack=event=>{
      incoming=event.track;remoteAudio.srcObject=new MediaStream([event.track]);void remoteAudio.play();
    };
    sender.onicecandidate=event=>{if(event.candidate)void receiver.addIceCandidate(event.candidate);};
    receiver.onicecandidate=event=>{if(event.candidate)void sender.addIceCandidate(event.candidate);};
    const audioSender=sender.addTrack(raw,new MediaStream([raw]));
    await sender.setLocalDescription(await sender.createOffer());await receiver.setRemoteDescription(sender.localDescription!);
    await receiver.setLocalDescription(await receiver.createAnswer());await sender.setRemoteDescription(receiver.localDescription!);
    for(let i=0;i<100 && (receiver.connectionState!=='connected'||!incoming);i++)await delay(100);
    check(receiver.connectionState==='connected' && incoming,'speech WebRTC connects');
    report.input=await capture(ctx,raw);delete report.input.data;
    check(report.input.rms>.001,'recorded speech input is non-silent');
    let previousOutput:MediaStreamTrack|undefined;
    for(const preset of ['clean','robotic-vocoder','deep-pitch-neural','synthetic-clone'] as const){
      if(preset!=='clean'){
        const transformed=await voice.start(raw,preset as VoicePreset);await audioSender.replaceTrack(transformed);
        if(previousOutput)check(previousOutput.readyState==='ended','restarting voice stops the previous derived track');
        previousOutput=transformed;
      }
      await delay(500);const measurement=await capture(ctx,incoming!);
      report.audio[preset]=measurement.data;delete (measurement as Partial<typeof measurement>).data;
      report.measurements[preset]=measurement;
      check(measurement.rms>.001,`${preset}: received speech is non-silent`);
      check(measurement.clippedRatio<.01,`${preset}: less than 1% clipped samples`);
    }
    voice.stop();await audioSender.replaceTrack(raw);check(raw.readyState==='live','voice effects preserve original speech track');
    report.checks.transport=true;report.checks.presets=true;report.passed=true;
  }catch(error){report.passed=false;report.error=String(error);status.textContent+='\nFAILED: '+String(error);}
  finally{
    voice.stop();sender.close();receiver.close();speech?.stop();raw?.stop();await ctx.close();remoteAudio.pause();remoteAudio.srcObject=null;remoteAudio.remove();
    await fetch('/__acceptance/audio-result',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(report)});
    status.textContent+='\nResults saved locally.';
  }
};
