import { useEffect,useState } from 'react';

export interface MediaMeasurement {
  status: string;
  score?: number | null;
  model?: string;
  reason?: string;
  receivedAt?: number;
}
export function useMediaDetection(stream: MediaStream | null, token: string, enabled: boolean) {
  const [measurements,setMeasurements]=useState<{video:MediaMeasurement|null;audio:MediaMeasurement|null}>({video:null,audio:null});
  const [previousInput, setPreviousInput] = useState({ stream, token, enabled });
  if (previousInput.stream !== stream || previousInput.token !== token || previousInput.enabled !== enabled) {
    setPreviousInput({ stream, token, enabled });
    setMeasurements({ video: null, audio: null });
  }
  useEffect(() => {
    if (!enabled || !stream || import.meta.env.VITE_ENABLE_MEDIA_DETECTION !== 'true') return;
    let disposed=false,videoBusy=false,audioBusy=false,audioStarting=false;
    let context:AudioContext|null=null,source:MediaStreamAudioSourceNode|null=null,worklet:AudioWorkletNode|null=null,gain:GainNode|null=null;
    const controllers=new Set<AbortController>();
    const video=document.createElement('video');video.muted=true;video.playsInline=true;video.srcObject=stream;void video.play().catch(()=>{});
    const canvas=document.createElement('canvas');const paint=canvas.getContext('2d');
    const publish=(kind:'video'|'audio',value:MediaMeasurement) => {
      if(!disposed)setMeasurements(old=>({...old,[kind]:{...value,receivedAt:Date.now()}}));
    };
    const send=async(kind:'video'|'audio',body:BodyInit,extra:Record<string,string>={}) => {
      const controller=new AbortController();controllers.add(controller);
      const timeout=setTimeout(()=>controller.abort(),10000);
      try {
        const response=await fetch(`/api/ml/detection/${kind}`,{method:'POST',body,headers:{Authorization:`Bearer ${token}`,...extra},signal:controller.signal});
        if(!response.ok) {publish(kind,{status:response.status===429?'busy':'unavailable',reason:response.status===429?'Waiting for GPU':'Independent detector unavailable'});return;}
        const result=await response.json();
        if(result.score!==null && result.score!==undefined && (!Number.isFinite(result.score)||result.score<0||result.score>1))throw new Error('Invalid detector score');
        publish(kind,result);
      } catch { if(!disposed)publish(kind,{status:'unavailable',reason:'No fresh detector result'}); }
      finally {clearTimeout(timeout);controllers.delete(controller);}
    };
    const initializeAudio=async() => {
      if(audioStarting||context||!stream.getAudioTracks().length)return;
      audioStarting=true;
      try {
        context=new AudioContext({sampleRate:16000});await context.audioWorklet.addModule('/receiver-pcm-worklet.js');
        if(disposed)return;
        source=context.createMediaStreamSource(stream);worklet=new AudioWorkletNode(context,'receiver-pcm',{processorOptions:{segmentFrames:Math.floor(context.sampleRate*4.1)}});
        const rate=context.sampleRate;
        worklet.port.onmessage=async event=> {
          if(disposed||audioBusy)return;audioBusy=true;
          try {await send('audio',event.data,{'Content-Type':'application/octet-stream','X-Sample-Rate':String(rate)});}finally{audioBusy=false;}
        };
        gain=context.createGain();gain.gain.value=0;source.connect(worklet);worklet.connect(gain);gain.connect(context.destination);
        await context.resume();
      } catch {
        source?.disconnect();worklet?.disconnect();gain?.disconnect();
        if(context && context.state!=='closed')void context.close().catch(()=>{});
        context=null;source=null;worklet=null;gain=null;
        publish('audio',{status:'unavailable',reason:'Incoming audio analysis could not start'});
      }
      finally{audioStarting=false;}
    };
    const tick=async() => {
      void initializeAudio();
      if(context?.state==='suspended')void context.resume().catch(()=>{});
      if(disposed||videoBusy||!paint||video.readyState<2||!video.videoWidth)return;
      videoBusy=true;
      try {
        canvas.width=Math.min(640,video.videoWidth);canvas.height=Math.round(video.videoHeight*canvas.width/video.videoWidth);
        paint.drawImage(video,0,0,canvas.width,canvas.height);
        const blob=await new Promise<Blob|null>(resolve=>canvas.toBlob(resolve,'image/jpeg',.85));
        if(!blob||disposed)return;
        const body=new FormData();body.append('image',blob,'received.jpg');await send('video',body);
      } catch {publish('video',{status:'unavailable',reason:'Received frame could not be sampled'});}
      finally{videoBusy=false;}
    };
    const timer=setInterval(()=>void tick(),1500);void tick();
    return () => {
      disposed=true;clearInterval(timer);for(const controller of controllers)controller.abort();
      video.pause();video.srcObject=null;
      source?.disconnect();worklet?.disconnect();gain?.disconnect();if(worklet)worklet.port.onmessage=null;
      if(context && context.state!=='closed')void context.close().catch(()=>{});
    };
  },[stream,token,enabled]);
  return measurements;
}
