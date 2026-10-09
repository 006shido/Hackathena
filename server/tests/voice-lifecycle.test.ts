import assert from 'node:assert/strict';
const resumeResolvers:Array<()=>void>=[];
const sources:any[]=[];
let delayResume=false,rawStops=0;
const parameter=()=>({value:0,setValueAtTime(){}});
class Node {
  connections:any[]=[];gain=parameter();frequency=parameter();Q=parameter();
  connect(other:any){this.connections.push(other);return other;}
  disconnect(){this.connections=[];}
  start(){} stop(){}
}
class Context {
  state='suspended';currentTime=0;onstatechange:any=null;destination=new Node();
  async resume(){if(delayResume)await new Promise<void>(resolve=>resumeResolvers.push(resolve));if(this.state!=='closed')this.state='running';}
  async close(){this.state='closed';}
  createMediaStreamSource(){const source=new Node();sources.push(source);return source;}
  createMediaStreamDestination(){const node:any=new Node();const track={readyState:'live',enabled:true,stop(){this.readyState='ended';}};node.stream={getTracks:()=>[track],getAudioTracks:()=>[track]};return node;}
  createGain(){return new Node();} createOscillator(){return new Node();}
  createBiquadFilter(){return new Node();} createWaveShaper(){return new Node();}
}
(globalThis as any).window={AudioContext:Context};
(globalThis as any).MediaStream=class {constructor(public tracks:any[]){} getAudioTracks(){return this.tracks;}};
const raw={stop(){rawStops++;}};
const {VoiceTransformationPipeline}=await import('../../client/src/attack/voiceTransformation.ts');
const pipeline=new VoiceTransformationPipeline();
const first=await pipeline.start(raw as any,'robotic-vocoder');
const input=sources.at(-1);assert.equal(input.connections.length,2);
pipeline.setPreset('synthetic-clone');assert.equal(input.connections.length,2,'Preset changes accumulated anonymous input branches');
pipeline.setPreset('deep-pitch-neural');assert.equal(input.connections.length,1);
const second=await pipeline.start(raw as any);assert.equal(first.readyState,'ended');
pipeline.stop();assert.equal(second.readyState,'ended');assert.equal(rawStops,0);
delayResume=true;
const cancelled=pipeline.start(raw as any).then(()=>null,error=>error);
const newest=pipeline.start(raw as any);
resumeResolvers[0]();assert((await cancelled) instanceof Error);
resumeResolvers[1]();const current=await newest;assert.equal(current.readyState,'live');
pipeline.stop();assert.equal(current.readyState,'ended');assert.equal(rawStops,0);
console.log('PASS: preset branch cleanup, derived-track replacement, cancelled startup and original audio preservation.');
