import path from 'node:path';import {fileURLToPath,pathToFileURL} from 'node:url';import {createRequire} from 'node:module';import {readFile,writeFile,mkdir} from 'node:fs/promises';
const root=fileURLToPath(new URL('../../',import.meta.url));const requireClient=createRequire(new URL('../../client/package.json',import.meta.url));const {createServer}=await import(pathToFileURL(requireClient.resolve('vite')).href);
const directory=path.join(root,'ml/detection/call_domain');await mkdir(path.join(directory,'browser_received'),{recursive:true});
const manifest=JSON.parse(await readFile(path.join(directory,'natural/manifest.json'),'utf8'));
const speakers=[...new Set<string>(manifest.records.map((row:any)=>row.speaker))].sort();
const clips=speakers.map((speaker,index)=>({...manifest.records.find((row:any)=>row.speaker===speaker),withEffects:index%2===0,split:index%5===4?'test':index%5===3?'calibration':'train'}));
const records:any[]=[];
const server=await createServer({root:path.join(root,'client'),configFile:false,server:{host:'127.0.0.1',port:6166,strictPort:true},plugins:[{name:'call-audio-dataset',configureServer(vite:any){vite.middlewares.use(async(req:any,res:any,next:any)=>{
 if(req.url==='/dataset/manifest'){res.setHeader('Content-Type','application/json');res.end(JSON.stringify(clips));return;}
 const match=/^\/dataset\/natural\/([0-9-]+\.flac)$/.exec(req.url??'');if(match&&clips.some(clip=>clip.file===match[1])){res.setHeader('Content-Type','audio/flac');res.end(await readFile(path.join(directory,'natural',match[1])));return;}
 if(req.url==='/dataset/record'&&req.method==='POST'){
  const clip=clips.find(clip=>clip.file===req.headers['x-clip']);const preset=req.headers['x-preset'];if(!clip||!['clean','robotic-vocoder','deep-pitch-neural','synthetic-clone'].includes(preset)){res.statusCode=400;res.end();return;}
  const chunks=[];let length=0;for await(const chunk of req){length+=chunk.length;if(length>258400){res.statusCode=413;res.end();return;}chunks.push(chunk);}if(length!==258400){res.statusCode=422;res.end();return;}
  const body=Buffer.concat(chunks);const samples=new Float32Array(body.buffer.slice(body.byteOffset,body.byteOffset+body.byteLength));const rms=Math.sqrt(samples.reduce((sum,value)=>sum+value*value,0)/samples.length);
  if(!Number.isFinite(rms)||rms<.0001){res.statusCode=422;res.end('No received speech');return;}
  const file=clip.file.replace('.flac','')+'-'+preset+'.f32';await writeFile(path.join(directory,'browser_received',file),body);records.push({file,speakers:[clip.speaker],source_file:clip.file,method:'actual-browser-webrtc',variant:preset,split:clip.split,label:preset==='clean'?0:1,sample_rate:16000,rms});
  await writeFile(path.join(directory,'browser_received/manifest.json'),JSON.stringify({scope:'Recorded public natural speech through production VoiceTransformationPipeline, actual loopback WebRTC and receiver AudioWorklet at 16 kHz. Not microphone or two-device validation.',records},null,2));res.end('saved');return;
 }
 next();});}}]});await server.listen();console.log('http://localhost:6166/tests/call-audio-dataset.html');process.on('SIGINT',()=>void server.close());
