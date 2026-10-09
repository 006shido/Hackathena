/** Local-only acceptance runner. Fresh secret and fixtures never reach normal auth. */
import { randomBytes } from 'node:crypto';
import { spawn } from 'node:child_process';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';
import { createRequire } from 'node:module';

const root=fileURLToPath(new URL('../../',import.meta.url));
process.env.JWT_SECRET=randomBytes(32).toString('hex');
const { generateToken }=await import('../src/auth.js');
const fixture={
  tester:generateToken({username:'acceptance-tester',name:'Acceptance Tester',role:'tester'}),
  user:generateToken({username:'acceptance-user',name:'Acceptance User',role:'user'}),
};
const output=process.env.TEST_OUTPUT_DIR ? path.resolve(process.env.TEST_OUTPUT_DIR) : path.join(root,process.argv.includes('--speech') ? 'ml/experiments/reference_backend/browser_speech_call' : process.argv.includes('--tracked') ? 'ml/experiments/reference_backend/tracked_browser_call' : 'ml/experiments/reference_backend/browser_call');
await mkdir(output,{recursive:true});
const files:Record<string,string>={
  '/__acceptance/video.webm':path.join(root,'ml/experiments/reference_backend/videos/faceocc2.webm'),
  '/__acceptance/source.jpg':path.join(root,'ml/data/celeba/img_align_celeba/004831.jpg'),
  '/__acceptance/source-b.jpg':path.join(root,'ml/data/celeba/img_align_celeba/004931.jpg'),
  '/__acceptance/speech.wav':path.join(root,'ml/experiments/reference_backend/videos/speech.wav'),
};
const requireClient=createRequire(new URL('../../client/package.json',import.meta.url));
const { createServer }=await import(pathToFileURL(requireClient.resolve('vite')).href);
const signaling=spawn(process.execPath,['dist/index.js'],{
  cwd:path.join(root,'server'),windowsHide:true,
  env:{...process.env,PORT:'6011',HOST:'127.0.0.1',ML_SERVICE_PORT:'8001'},stdio:['ignore','pipe','pipe'],
});
signaling.stdout.on('data',data=>process.stdout.write(data));
signaling.stderr.on('data',data=>process.stderr.write(data));
const fullUI=process.argv.includes('--full-ui');
const vite=await createServer({root:path.join(root,'client'),configFile:fullUI ? path.join(root,'client/vite.config.ts') : false,
  define:fullUI ? {'import.meta.env.VITE_ENABLE_NEURAL_VIDEO_PREVIEW':'"true"'} : {},
  server:{host:'127.0.0.1',port:6173,strictPort:true,proxy:{
    '/api':{target:'http://127.0.0.1:6011',changeOrigin:true},
    '/socket.io':{target:'http://127.0.0.1:6011',ws:true},
  }},plugins:[{name:'local-acceptance-fixtures',configureServer(server:any){
    server.middlewares.use(async(req:any,res:any,next:any)=>{
      try{
        if(req.url==='/__acceptance/fixture'){
          res.setHeader('content-type','application/json');res.setHeader('cache-control','no-store');
          res.end(JSON.stringify(fixture));return;
        }
        if(files[req.url]){
          res.setHeader('content-type',req.url.endsWith('.webm')?'video/webm':req.url.endsWith('.wav')?'audio/wav':'image/jpeg');
          res.end(await readFile(files[req.url]));return;
        }
        if(req.url==='/__acceptance/audio-result' && req.method==='POST'){
          const chunks:Buffer[]=[];let size=0;
          for await(const chunk of req){size+=chunk.length;if(size>8*1024*1024)throw new Error('Audio result too large');chunks.push(chunk);}
          const result=JSON.parse(Buffer.concat(chunks).toString());
          for(const name of ['clean','robotic-vocoder','deep-pitch-neural','synthetic-clone'])if(result.audio?.[name]){
            const data=String(result.audio[name]).replace(/^data:audio\/wav;base64,/,'');
            await writeFile(path.join(output,name+'.wav'),Buffer.from(data,'base64'));
          }
          delete result.audio;await writeFile(path.join(output,'audio-results.json'),JSON.stringify(result,null,2));
          res.setHeader('content-type','application/json');res.end('{"saved":true}');return;
        }
        if(req.url==='/__acceptance/result' && req.method==='POST'){
          const chunks:Buffer[]=[];let size=0;
          for await(const chunk of req){size+=chunk.length;if(size>8*1024*1024)throw new Error('Result too large');chunks.push(chunk);}
          const result=JSON.parse(Buffer.concat(chunks).toString());
          for(const name of ['remote-a','remote-b','remote-restored'])if(result.snapshots?.[name]){
            const image=String(result.snapshots[name]).replace(/^data:image\/png;base64,/,'');
            await writeFile(path.join(output,name+'.png'),Buffer.from(image,'base64'));
          }
          delete result.snapshots;
          await writeFile(path.join(output,'results.json'),JSON.stringify(result,null,2));
          res.setHeader('content-type','application/json');res.end('{"saved":true}');return;
        }
        next();
      }catch(error){res.statusCode=500;res.end(String(error));}
    });
  }}]});
await vite.listen();
const userVite=fullUI ? await createServer({root:path.join(root,'client'),configFile:path.join(root,'client/vite.config.ts'),
  define:{'import.meta.env.VITE_ENABLE_NEURAL_VIDEO_PREVIEW':'"true"'},
  server:{host:'127.0.0.1',port:6174,strictPort:true,proxy:{
    '/api':{target:'http://127.0.0.1:6011',changeOrigin:true},
    '/socket.io':{target:'http://127.0.0.1:6011',ws:true},
  }},plugins:[{name:'local-user-fixtures',configureServer(server:any){
    server.middlewares.use(async(req:any,res:any,next:any)=>{
      if(req.url==='/__acceptance/fixture'){
        res.setHeader('content-type','application/json');res.setHeader('cache-control','no-store');
        res.end(JSON.stringify(fixture));return;
      }
      if(files[req.url]){res.setHeader('content-type',req.url.endsWith('.webm')?'video/webm':req.url.endsWith('.wav')?'audio/wav':'image/jpeg');res.end(await readFile(files[req.url]));return;}
      next();
    });
  }}]}) : null;
await userVite?.listen();
console.log('Acceptance page: http://127.0.0.1:6173/tests/neural-call.html');
if(fullUI)console.log('Full UI: http://127.0.0.1:6173/tests/call-ui.html?role=tester&room=UI-ACCEPT / http://127.0.0.1:6174/tests/call-ui.html?role=user&room=UI-ACCEPT');
let shuttingDown=false;
async function stop(){if(shuttingDown)return;shuttingDown=true;await vite.close();await userVite?.close();signaling.kill();process.exit(0);}
process.on('SIGINT',()=>void stop());process.on('SIGTERM',()=>void stop());
setTimeout(()=>void stop(),20*60*1000);
