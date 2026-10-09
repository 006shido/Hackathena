/** Local recorded-media fixture for the actual browser streaming adapter. */
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
const root=fileURLToPath(new URL('../../',import.meta.url));
const requireClient=createRequire(new URL('../../client/package.json',import.meta.url));
const {createServer}=await import(pathToFileURL(requireClient.resolve('vite')).href);
let failOffer=false;
const output=path.join(root,'ml/experiments/reference_backend/webrtc_transport');
await mkdir(output,{recursive:true});
const server=await createServer({root:path.join(root,'client'),configFile:false,
  define:{'import.meta.env.VITE_ENABLE_WEBRTC_VIDEO':'"true"'},
  server:{host:'127.0.0.1',port:6165,strictPort:true,proxy:{'/api':{target:process.env.TEST_APP_URL || 'http://127.0.0.1:5001',changeOrigin:true}}},
  plugins:[{name:'webrtc-recorded-fixture',configureServer(vite:any){vite.middlewares.use(async(req:any,res:any,next:any)=>{
    if(req.url==='/__webrtc/fail-offer' && req.method==='POST'){failOffer=true;res.end('ok');return;}
    if(failOffer && req.url?.endsWith('/offer')){res.statusCode=503;res.setHeader('Content-Type','application/json');res.end('{"error":"Fixture simulated streaming failure"}');return;}
    const media:Record<string,string>={
      '/__webrtc/video.webm':'ml/experiments/reference_backend/videos/faceocc2.webm',
      '/__webrtc/source.jpg':'ml/data/celeba/img_align_celeba/004831.jpg',
      '/__webrtc/source-b.jpg':'ml/data/celeba/img_align_celeba/004931.jpg',
    };
    if(media[req.url]){res.setHeader('Content-Type',req.url.endsWith('.webm')?'video/webm':'image/jpeg');res.end(await readFile(path.join(root,media[req.url])));return;}
    if(req.url==='/__webrtc/result' && req.method==='POST'){
      let value='';for await(const chunk of req){value+=chunk;if(value.length>100000){res.statusCode=413;res.end();return;}}
      const result=JSON.parse(value);await writeFile(path.join(output,result.fallback?'browser-fallback.json':'browser-webrtc.json'),JSON.stringify(result,null,2));res.end('saved');return;
    }
    next();
  });}}]});
await server.listen();console.log('Recorded-media test: http://localhost:6165/tests/webrtc-video.html');
setTimeout(()=>void server.close(),15*60*1000);
process.on('SIGINT',()=>void server.close());
