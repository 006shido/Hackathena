import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {createRequire} from 'node:module';
import {readFile,writeFile} from 'node:fs/promises';
const root=fileURLToPath(new URL('../../',import.meta.url));
const requireClient=createRequire(new URL('../../client/package.json',import.meta.url));
const {createServer}=await import(pathToFileURL(requireClient.resolve('vite')).href);
const server=await createServer({root:path.join(root,'client'),configFile:false,server:{host:'127.0.0.1',port:6166,strictPort:true},plugins:[{name:'voice-score-fixture',configureServer(vite:any){vite.middlewares.use(async(req:any,res:any,next:any)=>{
 const match=/^\/fixtures\/(clean|robotic-vocoder|deep-pitch-neural|synthetic-clone)\.wav$/.exec(req.url??'');
 if(match){res.setHeader('Content-Type','audio/wav');res.end(await readFile(path.join(root,'ml/experiments/reference_backend/browser_speech_call',match[1]+'.wav')));return;}
 if(req.url==='/result'&&req.method==='POST'){let body='';for await(const chunk of req){body+=chunk;if(body.length>2000000){res.statusCode=413;res.end();return;}}JSON.parse(body);await writeFile(path.join(root,'ml/experiments/reference_backend/browser_speech_call/score-current.json'),body);res.end('saved');return;}
 next();
 });}}]});
await server.listen();console.log('http://localhost:6166/tests/voice-score.html');
process.on('SIGINT',()=>void server.close());
