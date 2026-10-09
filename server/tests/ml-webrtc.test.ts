import assert from 'node:assert/strict';
import http from 'node:http';
import { once } from 'node:events';
import express from 'express';
import {generateToken} from '../src/auth.js';
const seen:any[]=[];
const upstream=http.createServer(async(req,res)=>{
  let body='';for await(const chunk of req)body+=chunk;
  seen.push({owner:req.headers['x-ml-owner'],body:JSON.parse(body),path:req.url});
  res.setHeader('Content-Type','application/json');res.end('{"type":"answer","sdp":"test-answer"}');
});
upstream.listen(0,'127.0.0.1');await once(upstream,'listening');
process.env.ML_SERVICE_PORT=String((upstream.address() as import('node:net').AddressInfo).port);
const {mlRouter}=await import('../src/ml.js');
const app=express();app.use(express.json());app.use('/api/ml',mlRouter);
const server=app.listen(0,'127.0.0.1');await once(server,'listening');
const base=`http://127.0.0.1:${(server.address() as import('node:net').AddressInfo).port}/api/ml/video/sessions`;
const session='a'.repeat(32),body={type:'offer',sdp:'v=0\r\nfixture'};
const send=(role:'user'|'tester'|null,id=session)=>fetch(`${base}/${id}/offer`,{method:'POST',headers:{'Content-Type':'application/json',...(role?{Authorization:'Bearer '+generateToken({username:role,role,name:role})}:{})},body:JSON.stringify(body)});
try {
  assert.equal((await send(null)).status,401);
  assert.equal((await send('user')).status,403);
  assert.equal((await send('tester','invalid')).status,400);
  assert.equal(seen.length,0);
  const response=await send('tester');assert.equal(response.status,200);
  assert.equal((await response.json()).sdp,'test-answer');
  assert.deepEqual(seen,[{owner:'tester',body,path:`/video/sessions/${session}/offer`}]);
  console.log('PASS: WebRTC signaling requires tester JWT, validates session ID, and forwards parsed SDP JSON intact.');
} finally {server.closeAllConnections();server.close();upstream.closeAllConnections();upstream.close();}
