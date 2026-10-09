import assert from 'node:assert/strict';
import http from 'node:http';
import { once } from 'node:events';
import express from 'express';
import { generateToken } from '../src/auth.js';

const seen: {owner: unknown; path: unknown; rate: unknown}[] = [];
const upstream = http.createServer((req,res) => {
  seen.push({owner:req.headers['x-ml-owner'],path:req.url,rate:req.headers['x-sample-rate']});
  req.resume();res.writeHead(200,{'content-type':'application/json'});
  res.end(JSON.stringify({status:'measured',score:.37,validated_for_live_calls:false}));
});
upstream.listen(0,'127.0.0.1');await once(upstream,'listening');
process.env.ML_SERVICE_PORT=String((upstream.address() as import('node:net').AddressInfo).port);
const {mlRouter}=await import('../src/ml.js');
const app=express();app.use('/api/ml',mlRouter);
const server=app.listen(0,'127.0.0.1');await once(server,'listening');
const base=`http://127.0.0.1:${(server.address() as import('node:net').AddressInfo).port}/api/ml/detection`;
try {
  assert.equal((await fetch(`${base}/video`,{method:'POST',body:'frame'})).status,401);
  assert.equal(seen.length,0);
  for(const role of ['user','tester'] as const) {
    const token=generateToken({username:role,role,name:role});
    const response=await fetch(`${base}/audio`,{method:'POST',body:new Uint8Array(16),headers:{Authorization:`Bearer ${token}`,'Content-Type':'application/octet-stream','X-Sample-Rate':'16000','X-ML-Owner':'spoofed'}});
    assert.equal(response.status,200);assert.equal(response.headers.get('cache-control'),'no-store');
    assert.equal((await response.json()).score,.37);
    assert.deepEqual(seen.at(-1),{owner:role,path:'/detection/audio',rate:'16000'});
  }
  console.log('PASS: independent detection requires authentication, accepts both receiver roles, and ignores spoofed ownership.');
} finally {server.closeAllConnections();server.close();upstream.closeAllConnections();upstream.close();}
