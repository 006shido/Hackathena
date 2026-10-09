import { Router, Request, Response } from 'express';
import http from 'http';
import { verifyToken } from './auth.js';

export const mlRouter = Router();

const FASTAPI_HOST = process.env.ML_SERVICE_HOST || '127.0.0.1';
const FASTAPI_PORT = process.env.ML_SERVICE_PORT ? parseInt(process.env.ML_SERVICE_PORT, 10) : 8000;
const ML_TIMEOUT_MS = process.env.ML_TIMEOUT_MS ? parseInt(process.env.ML_TIMEOUT_MS, 10) : 60000; // 60s timeout

function proxyVideo(req: Request, res: Response, destination: string) {
  const bearer = req.headers.authorization;
  const user = bearer?.startsWith('Bearer ') ? verifyToken(bearer.slice(7)) : null;
  if (!user) return res.status(401).json({ error: 'Authenticated tester session required.' });
  if (user.role !== 'tester') return res.status(403).json({ error: 'Neural video preview is available only to testers.' });
  const headers: http.OutgoingHttpHeaders = { 'x-ml-owner': user.username };
  if (req.headers['content-type']) headers['content-type'] = req.headers['content-type'];
  if (req.headers['content-length']) headers['content-length'] = req.headers['content-length'];
  const jsonBody = req.is('application/json') ? Buffer.from(JSON.stringify(req.body)) : null;
  if (jsonBody) headers['content-length'] = jsonBody.length;
  const upstream = http.request({ host: FASTAPI_HOST, port: FASTAPI_PORT, path: destination,
    method: req.method, headers, timeout: destination.endsWith('/offer') ? 20000 : 10000 }, response => {
    res.status(response.statusCode || 502);
    for (const name of ['content-type', 'x-face-detected', 'x-face-changed', 'x-pipeline-ms', 'x-video-backend', 'cache-control']) {
      const value = response.headers[name];
      if (value) res.setHeader(name, value);
    }
    response.pipe(res);
  });
  upstream.on('timeout', () => upstream.destroy(new Error('Neural video preview timed out.')));
  upstream.on('error', () => {
    if (!res.headersSent) res.status(503).json({ error: 'Neural video preview service unavailable.' });
    else res.destroy();
  });
  req.on('aborted', () => upstream.destroy());
  if (jsonBody) upstream.end(jsonBody); else req.pipe(upstream);
}

mlRouter.post('/video/sessions', (req, res) => proxyVideo(req, res, '/video/sessions'));
mlRouter.post('/video/sessions/:sessionId/offer', (req, res) => {
  const id = String(req.params.sessionId);
  if (!/^[a-f0-9]{32}$/.test(id)) return res.status(400).json({ error: 'Invalid session ID.' });
  return proxyVideo(req, res, `/video/sessions/${id}/offer`);
});
function proxyDetection(req: Request, res: Response, media: 'video' | 'audio') {
  const bearer = req.headers.authorization;
  const user = bearer?.startsWith('Bearer ') ? verifyToken(bearer.slice(7)) : null;
  if (!user) return res.status(401).json({ error: 'Authenticated receiver session required.' });
  const headers: http.OutgoingHttpHeaders = { 'x-ml-owner': user.username };
  for (const name of ['content-type','content-length','x-sample-rate']) {
    if (req.headers[name]) headers[name] = req.headers[name];
  }
  const upstream = http.request({ host: FASTAPI_HOST, port: FASTAPI_PORT,
    path: `/detection/${media}`, method: 'POST', headers, timeout: 10000 }, response => {
    res.status(response.statusCode || 502);res.setHeader('Cache-Control','no-store');
    if (response.headers['content-type']) res.setHeader('Content-Type',response.headers['content-type']);
    response.pipe(res);
  });
  upstream.on('timeout', () => upstream.destroy(new Error('Detector timed out')));
  upstream.on('error', () => { if (!res.headersSent) res.status(503).json({error:'Independent detector unavailable.'}); else res.destroy(); });
  req.on('aborted', () => upstream.destroy());req.pipe(upstream);
}
mlRouter.post('/detection/video', (req,res) => proxyDetection(req,res,'video'));
mlRouter.post('/detection/audio', (req,res) => proxyDetection(req,res,'audio'));
mlRouter.post('/video/sessions/:sessionId/frame', (req, res) => {
  const id = String(req.params.sessionId);
  if (!/^[a-f0-9]{32}$/.test(id)) return res.status(400).json({ error: 'Invalid session ID.' });
  return proxyVideo(req, res, `/video/sessions/${id}/frame`);
});
mlRouter.delete('/video/sessions/:sessionId', (req, res) => {
  const id = String(req.params.sessionId);
  if (!/^[a-f0-9]{32}$/.test(id)) return res.status(400).json({ error: 'Invalid session ID.' });
  return proxyVideo(req, res, `/video/sessions/${id}`);
});

/**
 * GET /api/ml/health
 * Proxies health status from the Python FastAPI ML service.
 */
mlRouter.get('/health', (_req: Request, res: Response) => {
  const proxyReq = http.request(
    {
      host: FASTAPI_HOST,
      port: FASTAPI_PORT,
      path: '/health',
      method: 'GET',
      headers: {
        Accept: 'application/json',
      },
      timeout: 5000,
    },
    (proxyRes) => {
      const chunks: Buffer[] = [];
      proxyRes.on('data', (chunk) => chunks.push(chunk));
      proxyRes.on('end', () => {
        const body = Buffer.concat(chunks).toString('utf8');
        const statusCode = proxyRes.statusCode || 500;
        try {
          const json = JSON.parse(body);
          return res.status(statusCode).json(json);
        } catch {
          return res.status(statusCode).send(body);
        }
      });
    }
  );

  proxyReq.on('timeout', () => {
    proxyReq.destroy();
    if (!res.headersSent) {
      return res.status(504).json({
        error: 'ML service health check timed out.',
        ready: false,
      });
    }
  });

  proxyReq.on('error', (err: any) => {
    if (!res.headersSent) {
      if (err.code === 'ECONNREFUSED') {
        return res.status(503).json({
          error: `ML Inference Service Unavailable: Connection to FastAPI (${FASTAPI_HOST}:${FASTAPI_PORT}) refused.`,
          ready: false,
        });
      }
      return res.status(502).json({
        error: `ML service communication error: ${err.message}`,
        ready: false,
      });
    }
  });

  proxyReq.end();
});

/**
 * POST /api/ml/face-swap
 * Thin proxy forwarding multipart/form-data (source, target) to FastAPI's POST /infer.
 * Preserves the full diagnostic response and converts backend error formats into clean JSON.
 */
mlRouter.post('/face-swap', (req: Request, res: Response) => {
  const contentType = req.headers['content-type'];
  if (!contentType || !contentType.toLowerCase().startsWith('multipart/form-data')) {
    return res.status(400).json({
      error: 'Invalid Content-Type. Expected multipart/form-data with source and target files.',
    });
  }

  // Prepare proxy headers
  const proxyHeaders: Record<string, string | string[] | undefined> = {
    'content-type': req.headers['content-type'],
    host: `${FASTAPI_HOST}:${FASTAPI_PORT}`,
  };

  if (req.headers['content-length']) {
    proxyHeaders['content-length'] = req.headers['content-length'];
  }

  const proxyReq = http.request(
    {
      host: FASTAPI_HOST,
      port: FASTAPI_PORT,
      path: '/infer',
      method: 'POST',
      headers: proxyHeaders,
      timeout: ML_TIMEOUT_MS,
    },
    (proxyRes) => {
      const chunks: Buffer[] = [];
      proxyRes.on('data', (chunk) => chunks.push(chunk));
      proxyRes.on('end', () => {
        const rawBody = Buffer.concat(chunks).toString('utf8');
        const statusCode = proxyRes.statusCode || 500;

        try {
          const json = JSON.parse(rawBody);

          // Success path (HTTP 200)
          if (statusCode === 200) {
            return res.status(200).json(json);
          }

          // Concurrency rate limiting from FastAPI (HTTP 429)
          if (statusCode === 429) {
            return res.status(429).json({
              error: json.detail || 'Inference engine is currently busy. Single GPU worker limit enforced.',
            });
          }

          // Validation / Input format error (HTTP 400)
          if (statusCode === 400) {
            return res.status(400).json({
              error: json.detail || 'Invalid request format or unsupported image file.',
            });
          }

          // File payload too large (HTTP 413)
          if (statusCode === 413) {
            return res.status(413).json({
              error: json.detail || 'Uploaded image exceeds maximum allowed size.',
            });
          }

          // FastAPI Pydantic missing field validation (HTTP 422) -> normalize to 400
          if (statusCode === 422) {
            let errorMsg = 'Missing required field in multipart request.';
            if (Array.isArray(json.detail)) {
              const missingFields = json.detail
                .map((d: any) => d.loc?.[d.loc.length - 1])
                .filter(Boolean);
              if (missingFields.length > 0) {
                errorMsg = `Missing required file field: ${missingFields.join(', ')}. Both 'source' and 'target' images must be uploaded.`;
              }
            } else if (typeof json.detail === 'string') {
              errorMsg = json.detail;
            }
            return res.status(400).json({ error: errorMsg });
          }

          // Server errors (HTTP 500+)
          return res.status(statusCode).json({
            error: typeof json.detail === 'string' ? json.detail : 'ML Inference failed on server.',
          });
        } catch {
          // If response body is not JSON
          return res.status(statusCode).json({
            error: `Inference proxy received non-JSON response with HTTP ${statusCode}.`,
          });
        }
      });
    }
  );

  // Handle timeout cleanly
  proxyReq.on('timeout', () => {
    proxyReq.destroy();
    if (!res.headersSent) {
      return res.status(504).json({
        error: `ML Inference Gateway Timeout: FastAPI service took longer than ${ML_TIMEOUT_MS / 1000}s to respond.`,
      });
    }
  });

  // Handle connection errors
  proxyReq.on('error', (err: any) => {
    console.error('[ML Proxy Error]', err.message);
    if (!res.headersSent) {
      if (err.code === 'ECONNREFUSED') {
        return res.status(503).json({
          error: `ML Inference Service Unavailable: Connection to FastAPI (${FASTAPI_HOST}:${FASTAPI_PORT}) refused. Ensure the Python ML service is running.`,
        });
      }
      return res.status(502).json({
        error: `ML Inference Gateway Error: ${err.message}`,
      });
    }
  });

  // Handle client aborting the request
  req.on('aborted', () => {
    proxyReq.destroy();
  });

  req.on('error', (err) => {
    proxyReq.destroy(err);
  });

  // Pipe raw multipart request body stream directly to FastAPI
  req.pipe(proxyReq);
});
