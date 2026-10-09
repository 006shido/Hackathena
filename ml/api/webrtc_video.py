"""Authenticated WebRTC video inference, with a single latest-frame slot."""
import asyncio
import json
import os
import time
from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, Field
from PIL import Image
from ml.api.video_preview import registry, require_owner, run_worker

router = APIRouter(prefix='/video')

class Offer(BaseModel):
    type: str = Field(pattern='^offer$')
    sdp: str = Field(min_length=1, max_length=100000)

def rtc_configuration():
    from aiortc import RTCConfiguration, RTCIceServer
    servers = [RTCIceServer(urls='stun:stun.l.google.com:19302')]
    urls = [value.strip() for value in os.environ.get('VITE_TURN_URLS', '').split(',') if value.strip()]
    if urls:
        servers.append(RTCIceServer(urls=urls, username=os.environ.get('VITE_TURN_USERNAME'),
                                   credential=os.environ.get('VITE_TURN_CREDENTIAL')))
    return RTCConfiguration(iceServers=servers)

async def close_peer(app, session_id):
    peers = getattr(app.state, 'video_peers', {})
    peer = peers.pop(session_id, None)
    if peer:
        expiry = getattr(peer, 'swap_expiry', None)
        if expiry and expiry is not asyncio.current_task():
            expiry.cancel()
        negotiation = getattr(peer, 'swap_negotiation', None)
        if negotiation and negotiation is not asyncio.current_task() and not negotiation.done():
            negotiation.cancel()
            await asyncio.gather(negotiation, return_exceptions=True)
        tracks = getattr(peer, 'swap_tracks', [])
        for track in tracks:
            track.stop()
        await asyncio.gather(*(track.reader for track in tracks), return_exceptions=True)
        await peer.close()

async def close_all(app):
    await asyncio.gather(*(close_peer(app, key) for key in list(getattr(app.state, 'video_peers', {}))))

@router.post('/sessions/{session_id}/offer')
async def offer(session_id: str, body: Offer, request: Request, x_ml_owner: str = Header(default='')):
    require_owner(x_ml_owner)
    if os.environ.get('ML_ENABLE_WEBRTC_VIDEO') != '1':
        raise HTTPException(503, 'WebRTC inference is not enabled.')
    sessions = registry(request)
    entry = sessions.get(session_id)
    if entry is None or entry.owner != x_ml_owner:
        raise HTTPException(404, 'Video session not found.')
    from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
    from av import VideoFrame
    if not hasattr(request.app.state, 'video_peers'):
        request.app.state.video_peers = {}
    peers = request.app.state.video_peers
    if session_id in peers:
        raise HTTPException(409, 'Session already has a media connection.')
    pc = RTCPeerConnection(rtc_configuration())
    pc.swap_negotiation = asyncio.current_task()
    pc.swap_tracks = []
    peers[session_id] = pc
    channels = []

    class Transform(VideoStreamTrack):
        def __init__(self, incoming):
            super().__init__()
            self.incoming = incoming
            self.latest = None
            self.available = asyncio.Event()
            self.reader = asyncio.create_task(self.read_latest())
            self.processed = 0
            self.started = time.monotonic()

        async def read_latest(self):
            try:
                while True:
                    self.latest = await self.incoming.recv()
                    self.available.set()
            except asyncio.CancelledError:
                raise
            except Exception:
                self.latest = None
            finally:
                self.available.set()

        async def recv(self):
            await self.available.wait()
            # Acquire GPU before selecting the latest frame, avoiding old-frame queues.
            async with request.app.state.inference_lock:
                if self.reader.done() and self.latest is None:
                    from aiortc.mediastreams import MediaStreamError
                    raise MediaStreamError
                if sessions.get(session_id) is not entry:
                    from aiortc.mediastreams import MediaStreamError
                    raise MediaStreamError
                frame = self.latest
                self.latest = None
                self.available.clear()
                if frame is None:
                    from aiortc.mediastreams import MediaStreamError
                    raise MediaStreamError
                if frame.width * frame.height > 1920 * 1080:
                    from aiortc.mediastreams import MediaStreamError
                    raise MediaStreamError
                entry.busy = True
                try:
                    image = Image.fromarray(frame.to_ndarray(format='rgb24'))
                    rgb, diagnostics = await run_worker(entry.pipeline.process, image, frame.time * 1000)
                    entry.touched = time.monotonic()
                finally:
                    entry.busy = False
            result = VideoFrame.from_ndarray(rgb, format='rgb24')
            result.pts, result.time_base = frame.pts, frame.time_base
            self.processed += 1
            telemetry = dict(face_detected=bool(diagnostics['face_detected']),
                             face_changed=bool(diagnostics.get('face_changed', diagnostics.get('changed_pixels', 0) > 0)),
                             pipeline_ms=diagnostics['pipeline_ms'],
                             fps=self.processed / max(.001, time.monotonic() - self.started), transport='webrtc')
            for channel in channels:
                if channel.readyState == 'open' and channel.bufferedAmount < 4096:
                    channel.send(json.dumps(telemetry))
            return result

        def stop(self):
            if self.readyState == 'ended':
                return
            super().stop()
            self.reader.cancel()

    @pc.on('datachannel')
    def datachannel(channel):
        channels.append(channel)

    @pc.on('track')
    def track(incoming):
        if incoming.kind == 'video':
            transformed = Transform(incoming)
            pc.swap_tracks.append(transformed)
            pc.addTrack(transformed)
            @incoming.on('ended')
            async def ended():
                transformed.stop()
                await close_peer(request.app, session_id)
        else:
            incoming.stop()

    @pc.on('connectionstatechange')
    async def state_change():
        if pc.connectionState in ('failed', 'closed'):
            await close_peer(request.app, session_id)

    async def expire_unconnected():
        await asyncio.sleep(30)
        if peers.get(session_id) is pc and pc.connectionState != 'connected':
            await close_peer(request.app, session_id)
    pc.swap_expiry = asyncio.create_task(expire_unconnected())
    try:
        await pc.setRemoteDescription(RTCSessionDescription(sdp=body.sdp, type=body.type))
        await pc.setLocalDescription(await pc.createAnswer())
        return dict(sdp=pc.localDescription.sdp, type=pc.localDescription.type, transport='webrtc')
    except (ValueError, AssertionError) as error:
        await close_peer(request.app, session_id)
        raise HTTPException(400, 'Invalid WebRTC offer.') from error
    except BaseException:
        await close_peer(request.app, session_id)
        raise
    finally:
        pc.swap_negotiation = None
