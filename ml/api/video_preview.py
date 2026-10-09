"""Opt-in video preview routes; production remains disabled until acceptance."""
import asyncio, io, os, time, uuid
from dataclasses import dataclass
from fastapi import APIRouter, File, Header, HTTPException, Request, UploadFile, Form
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError
from ml.inference.frame_pipeline import NeuralFrameSession
from ml.training.face_preprocessing import FaceDetectionError

router=APIRouter(prefix='/video')
TTL_SECONDS=120
MAX_SESSIONS=4
MAX_BYTES=10*1024*1024


@dataclass
class Entry:
    owner: str
    pipeline: NeuralFrameSession
    touched: float
    backend: str='trained-model'
    busy: bool=False

def close_entry(entry):
    close=getattr(entry.pipeline,'close',None)
    if close:close()

async def run_worker(function,*args):
    # A cancelled request must not release the GPU lock while its thread runs.
    worker=asyncio.create_task(asyncio.to_thread(function,*args))
    try:return await asyncio.shield(worker)
    except asyncio.CancelledError:
        result=await worker
        close=getattr(result,'close',None)
        if close:close()
        raise


def create_pipeline(request,engine,image,mode,backend):
    if backend=='trained-model':return NeuralFrameSession(engine,image,mode)
    if backend!='research-reference':raise ValueError('Unknown video preview backend.')
    # Explicit local research opt-in; never replaces still inference or defaults.
    from types import SimpleNamespace
    from ml.experiments.reference_backend.frame_session import ReferenceFrameSession
    if not hasattr(request.app.state,'research_reference'):
        raise HTTPException(503,'Research backend is not warmed. Restart with the research preview flags enabled.')
    reference_engine=SimpleNamespace(reference=request.app.state.research_reference,preprocessor=engine.preprocessor,
        identity_model=engine.model.arcface,identity_guard=True,
        video_landmarks=os.environ.get('ML_VIDEO_TRACKING')=='1')
    return ReferenceFrameSession(reference_engine,image)


def registry(request):
    if os.environ.get('ML_ENABLE_VIDEO_PREVIEW') != '1':
        raise HTTPException(503,'Neural video preview is disabled pending quality acceptance.')
    if not getattr(request.app.state,'ready',False):
        raise HTTPException(503,'Model is not ready.')
    if not hasattr(request.app.state,'video_sessions'):
        request.app.state.video_sessions={}
    sessions=request.app.state.video_sessions
    now=time.monotonic()
    for key in list(sessions):
        if not sessions[key].busy and now-sessions[key].touched>TTL_SECONDS:
            close_entry(sessions.pop(key))
    return sessions


async def decode(file):
    data=await file.read(MAX_BYTES+1)
    if not data:raise HTTPException(400,'Image is empty.')
    if len(data)>MAX_BYTES:raise HTTPException(413,'Image exceeds 10 MB.')
    try:
        image=Image.open(io.BytesIO(data))
        if image.width*image.height>1920*1080:raise HTTPException(413,'Preview images must be at most 1920x1080.')
        return image.convert('RGB')
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError):
        raise HTTPException(400,'Invalid image.')


def require_owner(owner):
    if not owner:raise HTTPException(401,'Authenticated tester identity is required.')


@router.post('/sessions')
async def create_session(request: Request, source: UploadFile=File(...), x_ml_owner: str=Header(default='')):
    require_owner(x_ml_owner)
    sessions=registry(request)
    image=await decode(source)
    lock=request.app.state.inference_lock
    if lock.locked():raise HTTPException(429,'GPU is busy; retry later.')
    async with lock:
        if len(sessions)>=MAX_SESSIONS:raise HTTPException(429,'Video session limit reached.')
        engine=request.app.state.engine
        mode=engine.correspondence
        backend=os.environ.get('ML_VIDEO_BACKEND','trained-model')
        if backend not in ('trained-model','research-reference'):raise HTTPException(503,'Unknown video preview backend.')
        if backend=='trained-model' and mode not in ('legacy','visibility'):raise HTTPException(503,'This correspondence is not supported for video preview.')
        try:
            pipeline=await run_worker(create_pipeline,request,engine,image,mode,backend)
        except FaceDetectionError as error:
            raise HTTPException(422,str(error))
        key=uuid.uuid4().hex
        sessions[key]=Entry(x_ml_owner,pipeline,time.monotonic(),backend)
    return dict(session_id=key,expires_after_seconds=TTL_SECONDS,preview=True,backend=backend)


@router.post('/sessions/{session_id}/frame')
async def process_frame(session_id: str, request: Request, frame: UploadFile=File(...),
        timestamp_ms: float=Form(...), x_ml_owner: str=Header(default='')):
    require_owner(x_ml_owner)
    sessions=registry(request)
    entry=sessions.get(session_id)
    if entry is None or entry.owner!=x_ml_owner:raise HTTPException(404,'Video session not found.')
    if session_id in getattr(request.app.state,'video_peers',{}):raise HTTPException(409,'Session is using WebRTC media.')
    image=await decode(frame)
    lock=request.app.state.inference_lock
    if lock.locked():raise HTTPException(429,'GPU is busy; drop this frame.')
    async with lock:
        if sessions.get(session_id) is not entry:raise HTTPException(404,'Video session not found.')
        entry.busy=True
        try:
            rgb,diagnostics=await run_worker(entry.pipeline.process,image,timestamp_ms)
        except ValueError as error:
            raise HTTPException(400,str(error))
        finally:
            entry.busy=False
        entry.touched=time.monotonic()
        buffer=io.BytesIO()
        Image.fromarray(rgb).save(buffer,format='JPEG',quality=90)
    return Response(buffer.getvalue(),media_type='image/jpeg',headers={
        'X-Face-Detected':str(diagnostics['face_detected']).lower(),
        'X-Face-Changed':str(diagnostics.get('face_changed',diagnostics['changed_pixels']>0)).lower(),
        'X-Pipeline-Ms':str(round(diagnostics['pipeline_ms'],2)),
        'Cache-Control':'no-store',
        'X-Video-Backend':entry.backend,
    })


@router.delete('/sessions/{session_id}')
async def delete_session(session_id: str, request: Request, x_ml_owner: str=Header(default='')):
    require_owner(x_ml_owner)
    sessions=registry(request)
    entry=sessions.get(session_id)
    if entry is not None and entry.owner!=x_ml_owner:raise HTTPException(404,'Video session not found.')
    from ml.api.webrtc_video import close_peer
    await close_peer(request.app,session_id)
    async with request.app.state.inference_lock:
        removed=sessions.pop(session_id,None)
        if removed:close_entry(removed)
    return dict(deleted=True)
