"""Opt-in receiver-media research analysis, never sender-flag classification."""
import asyncio, io
import numpy as np
from PIL import Image,UnidentifiedImageError
from fastapi import APIRouter,Request,UploadFile,File,Header,HTTPException
from ml.api.video_preview import run_worker

router=APIRouter(prefix='/detection')

def engine(request,owner):
    if not owner:raise HTTPException(401,'Authenticated receiver is required.')
    model=getattr(request.app.state,'media_detector',None)
    if model is None:raise HTTPException(503,'Independent research detection is not enabled.')
    return model

async def execute(request,key,function,*args):
    lock=request.app.state.inference_lock
    if not hasattr(request.app.state,'pending_detection'):
        request.app.state.pending_detection=set()
    pending=request.app.state.pending_detection
    if key in pending or len(pending)>=4:
        raise HTTPException(429,'A detector sample is already pending; retry with fresh media.')
    pending.add(key)
    acquired=False
    try:
        # FIFO lock acquisition gives periodic analysis a turn between swap frames.
        # Bound waiting and pending samples to avoid a backlog of stale media.
        try:
            await asyncio.wait_for(lock.acquire(),timeout=2)
            acquired=True
        except asyncio.TimeoutError:
            raise HTTPException(429,'GPU busy; no fresh detector measurement.')
        return await run_worker(function,*args)
    finally:
        if acquired:lock.release()
        pending.discard(key)

@router.post('/video')
async def video(request:Request,image:UploadFile=File(...),x_ml_owner:str=Header(default='')):
    model=engine(request,x_ml_owner)
    data=await image.read(4*1024*1024+1)
    if len(data)>4*1024*1024:raise HTTPException(413,'Frame exceeds 4 MB.')
    try:
        decoded=Image.open(io.BytesIO(data))
        if decoded.width>1920 or decoded.height>1080:raise HTTPException(413,'Frame dimensions too large.')
        decoded=decoded.convert('RGB')
    except (UnidentifiedImageError,OSError,ValueError):raise HTTPException(422,'Invalid image.')
    result=await execute(request,(x_ml_owner,'video'),model.video_score,decoded)
    return dict(**result,interpretation='Experimental manipulation evidence; not proof of fake or genuine media.',source='received pixels',validated_for_live_calls=False)

@router.post('/audio')
async def audio(request:Request,x_ml_owner:str=Header(default=''),x_sample_rate:int=Header(default=0)):
    model=engine(request,x_ml_owner);chunks=[];size=0
    async for chunk in request.stream():
        size+=len(chunk)
        if size>2304000:raise HTTPException(413,'Audio segment exceeds six seconds at 96 kHz.')
        chunks.append(chunk)
    data=b''.join(chunks)
    if len(data)%4:raise HTTPException(422,'Expected float32 mono PCM.')
    pcm=np.frombuffer(data,dtype='<f4').copy()
    try:result=await execute(request,(x_ml_owner,'audio'),model.audio_score,pcm,x_sample_rate)
    except ValueError as error:raise HTTPException(422,str(error))
    return dict(**result,interpretation='Experimental spoof evidence; codec-related false alarms remain unresolved.',source='received audio PCM',validated_for_live_calls=False)
