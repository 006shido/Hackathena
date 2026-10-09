"""Bounded real RTP/DTLS inference check using recorded media, never a webcam."""
import asyncio, io, json, time, sys, wave, os
from fractions import Fraction
from pathlib import Path
import av
import numpy as np
import requests
from PIL import Image
from aiortc import RTCPeerConnection, RTCConfiguration, RTCSessionDescription, VideoStreamTrack

BASE = os.environ.get('TEST_APP_URL', 'http://127.0.0.1:5001')
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'ml/experiments/reference_backend/webrtc_transport'

async def main():
    with_detection='--with-detection' in sys.argv
    OUTPUT.mkdir(parents=True, exist_ok=True)
    login = requests.post(BASE+'/api/auth/login',json={'username':'tester','password':'tester123'},timeout=10)
    login.raise_for_status()
    headers = {'Authorization':'Bearer '+login.json()['token']}
    container = av.open(str(ROOT/'ml/experiments/reference_backend/videos/faceocc2.webm'))
    original = next(container.decode(video=0)).to_ndarray(format='rgb24')
    container.close()
    image = Image.fromarray(original)
    image.thumbnail((640,480)); original = np.asarray(image).copy()
    source = ROOT/'ml/data/celeba/img_align_celeba/004831.jpg'

    def create_session():
        with source.open('rb') as handle:
            response=requests.post(BASE+'/api/ml/video/sessions',headers=headers,files={'source':('source.jpg',handle,'image/jpeg')},timeout=20)
        response.raise_for_status(); return response.json()['session_id']

    session = create_session()
    buffer=io.BytesIO();image.save(buffer,format='JPEG',quality=90)
    http_times=[]
    try:
        for index in range(20):
            started=time.perf_counter()
            response=requests.post(BASE+f'/api/ml/video/sessions/{session}/frame',headers=headers,
                files={'frame':('frame.jpg',buffer.getvalue(),'image/jpeg')},data={'timestamp_ms':index*100},timeout=10)
            response.raise_for_status();http_times.append((time.perf_counter()-started)*1000)
    finally:
        requests.delete(BASE+f'/api/ml/video/sessions/{session}',headers=headers,timeout=10).raise_for_status()
    session=create_session()
    sent={}; telemetry=[]; ages=[]; frames=[]
    class Camera(VideoStreamTrack):
        def __init__(self):super().__init__();self.index=0;self.started=None
        async def recv(self):
            if self.started is None:self.started=time.perf_counter()
            await asyncio.sleep(max(0,self.started+self.index/15-time.perf_counter()))
            rgb=original.copy()
            # Binary frame ID outside the face: survives video compression.
            for bit in range(12):rgb[:8,bit*8:(bit+1)*8]=255 if self.index&(1<<bit) else 0
            frame=av.VideoFrame.from_ndarray(rgb,format='rgb24')
            frame.pts=self.index*6000;frame.time_base=Fraction(1,90000)
            sent[self.index]=time.perf_counter();self.index+=1
            return frame
    pc=RTCPeerConnection(RTCConfiguration(iceServers=[]))
    camera=Camera();pc.addTrack(camera)
    channel=pc.createDataChannel('swap-telemetry',ordered=False,maxRetransmits=0)
    @channel.on('message')
    def message(data):telemetry.append(json.loads(data))
    received=asyncio.Future()
    detection_results=[]
    async def detection_probe():
        with wave.open(str(ROOT/'ml/experiments/reference_backend/videos/speech.wav'),'rb') as speech:
            rate=speech.getframerate();pcm=np.frombuffer(speech.readframes(speech.getnframes()),dtype='<i2').astype('<f4')/32768
        for _ in range(3):
            response=await asyncio.to_thread(requests.post,BASE+'/api/ml/detection/video',headers=headers,
                files={'image':('received.jpg',buffer.getvalue(),'image/jpeg')},timeout=10)
            detection_results.append(dict(kind='video',http_status=response.status_code,status=response.json().get('status')))
            response=await asyncio.to_thread(requests.post,BASE+'/api/ml/detection/audio',
                headers={**headers,'Content-Type':'application/octet-stream','X-Sample-Rate':str(rate)},data=pcm.tobytes(),timeout=10)
            detection_results.append(dict(kind='audio',http_status=response.status_code,status=response.json().get('status')))
            await asyncio.sleep(.7)
    @pc.on('track')
    def track(value):
        if value.kind=='video' and not received.done():received.set_result(value)
    try:
        await pc.setLocalDescription(await pc.createOffer())
        def negotiate():
            response=requests.post(BASE+f'/api/ml/video/sessions/{session}/offer',headers=headers,
                                   json={'sdp':pc.localDescription.sdp,'type':pc.localDescription.type},timeout=20)
            response.raise_for_status();return response.json()
        answer=await asyncio.to_thread(negotiate)
        await pc.setRemoteDescription(RTCSessionDescription(sdp=answer['sdp'],type=answer['type']))
        incoming=await asyncio.wait_for(received,10)
        probe=asyncio.create_task(detection_probe()) if with_detection else None
        first=None
        while len(frames)<40:
            frame=await asyncio.wait_for(incoming.recv(),10)
            now=time.perf_counter();first=now if first is None else first
            rgb=frame.to_ndarray(format='rgb24')
            index=sum((1<<bit) for bit in range(12) if rgb[:8,bit*8:(bit+1)*8].mean()>128)
            if index in sent:ages.append((now-sent[index])*1000)
            frames.append(now)
        if probe:await probe
        result=dict(scope='Same-machine recorded face, actual RTP/DTLS and actual swap pipeline. Not a forwarded two-device latency benchmark.',
                    http_requests=20,http_roundtrip_median_ms=float(np.median(http_times)),
                    http_processing_loop_fps=1000/float(np.mean(http_times)),
                    webrtc_received_frames=len(frames),webrtc_received_fps=(len(frames)-1)/(frames[-1]-frames[0]),
                    webrtc_age_samples=len(ages),webrtc_frame_age_median_ms=float(np.median(ages)),
                    webrtc_frame_age_p95_ms=float(np.percentile(ages,95)),
                    telemetry_messages=len(telemetry),changed_frames=sum(value['face_changed'] for value in telemetry),
                    pipeline_median_ms=float(np.median([value['pipeline_ms'] for value in telemetry])))
        if with_detection:
            result['concurrent_detection']=detection_results
            assert all(value['http_status']==200 for value in detection_results),'Detection was starved during streaming.'
        assert telemetry and result['changed_frames']>0,'No swapped video telemetry received.'
        assert len(ages)>=30,'Frame ages could not be verified.'
        Image.fromarray(rgb).save(OUTPUT/'received_frame.png')
        (OUTPUT/('benchmark_with_detection.json' if with_detection else 'benchmark.json')).write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2))
    finally:
        camera.stop();await pc.close()
        await asyncio.to_thread(requests.delete,BASE+f'/api/ml/video/sessions/{session}',headers=headers,timeout=10)

if __name__=='__main__':asyncio.run(main())
