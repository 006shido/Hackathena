"""Real MediaPipe timestamp isolation and registry resource lifecycle."""
import os,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from PIL import Image
from ml.experiments.reference_backend.video_landmarks import VideoLandmarkDetector
from ml.api.video_preview import Entry,registry,TTL_SECONDS
from ml.experiments.reference_backend.frame_session import ReferenceFrameSession
from ml.training.face_preprocessing import RealFacePreprocessor

image=Image.open(ROOT/'ml/data/celeba/img_align_celeba/098180.jpg').convert('RGB')
first=VideoLandmarkDetector();second=VideoLandmarkDetector()
try:
    first.set_timestamp(1234.2);a=first.detect(image)
    second.set_timestamp(5.1);b=second.detect(image)
    assert first.task is not second.task
    assert first.last_timestamp==1234 and second.last_timestamp==5
    first.set_timestamp(1234.6);first.detect(image)
    assert first.last_timestamp==1235 and second.last_timestamp==5
    assert len(a.dense_landmarks)==478 and len(b.dense_landmarks)==478
finally:first.close();second.close()
try:first.detect(image);raise AssertionError('Closed detector was reused')
except ValueError:pass

shared=SimpleNamespace(preprocessor=RealFacePreprocessor(image_size=128),
    reference=SimpleNamespace(source_identity=lambda *_:None),video_landmarks=True,occlusion_temporal=True)
one=ReferenceFrameSession(shared,image);two=ReferenceFrameSession(shared,image)
assert one.preprocessor is not two.preprocessor and one.preprocessor is not shared.preprocessor
assert one.video_detector is not two.video_detector
assert one.occlusion_state is not two.occlusion_state
one.occlusion_state['marker']='one';two.occlusion_state['marker']='two'
one.close();assert not two.video_detector.closed
assert one.occlusion_state=={} and two.occlusion_state=={'marker':'two'}
try:one.process(image,1);raise AssertionError('Closed call session was reused')
except ValueError:pass
two.close()

class Pipeline:
    closed=0
    def close(self):self.closed+=1
old=Pipeline();busy=Pipeline();live=Pipeline();now=time.monotonic()
sessions={'old':Entry('a',old,now-TTL_SECONDS-1),'busy':Entry('b',busy,now-TTL_SECONDS-1,busy=True),'live':Entry('c',live,now)}
request=SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(ready=True,video_sessions=sessions)))
previous=os.environ.get('ML_ENABLE_VIDEO_PREVIEW');os.environ['ML_ENABLE_VIDEO_PREVIEW']='1'
try:
    assert registry(request) is sessions
    assert set(sessions)=={'busy','live'} and old.closed==1 and busy.closed==0 and live.closed==0
    sessions['busy'].busy=False;registry(request)
    assert busy.closed==1 and 'busy' not in sessions
finally:
    if previous is None:os.environ.pop('ML_ENABLE_VIDEO_PREVIEW',None)
    else:os.environ['ML_ENABLE_VIDEO_PREVIEW']=previous
print('PASS: real detector timestamp/state isolation, closed-detector rejection, expiry cleanup and busy-session protection.')
