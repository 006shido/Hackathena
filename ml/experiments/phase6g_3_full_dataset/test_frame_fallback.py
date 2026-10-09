"""Missing faces preserve the camera frame and reset tracking without a model."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from types import SimpleNamespace
import numpy as np
from PIL import Image
from ml.inference.frame_pipeline import NeuralFrameSession
from ml.training.face_preprocessing import NoFaceDetectedError

class MissingFace:
    def detect_landmarks(self, image):
        raise NoFaceDetectedError('No face in frame')

session=NeuralFrameSession.__new__(NeuralFrameSession)
session.engine=SimpleNamespace(preprocessor=MissingFace())
session.timestamp=None
session.previous_landmarks=np.ones((478,2))
pixels=np.random.default_rng(42).integers(0,256,(96,128,3),dtype=np.uint8)
output,diagnostic=session.process(Image.fromarray(pixels),1)
assert np.array_equal(output,pixels)
assert session.previous_landmarks is None
assert not diagnostic['face_detected'] and diagnostic['changed_pixels']==0
for timestamp in [1,0,float('nan'),float('inf')]:
    try:session.process(Image.fromarray(pixels),timestamp)
    except ValueError:pass
    else:raise AssertionError('Invalid timestamp accepted')
assert session.timestamp==1
print('PASS: missing-face camera fallback, tracking reset and timestamp validation.')
