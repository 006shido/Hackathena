"""Regression: the held-out target-identity failure must preserve original pixels."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from types import SimpleNamespace
import numpy as np,torch
from PIL import Image
from ml.models.face_swap_model import ArcFaceIdentityExtractor
from ml.training.face_preprocessing import RealFacePreprocessor
from ml.experiments.reference_backend.opencv_reference import OpenCVReference
from ml.experiments.reference_backend.frame_session import ReferenceFrameSession

ROOT=Path(__file__).resolve().parents[3]
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
weights=ROOT/'ml/models/weights/ms1mv2_iresnet50.pth'
assert hashlib.sha256(weights.read_bytes()).hexdigest().upper()=='2B75B93C48B01C78A4263F7295AB2DBF84F85190C51BE45B92C4C0B0AEDCEAA3'
engine=SimpleNamespace(reference=OpenCVReference(gpu=True),preprocessor=RealFacePreprocessor(image_size=128),
    identity_model=ArcFaceIdentityExtractor(str(weights)).cuda().eval(),identity_guard=True)
records=[]
for source,target,rejected in [('186165.jpg','070860.jpg',True),('197935.jpg','098180.jpg',False)]:
    image=Image.open(ROOT/'ml/data/celeba/img_align_celeba'/target).convert('RGB')
    session=ReferenceFrameSession(engine,Image.open(ROOT/'ml/data/celeba/img_align_celeba'/source))
    output,diagnostics=session.process(image,1)
    assert diagnostics['face_detected']
    assert diagnostics['identity_accepted'] is not rejected,diagnostics
    assert diagnostics['face_changed'] is not rejected,diagnostics
    if rejected:assert np.array_equal(output,np.asarray(image)), 'Rejected swap changed camera pixels.'
    else:assert not np.array_equal(output,np.asarray(image)), 'Valid swap did not change pixels.'
    records.append(dict(source=source,target=target,diagnostics=diagnostics))
    print(records[-1],flush=True)
(Path(__file__).parent/'identity_guard_test.json').write_text(json.dumps(records,indent=2))
print('PASS: known identity failure rejected exactly; valid swap accepted.')
