"""Held-out photos through the actual default research frame-session path."""
import hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from ml.experiments.reference_backend.opencv_reference import OpenCVReference
from ml.experiments.reference_backend.frame_session import ReferenceFrameSession
from ml.models.face_swap_model import ArcFaceIdentityExtractor
from ml.training.face_preprocessing import RealFacePreprocessor,FaceDetectionError

ROOT=Path(__file__).resolve().parents[3]
def main():
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    weights=ROOT/'ml/models/weights/ms1mv2_iresnet50.pth'
    if hashlib.sha256(weights.read_bytes()).hexdigest()!='2b75b93c48b01c78a4263f7295ab2dbf84f85190c51be45b92c4c0b0aedceaa3':raise ValueError('Scorer checksum mismatch')
    root=Path(__file__).parent;previous=json.loads((root/'holdout/results.json').read_text());pairs=previous['pairs']
    engine=SimpleNamespace(reference=OpenCVReference(gpu=True),preprocessor=RealFacePreprocessor(image_size=128),identity_model=ArcFaceIdentityExtractor(str(weights)).cuda().eval(),identity_guard=True)
    engine.reference.warmup();out=root/'live_holdout';out.mkdir(exist_ok=True);records=[]
    for index,pair in enumerate(pairs):
        source=Image.open(ROOT/'ml/data/celeba/img_align_celeba'/pair['source']).convert('RGB');target=Image.open(ROOT/'ml/data/celeba/img_align_celeba'/pair['target']).convert('RGB');session=None
        try:
            session=ReferenceFrameSession(engine,source);output,diagnostic=session.process(target,0)
            if not diagnostic['face_changed']:assert np.array_equal(output,np.asarray(target))
            if index<12:Image.fromarray(np.concatenate([np.asarray(target),output],1)).save(out/f'pair_{index}.png')
            records.append(dict(index=index,source=pair['source'],target=pair['target'],**diagnostic))
        except FaceDetectionError as error:records.append(dict(index=index,source=pair['source'],target=pair['target'],source_failure=str(error),face_changed=False))
        finally:
            if session:session.close()
        if (index+1)%25==0:print(f'Live holdout {index+1}/{len(pairs)}',flush=True)
    accepted=[r for r in records if r['face_changed']];latency=[r['pipeline_ms'] for r in records if 'pipeline_ms' in r]
    report=dict(scope='Same existing 300 held-out photo pairs through complete default IMAGE research sessions on original images. No JPEG/WebRTC, physical device, video motion or all-frame perceptual pass.',seed=previous['seed'],count=len(records),accepted=len(accepted),source_failures=sum('source_failure' in r for r in records),identity_rejections=sum(r.get('identity_accepted') is False for r in records),missing_targets=sum(r.get('face_detected') is False for r in records),median_pipeline_ms=float(np.median(latency)),p95_pipeline_ms=float(np.percentile(latency,95)),accepted_mean_source_similarity=float(np.mean([r['source_similarity'] for r in accepted])) if accepted else None,records=records)
    (out/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='records'}))
if __name__=='__main__':main()
