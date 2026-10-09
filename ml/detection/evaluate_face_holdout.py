"""Full original/photo swap domain evaluation, with identity-disjoint calibration.

This diagnostic creates no new training checkpoint and changes no live settings.
It measures original and actually accepted swaps through identical crop/JPEG paths.
"""
import hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
from io import BytesIO
import numpy as np
import torch
from PIL import Image
from ml.detection.pretrained import PretrainedDetectors
from ml.experiments.reference_backend.opencv_reference import OpenCVReference
from ml.experiments.reference_backend.frame_session import ReferenceFrameSession
from ml.models.face_swap_model import ArcFaceIdentityExtractor
from ml.training.face_preprocessing import RealFacePreprocessor, FaceDetectionError

def jpeg(image):
    buffer=BytesIO();image.save(buffer,format='JPEG',quality=85)
    buffer.seek(0);return Image.open(buffer).convert('RGB')

def main():
    root=Path(__file__).resolve().parents[2];out=root/'ml/detection/evaluation';out.mkdir(exist_ok=True)
    previous=json.loads((root/'ml/experiments/reference_backend/holdout/results.json').read_text())
    identity_file=root/'ml/data/celeba/identity_CelebA.txt'
    if not identity_file.exists():identity_file=root/'ml/data/celeba/Anno/identity_CelebA.txt'
    identities=dict(line.split() for line in identity_file.read_text().splitlines() if line.strip())
    # Fixed identity partition independent of classifier scores and image labels.
    fold=lambda value:int(hashlib.sha256(('detector-holdout-20261006-'+value).encode()).hexdigest(),16)%2
    pairs=[(p,fold(identities[p['source']])) for p in previous['pairs']
           if fold(identities[p['source']])==fold(identities[p['target']])]
    weights=root/'ml/models/weights/ms1mv2_iresnet50.pth'
    if hashlib.sha256(weights.read_bytes()).hexdigest()!='2b75b93c48b01c78a4263f7295ab2dbf84f85190c51be45b92c4c0b0aedceaa3':raise ValueError('Identity asset mismatch')
    model=PretrainedDetectors()
    engine=SimpleNamespace(reference=OpenCVReference(gpu=True),preprocessor=RealFacePreprocessor(image_size=128),identity_model=ArcFaceIdentityExtractor(str(weights)).cuda().eval(),identity_guard=True)
    engine.reference.warmup();records=[];started=time.time()
    try:
        for index,(pair,part) in enumerate(pairs):
            source=Image.open(root/'ml/data/celeba/img_align_celeba'/pair['source']).convert('RGB')
            target=Image.open(root/'ml/data/celeba/img_align_celeba'/pair['target']).convert('RGB');session=None
            try:
                session=ReferenceFrameSession(engine,source);output,diagnostic=session.process(target,0)
                real=jpeg(target);fake=jpeg(Image.fromarray(output))
                negative=model.video_score(real)
                positive=model.video_score(fake,negative.get('box')) if diagnostic['face_changed'] else None
                records.append(dict(source=pair['source'],target=pair['target'],partition='calibration' if part==0 else 'test',swap_accepted=diagnostic['face_changed'],real=negative,fake=positive))
            except FaceDetectionError as error:records.append(dict(source=pair['source'],target=pair['target'],error=str(error)))
            finally:
                if session:session.close()
            if (index+1)%20==0:print(f'Independent face domain evaluation {index+1}/{len(pairs)}',flush=True)
            (out/'face_holdout_progress.json').write_text(json.dumps(dict(processed=index+1,total=len(pairs),elapsed_seconds=time.time()-started)))
        valid=[r for r in records if r.get('real',{}).get('score') is not None and r.get('fake') and r['fake'].get('score') is not None]
        calibration=[r for r in valid if r['partition']=='calibration'];test=[r for r in valid if r['partition']=='test']
        threshold=max([r['real']['score'] for r in calibration],default=1)+1e-6
        real=np.array([r['real']['score'] for r in test]);fake=np.array([r['fake']['score'] for r in test])
        result=dict(scope='Same previously evaluated CelebA photo pool. Source and target identities partitioned together into disjoint calibration/test groups; cross-group pairs excluded. Not an independent video benchmark. JPEG85 matched for both labels; crop derived from original target for both. No voice validation or live promotion.',
          calibration_pairs=len(calibration),test_pairs=len(test),threshold_from_calibration_negatives=threshold,
          test_false_positives=int((real>=threshold).sum()),test_true_positives=int((fake>=threshold).sum()),
          test_auc=float(((fake[:,None]>real[None,:])+.5*(fake[:,None]==real[None,:])).mean()) if len(test) else None,
          elapsed_seconds=time.time()-started,records=records)
        (out/'face_holdout.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
    finally:model.close()

if __name__=='__main__':main()
