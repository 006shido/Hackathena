"""Controlled foreground nuisance test; never calibrates live guard thresholds.

Positive outputs are the actual selected source photo, with a shared synthetic
foreground over source/target outputs. Negative outputs retain target identity.
This isolates occlusion effects; it is not generated-swap quality evaluation.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ml.models.face_swap_model import ArcFaceIdentityExtractor
from ml.training.face_preprocessing import RealFacePreprocessor,align_face_similarity,normalize_image_tensor

ROOT=Path(__file__).resolve().parents[3]
@torch.inference_mode()
def main():
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    weights=ROOT/'ml/models/weights/ms1mv2_iresnet50.pth'
    if hashlib.sha256(weights.read_bytes()).hexdigest()!='2b75b93c48b01c78a4263f7295ab2dbf84f85190c51be45b92c4c0b0aedceaa3':raise ValueError('ArcFace checksum mismatch')
    model=ArcFaceIdentityExtractor(str(weights)).cuda().eval();pre=RealFacePreprocessor(image_size=128)
    folder=Path(__file__).parent/'occluded_identity_control';folder.mkdir(exist_ok=True)
    pairs=json.loads((Path(__file__).parent/'holdout/results.json').read_text())['pairs'][:20]
    records=[];rng=np.random.default_rng(20261007)
    def embed(images):
        batch=torch.from_numpy(np.stack([normalize_image_tensor(im) for im in images])).cuda()
        return F.normalize(model(batch),dim=1)
    for index,pair in enumerate(pairs):
        images=[]
        for name in (pair['source'],pair['target']):
            image=Image.open(ROOT/'ml/data/celeba/img_align_celeba'/name).convert('RGB')
            det=pre.detect_landmarks(image)
            aligned,_,_=align_face_similarity(image,det.key_landmarks_5pts,128)
            images.append(np.asarray(aligned,dtype=np.uint8))
        source,target=images;texture=rng.integers(0,256,(128,128,3),dtype=np.uint8)
        clean=embed(images)
        for fraction in (0,.2,.35,.5,.65):
            start=round(128*(1-fraction));positive=source.copy();negative=target.copy()
            positive[start:]=texture[start:];negative[start:]=texture[start:]
            foreground=embed([positive,negative])
            # Current guard compares unoccluded source with occluded output and target.
            a=float(clean[0]@foreground[0]);b=float(foreground[1]@foreground[0])
            neg_a=float(clean[0]@foreground[1]);neg_b=1.0
            records.append(dict(pair=index,source=pair['source'],target=pair['target'],occluded_fraction=fraction,
                known_source_similarity=a,shared_foreground_target_similarity=b,
                known_source_accepted=a>=.3 and a-b>=.05,
                known_target_accepted=neg_a>=.3 and neg_a-neg_b>=.05))
            if index==0:
                Image.fromarray(np.concatenate([source,target,positive,negative],axis=1)).save(folder/f'fraction_{fraction}.png')
        print(f'Controlled pair {index+1}/{len(pairs)}',flush=True)
    aggregate=[]
    for fraction in (0,.2,.35,.5,.65):
        group=[r for r in records if r['occluded_fraction']==fraction]
        aggregate.append(dict(occluded_fraction=fraction,count=len(group),known_source_acceptance=float(np.mean([r['known_source_accepted'] for r in group])),known_target_acceptance=float(np.mean([r['known_target_accepted'] for r in group])),mean_source_similarity=float(np.mean([r['known_source_similarity'] for r in group])),mean_target_similarity=float(np.mean([r['shared_foreground_target_similarity'] for r in group]))))
    report=dict(scope='20 held-out cross-ID photo pairs with identical synthetic textured foreground. Positives are true source photos, negatives true target photos; not generated-swap quality, threshold calibration or real-object segmentation.',aggregate=aggregate,records=records)
    (folder/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(aggregate))
if __name__=='__main__':main()
