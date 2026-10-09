"""Oracle-mask verification research with disjoint CelebA validation identities.

Tests different-photo same-ID positives and different-ID negatives. Synthetic
occluder geometry is known exactly; this does not validate a predicted mask.
"""
import hashlib,json,random
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ml.models.face_swap_model import ArcFaceIdentityExtractor
from ml.training.face_preprocessing import RealFacePreprocessor,align_face_similarity,normalize_image_tensor,FaceDetectionError

ROOT=Path(__file__).resolve().parents[3]
@torch.inference_mode()
def main():
    weights=ROOT/'ml/models/weights/ms1mv2_iresnet50.pth'
    if hashlib.sha256(weights.read_bytes()).hexdigest()!='2b75b93c48b01c78a4263f7295ab2dbf84f85190c51be45b92c4c0b0aedceaa3':raise ValueError('Unexpected weights')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=ArcFaceIdentityExtractor(str(weights)).cuda().eval();detector=RealFacePreprocessor(image_size=128)
    identities=defaultdict(list)
    for line in (ROOT/'ml/data/celeba/identity_CelebA.txt').read_text().splitlines():
        name,identity=line.split();identities[int(identity)].append(name)
    all_ids=sorted(identities);random.Random(42).shuffle(all_ids)
    train=set(all_ids[:int(len(all_ids)*.85)]);validation=[i for i in all_ids[int(len(all_ids)*.85):] if len(identities[i])>=2]
    random.Random(20261008).shuffle(validation)
    folder=Path(__file__).parent/'occlusion_calibration';folder.mkdir(exist_ok=True)
    groups={};skipped=[]
    for split,candidates in [('calibration',validation[:160]),('evaluation',validation[160:320])]:
        examples=[]
        for identity in candidates:
            names=random.Random(identity+20261008).sample(identities[identity],2);images=[]
            try:
                for name in names:
                    image=Image.open(ROOT/'ml/data/celeba/img_align_celeba'/name).convert('RGB');det=detector.detect_landmarks(image)
                    aligned,_,_=align_face_similarity(image,det.key_landmarks_5pts,128);images.append(np.asarray(aligned,dtype=np.uint8))
            except FaceDetectionError as error:
                skipped.append(dict(split=split,identity=identity,reason=str(error)));continue
            examples.append(dict(identity=identity,names=names,images=images))
            if len(examples)%20==0:print(f'Aligned {split} {len(examples)}/60',flush=True)
            if len(examples)==60:break
        if len(examples)!=60:raise RuntimeError('Insufficient valid identity pairs')
        groups[split]=examples
    calibration_ids={e['identity'] for e in groups['calibration']};evaluation_ids={e['identity'] for e in groups['evaluation']}
    assert not calibration_ids&evaluation_ids and not (calibration_ids|evaluation_ids)&train
    masks={}
    for name in ['clean','lower35','lower50','lower65','left35','right35','center35']:
        mask=np.ones((128,128),bool)
        if name.startswith('lower'):mask[round(128*(1-int(name[5:])/100)):]=False
        elif name=='left35':mask[:,:45]=False
        elif name=='right35':mask[:,83:]=False
        elif name=='center35':mask[:,42:87]=False
        masks[name]=mask
    def embed(images):
        values=torch.from_numpy(np.stack([normalize_image_tensor(im) for im in images])).cuda()
        return F.normalize(model(values),dim=1)
    records=[];rng=np.random.default_rng(20261008)
    for split,examples in groups.items():
        for index,example in enumerate(examples):
            source,positive=example['images'];other=examples[(index+1)%len(examples)];negative=other['images'][1]
            foreground=rng.integers(0,256,(128,128,3),dtype=np.uint8)
            batch=[]
            for mask in masks.values():
                # Whole-crop comparison: clean enrollment vs object-covered outputs.
                batch.extend([source,np.where(mask[...,None],positive,foreground),np.where(mask[...,None],negative,foreground)])
                # Oracle conditioning: remove the same region from all three photos.
                batch.extend([np.where(mask[...,None],im,127).astype(np.uint8) for im in (source,positive,negative)])
            z=embed(batch)
            for offset,name in enumerate(masks):
                for method,start in [('whole',offset*6),('oracle',offset*6+3)]:
                    records.append(dict(split=split,identity=example['identity'],source=example['names'][0],positive=example['names'][1],negative_identity=other['identity'],negative=other['names'][1],mask=name,method=method,positive_score=float(z[start]@z[start+1]),negative_score=float(z[start]@z[start+2])))
            if (index+1)%20==0:print(f'Scored {split} {index+1}/60',flush=True)
    summary=[]
    for method in ('whole','oracle'):
        for name in masks:
            cal=[r for r in records if r['split']=='calibration' and r['method']==method and r['mask']==name]
            threshold=max(r['negative_score'] for r in cal)+1e-6
            ev=[r for r in records if r['split']=='evaluation' and r['method']==method and r['mask']==name]
            summary.append(dict(method=method,mask=name,threshold=threshold,calibration_count=len(cal),evaluation_count=len(ev),evaluation_true_accept_rate=float(np.mean([r['positive_score']>=threshold for r in ev])),evaluation_false_accept_rate=float(np.mean([r['negative_score']>=threshold for r in ev]))))
    report=dict(scope='120 identities from original validation split, disjoint calibration/evaluation groups, different-photo positives and cyclic cross-ID negatives. Synthetic foreground and oracle visibility masks. Per-mask thresholds chosen above maximum calibration negative; no live threshold changes. Small negative sample is not deployment false-accept calibration.',calibration_ids=sorted(calibration_ids),evaluation_ids=sorted(evaluation_ids),train_identity_overlap=0,skipped=skipped,summary=summary,records=records)
    (folder/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(summary))
if __name__=='__main__':main()
