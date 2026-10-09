"""Adapt a frozen independent Xception representation to this swapper's artifacts.

Separate identities for train/calibration/test. No sender telemetry is a feature.
This domain-specific classifier cannot establish universal deepfake detection.
"""
import argparse,hashlib,json,time
from pathlib import Path
from io import BytesIO
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from ml.detection.pretrained import PretrainedDetectors
from ml.experiments.reference_backend.opencv_reference import OpenCVReference
from ml.experiments.reference_backend.frame_session import ReferenceFrameSession
from ml.models.face_swap_model import ArcFaceIdentityExtractor
from ml.training.face_preprocessing import RealFacePreprocessor,FaceDetectionError

def auc(real,fake):
    return float(((fake[:,None]>real[None,:])+.5*(fake[:,None]==real[None,:])).mean())

def encode(image,quality):
    stream=BytesIO();image.save(stream,format='JPEG',quality=int(quality))
    stream.seek(0);return Image.open(stream).convert('RGB')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--train',type=int,default=500);parser.add_argument('--calibration',type=int,default=150);parser.add_argument('--test',type=int,default=150)
    args=parser.parse_args();root=Path(__file__).resolve().parents[2];out=root/'ml/detection/swap_domain';out.mkdir(exist_ok=True)
    if (out/'features.pt').exists() or (out/'head.pt').exists():raise RuntimeError('Existing experiment preserved. Use a new experiment path before another training run.')
    torch.manual_seed(20261006);rng=np.random.default_rng(20261006)
    identities={}
    for line in (root/'ml/data/celeba/identity_CelebA.txt').read_text().splitlines():
        name,identity=line.split();identities.setdefault(identity,[]).append(name)
    previous=json.loads((root/'ml/experiments/reference_backend/holdout/results.json').read_text())
    lookup={name:identity for identity,names in identities.items() for name in names}
    excluded={lookup[p[k]] for p in previous['pairs'] for k in ('source','target')}
    identity_ids=sorted(set(identities)-excluded,key=int);rng.shuffle(identity_ids)
    boundaries=[0,int(len(identity_ids)*.7),int(len(identity_ids)*.85),len(identity_ids)]
    partitions={name:identity_ids[boundaries[i]:boundaries[i+1]] for i,name in enumerate(('train','calibration','test'))}
    assert not set(partitions['train'])&set(partitions['test'])
    assert not set(partitions['train'])&set(partitions['calibration'])
    assert not set(partitions['calibration'])&set(partitions['test'])
    (out/'split.json').write_text(json.dumps(dict(seed=20261006,excluded_prior_evaluation_identities=sorted(excluded),identities=partitions),indent=2))
    model=PretrainedDetectors()
    weights=root/'ml/models/weights/ms1mv2_iresnet50.pth'
    if hashlib.sha256(weights.read_bytes()).hexdigest()!='2b75b93c48b01c78a4263f7295ab2dbf84f85190c51be45b92c4c0b0aedceaa3':raise ValueError('Identity weight mismatch')
    engine=SimpleNamespace(reference=OpenCVReference(gpu=True),preprocessor=RealFacePreprocessor(image_size=128),identity_model=ArcFaceIdentityExtractor(str(weights)).cuda().eval(),identity_guard=True)
    engine.reference.warmup();datasets={};records=[];started=time.time();processed=0;total=args.train+args.calibration+args.test
    try:
        for partition,count in [('train',args.train),('calibration',args.calibration),('test',args.test)]:
            features=[];labels=[];accepted=0;attempts=0;seen=set()
            while accepted<count and attempts<count*3:
                attempts+=1;chosen=rng.choice(partitions[partition],size=2,replace=False)
                source_name=str(rng.choice(identities[chosen[0]]));target_name=str(rng.choice(identities[chosen[1]]))
                if (source_name,target_name) in seen:continue
                seen.add((source_name,target_name));session=None
                try:
                    source=Image.open(root/'ml/data/celeba/img_align_celeba'/source_name).convert('RGB')
                    target=Image.open(root/'ml/data/celeba/img_align_celeba'/target_name).convert('RGB')
                    session=ReferenceFrameSession(engine,source);output,diagnostic=session.process(target,0)
                    if not diagnostic['face_changed']:
                        records.append(dict(partition=partition,source=source_name,target=target_name,accepted=False,reason='swap guard rejected'));continue
                    box=model.face_box(target);quality=int(rng.integers(70,96))
                    # Both labels get identical crop, scale and encoding. Actual generated pixels are the only class signal.
                    images=[encode(target,quality),encode(Image.fromarray(output),quality)]
                    tensors=[torch.from_numpy(np.asarray(image.crop(box).resize((256,256),Image.Resampling.BILINEAR),dtype=np.float32).copy()).permute(2,0,1)/127.5-1 for image in images]
                    with torch.inference_mode():
                        maps=model.face.features(torch.stack(tensors).cuda());pooled=torch.relu(maps).mean((2,3)).cpu()
                    features.append(pooled);labels.extend([0,1]);accepted+=1;processed+=1
                    records.append(dict(partition=partition,source=source_name,target=target_name,source_identity=chosen[0],target_identity=chosen[1],accepted=True,quality=quality))
                except (FaceDetectionError,ValueError) as error:records.append(dict(partition=partition,source=source_name,target=target_name,accepted=False,reason=str(error)))
                finally:
                    if session:session.close()
                (out/'status.json').write_text(json.dumps(dict(stage='feature extraction',processed=processed,total=total,elapsed_seconds=time.time()-started)))
                if accepted%25==0:print(f'Detector {partition}: {accepted}/{count} pairs',flush=True)
            if accepted<count:raise RuntimeError(f'Only {accepted}/{count} valid {partition} pairs; no automatic resampling across partitions.')
            datasets[partition]=(torch.cat(features),torch.tensor(labels,dtype=torch.long))
        torch.save(datasets,out/'features.pt');(out/'pairs.json').write_text(json.dumps(records,indent=2))
        x,y=datasets['train'];cx,cy=datasets['calibration'];tx,ty=datasets['test']
        # Standardization is estimated ONLY from training features.
        mean=x.mean(0);std=x.std(0).clamp_min(.05)
        x=((x-mean)/std).cuda();cx=((cx-mean)/std).cuda();tx=((tx-mean)/std).cuda();y=y.cuda();cy=cy.cuda()
        head=torch.nn.Linear(2048,2).cuda();optimizer=torch.optim.AdamW(head.parameters(),lr=.003,weight_decay=.01)
        best_loss=float('inf');best=None
        for step in range(400):
            head.train();optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(head(x),y);loss.backward();optimizer.step()
            head.eval()
            with torch.no_grad():cal_loss=float(torch.nn.functional.cross_entropy(head(cx),cy))
            if cal_loss<best_loss:best_loss=cal_loss;best={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        head.load_state_dict(best);head.eval()
        with torch.no_grad():cp=head(cx).softmax(1)[:,1].cpu().numpy();tp=head(tx).softmax(1)[:,1].cpu().numpy()
        cal_real=cp[cy.cpu().numpy()==0];threshold=float(cal_real.max()+1e-6)
        real=tp[ty.numpy()==0];fake=tp[ty.numpy()==1]
        report=dict(scope='Frozen Xception + trained linear head, specific to this reference swapper/photo domain. Newly sampled identities, excluding all previously inspected photo-pool identities. Not general video/audio deepfake validation.',
            train_pairs=args.train,calibration_pairs=args.calibration,test_pairs=args.test,threshold=threshold,test_auc=auc(real,fake),test_false_positives=int((real>=threshold).sum()),test_true_positives=int((fake>=threshold).sum()),best_calibration_loss=best_loss,elapsed_seconds=time.time()-started,
            live_promoted=False,test_real_scores=real.tolist(),test_fake_scores=fake.tolist())
        torch.save(dict(head=best,mean=mean,std=std,threshold=threshold,model='Xception swap-domain linear head',scope=report['scope']),out/'head.pt')
        (out/'results.json').write_text(json.dumps(report,indent=2));(out/'status.json').write_text(json.dumps(dict(stage='complete',processed=processed,total=total)))
        print(json.dumps({k:v for k,v in report.items() if not k.endswith('_scores')},indent=2))
    except Exception as error:
        (out/'status.json').write_text(json.dumps(dict(stage='failed',error=str(error),processed=processed,total=total)));raise
    finally:model.close()

if __name__=='__main__':main()
