"""Fine-tune actual detector weights, preserving the previous identity split.

Uses only train data for gradients and calibration data for model selection.
The previous test is reused for comparison; a later blind confirmation is needed.
"""
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from ml.detection.pretrained import PretrainedDetectors
from ml.detection.train_swap_domain import encode,auc
from ml.experiments.reference_backend.opencv_reference import OpenCVReference
from ml.experiments.reference_backend.frame_session import ReferenceFrameSession
from ml.models.face_swap_model import ArcFaceIdentityExtractor
from ml.training.face_preprocessing import RealFacePreprocessor

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--robust',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[2];out=root/('ml/detection/swap_domain_robust' if args.robust else 'ml/detection/swap_domain_finetune');out.mkdir(exist_ok=True)
    if (out/'best.pt').exists():raise RuntimeError('Preserve existing trained detector; use a new experiment for further tuning.')
    rows=[r for r in json.loads((root/'ml/detection/swap_domain/pairs.json').read_text()) if r['accepted']]
    pairs=len(rows);cache_path=(root/'ml/detection/swap_domain_finetune/crops.npy') if args.robust else out/'crops.npy';model=PretrainedDetectors();started=time.time()
    if not cache_path.exists():
        images=np.lib.format.open_memmap(cache_path,mode='w+',dtype=np.uint8,shape=(pairs*2,256,256,3))
        engine=SimpleNamespace(reference=OpenCVReference(gpu=True),preprocessor=RealFacePreprocessor(image_size=128),identity_model=ArcFaceIdentityExtractor(str(root/'ml/models/weights/ms1mv2_iresnet50.pth')).cuda().eval(),identity_guard=True)
        engine.reference.warmup()
        for index,row in enumerate(rows):
            source=Image.open(root/'ml/data/celeba/img_align_celeba'/row['source']).convert('RGB');target=Image.open(root/'ml/data/celeba/img_align_celeba'/row['target']).convert('RGB')
            session=ReferenceFrameSession(engine,source)
            try:
                output,diagnostic=session.process(target,0)
                if not diagnostic['face_changed']:raise RuntimeError('Previously accepted swap now rejected; incomplete cache is not reusable.')
                box=model.face_box(target)
                for label,image in enumerate([target,Image.fromarray(output)]):
                    images[index*2+label]=np.asarray(encode(image,row['quality']).crop(box).resize((256,256),Image.Resampling.BILINEAR))
            finally:session.close()
            if (index+1)%50==0:print(f'Detector training image cache {index+1}/{pairs}',flush=True)
            (out/'status.json').write_text(json.dumps(dict(stage='image cache',processed=index+1,total=pairs)))
        images.flush();del images,engine;torch.cuda.empty_cache()
        (out/'cache_complete.json').write_text(json.dumps(dict(pairs=pairs)))
    complete=(root/'ml/detection/swap_domain_finetune/cache_complete.json') if args.robust else out/'cache_complete.json'
    if not complete.exists():raise RuntimeError('Cache was interrupted; do not train from incomplete data.')
    images=np.load(cache_path,mmap_mode='r');labels=np.tile([0,1],pairs)
    split={name:np.array([i*2+j for i,r in enumerate(rows) if r['partition']==name for j in (0,1)]) for name in ['train','calibration','test']}
    def batch(indices):
        x=torch.from_numpy(np.array(images[indices],copy=True)).permute(0,3,1,2).cuda().float()/127.5-1
        return x,torch.from_numpy(labels[indices]).cuda()
    def scores(name):
        model.face.eval();results=[]
        with torch.inference_mode():
            for start in range(0,len(split[name]),8):
                x,_=batch(split[name][start:start+8])
                with torch.autocast('cuda',dtype=torch.bfloat16):logits,_=model.face(x)
                results.extend(logits.float().softmax(1)[:,1].cpu().tolist())
        return np.array(results)
    torch.manual_seed(20261006);rng=np.random.default_rng(20261006)
    model.face.requires_grad_(True);optimizer=torch.optim.AdamW(model.face.parameters(),lr=1e-5,weight_decay=.001)
    best_loss=float('inf');history=[]
    for epoch in range(6):
        model.face.train()
        for layer in model.face.modules():
            if isinstance(layer,torch.nn.modules.batchnorm._BatchNorm):layer.eval()
        indices=split['train'].copy();rng.shuffle(indices);losses=[]
        for start in range(0,len(indices),8):
            x,y=batch(indices[start:start+8]);optimizer.zero_grad(set_to_none=True)
            if args.robust:
                from torchvision.transforms.functional import gaussian_blur
                sigma=float(rng.uniform(.1,4.0))
                if rng.random()<.75:x=gaussian_blur(x,[21,21],[sigma,sigma])
                if rng.random()<.5:
                    size=int(rng.integers(96,225));x=torch.nn.functional.interpolate(torch.nn.functional.interpolate(x,size=(size,size),mode='bilinear',align_corners=False),size=(256,256),mode='bilinear',align_corners=False)
                if rng.random()<.5:x=x.flip(3)
            with torch.autocast('cuda',dtype=torch.bfloat16):logits,_=model.face(x)
            loss=torch.nn.functional.cross_entropy(logits.float(),y);loss.backward();torch.nn.utils.clip_grad_norm_(model.face.parameters(),1);optimizer.step();losses.append(float(loss.detach()))
        cp=scores('calibration');clabel=labels[split['calibration']]
        cal_loss=float(-(clabel*np.log(cp.clip(1e-7,1-1e-7))+(1-clabel)*np.log((1-cp).clip(1e-7,1-1e-7))).mean())
        metric=dict(epoch=epoch+1,train_loss=float(np.mean(losses)),calibration_loss=cal_loss,calibration_auc=auc(cp[clabel==0],cp[clabel==1]))
        history.append(metric);print(json.dumps(metric),flush=True)
        if cal_loss<best_loss:
            best_loss=cal_loss;torch.save({k:v.detach().cpu() for k,v in model.face.state_dict().items()},out/'best.pt')
        (out/'status.json').write_text(json.dumps(dict(stage='training',epoch=epoch+1,total_epochs=6,elapsed_seconds=time.time()-started)))
    model.face.load_state_dict(torch.load(out/'best.pt',map_location='cpu',weights_only=True),strict=True)
    cp=scores('calibration');tp=scores('test');clabel=labels[split['calibration']];tlabel=labels[split['test']]
    threshold=float(cp[clabel==0].max()+1e-6);real=tp[tlabel==0];fake=tp[tlabel==1]
    report=dict(scope='Fine-tuned Xception on this swapper/photo domain. Same identity-disjoint test as linear-head experiment, reused comparison; blind/video confirmation still required. No audio validation or live promotion.',benign_degradation_training=args.robust,threshold=threshold,
        test_pairs=len(real),test_auc=auc(real,fake),test_false_positives=int((real>=threshold).sum()),test_true_positives=int((fake>=threshold).sum()),history=history,elapsed_seconds=time.time()-started,live_promoted=False)
    (out/'results.json').write_text(json.dumps(report,indent=2));(out/'status.json').write_text(json.dumps(dict(stage='complete')));print(json.dumps(report,indent=2));model.close()

if __name__=='__main__':main()
