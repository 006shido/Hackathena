"""Controlled comparison of legacy and orientation-aware model inputs."""
import sys,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image,ImageDraw
from ml.experiments.phase6g_2_identity_margin.train import MultiScaleCorrespondenceFaceSwapModel6G1,CelebAPairedDataset,RealFacePreprocessor,prepare_validation_pairs
from ml.inference.face_correspondence import preprocess_single_face
from ml.training.robust_correspondence import robust_warp

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--pairs', type=int, default=100)
    parser.add_argument('--smooth-fallback', action='store_true')
    args=parser.parse_args()
    directory=Path(__file__).parent;output=directory/('pilot_evaluation' if args.pilot else 'correspondence_evaluation');output.mkdir(exist_ok=True)
    device=torch.device('cuda')
    model=MultiScaleCorrespondenceFaceSwapModel6G1(base_channels=64,bottleneck_channels=512,embedding_dim=512,
        fullres_channels=32,blur_kernel_size=9,blur_sigma=3,arcface_checkpoint_path=str(ROOT/'ml/models/weights/ms1mv2_iresnet50.pth')).to(device)
    state=torch.load(directory/'run/best_model.pt',map_location=device,weights_only=False)
    model.load_state_dict(state['model_state_dict']);del state;model.eval()
    pre=RealFacePreprocessor(image_size=128)
    val=CelebAPairedDataset(dataset_root=str(ROOT/'ml/data/celeba'),split='val',seed=42)
    pairs=prepare_validation_pairs(val,pre,args.pairs,20261005)
    assert len(pairs)==args.pairs
    if args.pilot:
        pilot=torch.load(directory/'robust_pilot/latest_model.pt',map_location=device,weights_only=False)['model_state_dict']
        original={k:v.cpu().clone() for k,v in model.state_dict().items()}
    def u8(t):return ((t.squeeze(0).cpu().permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype(np.uint8)
    records=[];rows=[]
    with torch.no_grad():
        for index,p in enumerate(pairs):
            si,_,_,_,s_dense,_=preprocess_single_face(pre,str(ROOT/'ml/data/celeba/img_align_celeba'/p['src_name']))
            aligned,confidence,details=robust_warp(si,s_dense,p['target_dense'],smooth_fallback=args.smooth_fallback)
            record=dict(pair=index+1,source=p['src_name'],target=p['tgt_name'],geometry=details)
            images=[]
            src,tgt=p['source_tensor'].to(device),p['target_tensor'].to(device)
            zs,zt=model.arcface(src),model.arcface(tgt)
            for name,aln,conf in [('legacy',p['aligned_source'],p['confidence_map']),('robust',aligned,confidence)]:
                if args.pilot: model.load_state_dict(original if name=='legacy' else pilot)
                result=model(i_source=src,i_target=tgt,l_target=p['target_l_map'].to(device),aligned_source=aln.to(device),confidence_map=conf.to(device))
                comp=result['i_composite'];zc=model.arcface(comp);zw=model.arcface(aln.to(device))
                lm=None
                try:
                    det=pre.detect_landmarks(Image.fromarray(u8(comp)))
                    lm=float(np.linalg.norm(det.key_landmarks_5pts-p['target_5pts'],axis=1).mean())
                except Exception:pass
                record[name]=dict(A=float(F.cosine_similarity(zs,zc)),B=float(F.cosine_similarity(zt,zc)),
                    warp_source=float(F.cosine_similarity(zs,zw)),landmark_error=lm)
                images.append(u8(comp))
            records.append(record)
            if index<24:rows.append(np.concatenate([u8(src),u8(tgt),u8(p['aligned_source']),u8(aligned),*images],axis=1))
            if (index+1)%25==0:print(f'Evaluated {index+1}/{len(pairs)}',flush=True)
    aggregate={}
    for name in ['legacy','robust']:
        subset=[r[name] for r in records]
        aggregate[name]=dict(A=float(np.mean([r['A'] for r in subset])),B=float(np.mean([r['B'] for r in subset])),
            source_wins=100*float(np.mean([r['A']>r['B'] for r in subset])),warp_source=float(np.mean([r['warp_source'] for r in subset])),
            redetection=100*float(np.mean([r['landmark_error'] is not None for r in subset])),
            landmark_error=float(np.mean([r['landmark_error'] for r in subset if r['landmark_error'] is not None])))
    (output/'results.json').write_text(json.dumps(dict(aggregate=aggregate,pairs=records),indent=2))
    for i in range(4):Image.fromarray(np.concatenate(rows[i*6:(i+1)*6],axis=0)).save(output/f'comparison_{i+1:02d}.png')
    print(json.dumps(aggregate),flush=True)

if __name__=='__main__':main()
