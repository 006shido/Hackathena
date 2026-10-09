"""Compare optional refinement to the selected model on the same 300 pairs."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from ml.experiments.phase6g_2_identity_margin.train import MultiScaleCorrespondenceFaceSwapModel6G1,CelebAPairedDataset,RealFacePreprocessor,prepare_validation_pairs
from ml.inference.face_refinement import refine_face

def main():
    directory=Path(__file__).parent
    output=directory/'refinement_evaluation'
    output.mkdir(exist_ok=True)
    device=torch.device('cuda')
    model=MultiScaleCorrespondenceFaceSwapModel6G1(base_channels=64,bottleneck_channels=512,embedding_dim=512,
        fullres_channels=32,blur_kernel_size=9,blur_sigma=3,arcface_checkpoint_path=str(ROOT/'ml/models/weights/ms1mv2_iresnet50.pth')).to(device)
    state=torch.load(directory/'run/best_model.pt',map_location=device,weights_only=False)
    model.load_state_dict(state['model_state_dict']);del state
    model.eval()
    pre=RealFacePreprocessor(image_size=128)
    val=CelebAPairedDataset(dataset_root=str(ROOT/'ml/data/celeba'),split='val',seed=42)
    pairs=prepare_validation_pairs(val,pre,300,20261005)
    assert len(pairs)==300
    recorded=json.loads((directory/'evaluation/results.json').read_text())['pairs']
    assert recorded==[dict(source=p['src_name'],target=p['tgt_name']) for p in pairs]
    def u8(t):return ((t.squeeze(0).cpu().permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype(np.uint8)
    records=[];rows=[]
    with torch.no_grad():
        for index,p in enumerate(pairs):
            src,tgt=p['source_tensor'].to(device),p['target_tensor'].to(device)
            result=model(i_source=src,i_target=tgt,l_target=p['target_l_map'].to(device),aligned_source=p['aligned_source'].to(device),confidence_map=p['confidence_map'].to(device))
            swap,original=u8(result['i_swap']),u8(result['i_composite'])
            target=u8(tgt)
            zs=model.arcface(src)
            def identity_score(image):
                candidate=torch.from_numpy(image.transpose(2,0,1).copy()).float().unsqueeze(0).to(device)/127.5-1
                return F.cosine_similarity(zs,model.arcface(candidate)).item()
            fixed,diagnostic=refine_face(swap,target,result['m_pred'].squeeze().cpu().numpy(),p['target_mask'].squeeze().numpy(),p['target_dense'],pre,identity_score)
            tensor=torch.from_numpy(fixed.transpose(2,0,1).copy()).float().unsqueeze(0).to(device)/127.5-1
            zs,zt,zf=model.arcface(src),model.arcface(tgt),model.arcface(tensor)
            a=float(F.cosine_similarity(zs,zf));b=float(F.cosine_similarity(zt,zf))
            lm=None
            try:
                det=pre.detect_landmarks(Image.fromarray(fixed))
                lm=float(np.linalg.norm(det.key_landmarks_5pts-p['target_5pts'],axis=1).mean())
            except Exception:pass
            records.append(dict(pair=index+1,A=a,B=b,landmark_error=lm,sharpness=float(cv2.Laplacian(cv2.cvtColor(fixed,cv2.COLOR_RGB2GRAY),cv2.CV_64F).var()),**diagnostic))
            if index<24: rows.append(np.concatenate([u8(src),target,original,fixed],axis=1))
            if (index+1)%50==0:print(f'Evaluated {index+1}/300',flush=True)
    aggregate=dict(A=float(np.mean([r['A'] for r in records])),B=float(np.mean([r['B'] for r in records])),
        source_wins=100*float(np.mean([r['A']>r['B'] for r in records])),redetection=100*float(np.mean([r['landmark_error'] is not None for r in records])),
        landmark_error=float(np.mean([r['landmark_error'] for r in records if r['landmark_error'] is not None])),
        sharpness=float(np.mean([r['sharpness'] for r in records])),accepted=sum(r['accepted'] for r in records))
    (output/'results.json').write_text(json.dumps(dict(aggregate=aggregate,pairs=records),indent=2))
    for i in range(4):Image.fromarray(np.concatenate(rows[i*6:(i+1)*6],axis=0)).save(output/f'comparison_{i+1:02d}.png')
    print(json.dumps(aggregate),flush=True)

if __name__=='__main__':main()
