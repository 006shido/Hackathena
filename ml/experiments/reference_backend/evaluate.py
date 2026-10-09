"""Independent ArcFace and landmark comparison in the same aligned frame."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import cv2,numpy as np,torch
import torch.nn.functional as F
from PIL import Image,ImageDraw
from ml.experiments.phase6g_2_identity_margin.train import (
    MultiScaleCorrespondenceFaceSwapModel6G1,CelebAPairedDataset,RealFacePreprocessor,prepare_validation_pairs,
)
from ml.experiments.reference_backend.opencv_reference import OpenCVReference
from ml.inference.frame_pipeline import restore_face


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pairs',type=int,default=100)
    parser.add_argument('--seed',type=int,default=20261005)
    parser.add_argument('--gpu',action='store_true')
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'screen')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    reference=OpenCVReference(gpu=args.gpu)
    device=torch.device('cuda')
    model=MultiScaleCorrespondenceFaceSwapModel6G1(base_channels=64,bottleneck_channels=512,embedding_dim=512,
        fullres_channels=32,blur_kernel_size=9,blur_sigma=3,arcface_checkpoint_path=str(ROOT/'ml/models/weights/ms1mv2_iresnet50.pth')).to(device)
    state=torch.load(ROOT/'ml/experiments/phase6g_3_full_dataset/run/best_model.pt',map_location=device,weights_only=False)
    model.load_state_dict(state['model_state_dict']);del state;model.eval()
    pre=RealFacePreprocessor(image_size=128)
    val=CelebAPairedDataset(dataset_root=str(ROOT/'ml/data/celeba'),split='val',seed=42)
    pairs=prepare_validation_pairs(val,pre,args.pairs,args.seed)
    assert len(pairs)==args.pairs
    def u8(t):return ((t.squeeze(0).cpu().permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype(np.uint8)
    records=[];rows=[]
    with torch.inference_mode():
        for index,pair in enumerate(pairs):
            src,tgt=pair['source_tensor'].to(device),pair['target_tensor'].to(device)
            zs,zt=model.arcface(src),model.arcface(tgt)
            original=model(i_source=src,i_target=tgt,l_target=pair['target_l_map'].to(device),
                aligned_source=pair['aligned_source'].to(device),confidence_map=pair['confidence_map'].to(device))['i_composite']
            source,target=u8(src),u8(tgt)
            # Reuse landmarks projected from the successfully detected source.
            # Re-detecting a small aligned crop can fail even for a valid pair.
            latent=reference.source_identity(source,pair['source_5pts'])
            start=time.perf_counter()
            swap,affine=reference.swap(target,pair['target_5pts'],latent)
            latency=1000*(time.perf_counter()-start)
            alpha=cv2.warpAffine(pair['target_mask'].squeeze().numpy(),affine,(128,128))
            fixed,_=restore_face(target,swap,alpha,affine)
            converted=torch.from_numpy(fixed.transpose(2,0,1).copy()).float().unsqueeze(0).to(device)/127.5-1
            record=dict(source=pair['src_name'],target=pair['tgt_name'],reference_swap_ms=latency)
            for name,tensor in [('selected',original),('reference',converted)]:
                rgb=u8(tensor);zc=model.arcface(tensor);lm=None
                try:
                    detection=pre.detect_landmarks(Image.fromarray(rgb))
                    lm=float(np.linalg.norm(detection.key_landmarks_5pts-pair['target_5pts'],axis=1).mean())
                except Exception:pass
                record[name]=dict(A=float(F.cosine_similarity(zs,zc)),B=float(F.cosine_similarity(zt,zc)),landmark_error=lm,
                    sharpness=float(cv2.Laplacian(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),cv2.CV_64F).var()))
            records.append(record)
            if index<24:rows.append([source,target,u8(original),fixed])
            if (index+1)%25==0:print(f'Evaluated {index+1}/{len(pairs)}',flush=True)
    aggregate={}
    for name in ['selected','reference']:
        values=[record[name] for record in records]
        aggregate[name]=dict(A=float(np.mean([r['A'] for r in values])),B=float(np.mean([r['B'] for r in values])),
            source_wins=100*float(np.mean([r['A']>r['B'] for r in values])),
            redetection=100*float(np.mean([r['landmark_error'] is not None for r in values])),
            landmark_error=float(np.mean([r['landmark_error'] for r in values if r['landmark_error'] is not None])),
            sharpness=float(np.mean([r['sharpness'] for r in values])))
    (args.output/'results.json').write_text(json.dumps(dict(protocol='cached-source-landmarks-v2',backend='torch-cuda-fp32' if args.gpu else 'opencv-cpu',seed=args.seed,count=len(pairs),aggregate=aggregate,
        reference_swap_ms_median=float(np.median([r['reference_swap_ms'] for r in records])),pairs=records),indent=2))
    for group in range((len(rows)+5)//6):
        batch=rows[group*6:(group+1)*6];canvas=Image.new('RGB',(512,len(batch)*150),'white');draw=ImageDraw.Draw(canvas)
        for row,tiles in enumerate(batch):
            for col,(tile,label) in enumerate(zip(tiles,['SOURCE','TARGET','SELECTED','REFERENCE'])):
                canvas.paste(Image.fromarray(tile),(col*128,row*150+22));draw.text((col*128+2,row*150+3),label,fill='black')
        canvas.save(args.output/f'comparison_{group+1:02d}.png')
    print(json.dumps(aggregate),flush=True)


if __name__=='__main__':main()
