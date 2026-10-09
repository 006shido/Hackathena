"""Test candidate on natural video and benign blur; no live promotion."""
import argparse,json
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image,ImageFilter
from ml.detection.pretrained import PretrainedDetectors

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--robust',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[2];model=PretrainedDetectors();candidate=root/('ml/detection/swap_domain_robust' if args.robust else 'ml/detection/swap_domain_finetune')
    model.face.load_state_dict(torch.load(candidate/'best.pt',map_location='cpu',weights_only=True),strict=True)
    threshold=json.loads((candidate/'results.json').read_text())['threshold'];records=[]
    try:
        for video in ['current_baseline_occlusion_video','guarded_david_video']:
            folder=root/'ml/experiments/reference_backend'/video
            metadata=json.loads((folder/'results.json').read_text());capture=cv2.VideoCapture(str(folder/'comparison.mp4'))
            for index in range(0,metadata['frames'],12):
                capture.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=capture.read()
                if not ok:raise RuntimeError('Video frame missing; evaluation incomplete.')
                frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB);width=frame.shape[1]//2
                real=Image.fromarray(frame[:,:width]);fake=Image.fromarray(frame[:,width:])
                negative=model.video_score(real);record=metadata['records'][index]
                changed=record.get('face_changed',False)
                positive=model.video_score(fake) if changed else None
                records.append(dict(video=video,index=index,real=negative,swapped=positive,swap_applied=changed))
            capture.release()
        pairs=[r for r in json.loads((root/'ml/detection/swap_domain/pairs.json').read_text()) if r['accepted'] and r['partition']=='test'][:50]
        blur=[]
        for row in pairs:
            image=Image.open(root/'ml/data/celeba/img_align_celeba'/row['target']).convert('RGB')
            for radius in [0,1,2]:
                result=model.video_score(image.filter(ImageFilter.GaussianBlur(radius)))
                blur.append(dict(file=row['target'],radius=radius,result=result))
        real=[r['real']['score'] for r in records if r['real']['score'] is not None];fake=[r['swapped']['score'] for r in records if r['swapped'] and r['swapped']['score'] is not None]
        report=dict(scope='Two previously inspected natural-video clips, correlated frames sampled every12, actual receiver-like independently detected crops and encoded comparison panels. Benign blur test uses reused photo holdout. Not a blind/cross-device benchmark.',
          threshold=threshold,video_real_frames=len(real),video_false_positives=sum(s>=threshold for s in real),video_swapped_frames=len(fake),video_true_positives=sum(s>=threshold for s in fake),
          benign_blur={str(radius):dict(count=sum(r['radius']==radius and r['result']['score'] is not None for r in blur),false_positives=sum(r['radius']==radius and r['result']['score'] is not None and r['result']['score']>=threshold for r in blur)) for radius in [0,1,2]},records=records,blur_records=blur,live_promoted=False)
        (root/('ml/detection/evaluation/robust_candidate_video.json' if args.robust else 'ml/detection/evaluation/candidate_video.json')).write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if not k.endswith('records')},indent=2))
    finally:model.close()

if __name__=='__main__':main()
