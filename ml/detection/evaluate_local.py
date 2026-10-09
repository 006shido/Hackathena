"""Domain-shift pilot, not benchmark validation or threshold calibration."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.io import wavfile
from ml.detection.pretrained import PretrainedDetectors

def main():
    root=Path(__file__).resolve().parents[2]
    model=PretrainedDetectors()
    records=[]
    try:
        for index in range(12):
            panel=Image.open(root/f'ml/experiments/reference_backend/live_holdout/pair_{index}.png').convert('RGB')
            width=panel.width//2
            real=panel.crop((0,0,width,panel.height)); fake=panel.crop((width,0,width*2,panel.height))
            first=model.video_score(real)
            second=model.video_score(fake,first.get('box'))
            records.append(dict(index=index,real=first,swapped=second,pixels_changed=bool(np.any(np.asarray(real)!=np.asarray(fake)))))
            print(f'Face pilot {index+1}/12',flush=True)
        audio=[]
        for name in ['clean','robotic-vocoder','deep-pitch-neural','synthetic-clone']:
            file=root/f'ml/experiments/reference_backend/browser_speech_call/{name}.wav'
            rate,pcm=wavfile.read(file)
            if np.issubdtype(pcm.dtype,np.integer):pcm=pcm.astype(np.float32)/max(abs(np.iinfo(pcm.dtype).min),np.iinfo(pcm.dtype).max)
            if pcm.ndim==2:pcm=pcm.mean(1)
            audio.append(dict(name=name,ground_truth='recorded natural speech' if name=='clean' else 'DSP-modified speech, not neural TTS',result=model.audio_score(pcm,rate)))
        valid=[r for r in records if r['pixels_changed'] and r['real']['score'] is not None and r['swapped']['score'] is not None]
        real=np.array([r['real']['score'] for r in valid]);fake=np.array([r['swapped']['score'] for r in valid])
        report=dict(scope='12 previously inspected held-out photo pairs and one speech clip with DSP variants. NOT independent benchmark validation. MediaPipe square crops differ from upstream landmark crops. No true TTS/voice-clone evaluation set.',
                    face_count=len(valid),face_real_mean=float(real.mean()) if len(real) else None,face_fake_mean=float(fake.mean()) if len(fake) else None,
                    face_auc=float(((fake[:,None]>real[None,:])+0.5*(fake[:,None]==real[None,:])).mean()) if len(real) else None,
                    false_positives_at_uncalibrated_half=int((real>=.5).sum()),true_positives_at_uncalibrated_half=int((fake>=.5).sum()),face=records,audio=audio)
        out=root/'ml/detection/evaluation';out.mkdir(exist_ok=True)
        (out/'local_pilot.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({k:v for k,v in report.items() if k!='face'},indent=2))
    finally:model.close()

if __name__=='__main__':main()
