"""Check genuine-recording boundary effects; no threshold/model promotion."""
import json,math
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly,correlate
from ml.detection.pretrained import PretrainedDetectors

def main():
    root=Path(__file__).resolve().parents[2];model=PretrainedDetectors();records=[]
    try:
        for name in ['clean','robotic-vocoder','deep-pitch-neural','synthetic-clone']:
            rate,x=wavfile.read(root/f'ml/experiments/reference_backend/browser_speech_call/{name}.wav')
            if x.ndim==2:x=x.mean(1)
            if np.issubdtype(x.dtype,np.integer):x=x.astype(np.float32)/max(abs(np.iinfo(x.dtype).min),np.iinfo(x.dtype).max)
            window=int(rate*.02);chunks=[float(np.sqrt(np.mean(x[i:i+window]**2))) for i in range(0,len(x),window)]
            gate=max(.005,max(chunks)*.03);active=np.where(np.array(chunks)>gate)[0]
            if len(active)==0:continue
            start=max(0,int(active[0]*window-.1*rate));end=min(len(x),int((active[-1]+1)*window+.1*rate));trimmed=x[start:end]
            if len(trimmed)<rate*3:trimmed=np.tile(trimmed,math.ceil(rate*3/len(trimmed)))[:rate*3]
            records.append(dict(name=name,trim_start_seconds=start/rate,trim_end_seconds=end/rate,score=model.audio_score(trimmed[:rate*6],rate)))
        (root/'ml/detection/evaluation/audio_boundaries.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
    finally:model.close()

if __name__=='__main__':main()
