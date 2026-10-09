"""Reconstruct experiment waveforms for matched temporal feature training."""
import json
from pathlib import Path
import numpy as np
from scipy.signal import butter,sosfilt
from ml.detection.train_call_audio import read,opus,effect,ROOT,OUT
from ml.detection.effect_model import effect_features

def main():
    records=json.loads((OUT/'results.json').read_text())['records'];vectors=[];cache={}
    for index,row in enumerate(records):
        file=row['file'];variant=row['variant']
        if file.endswith('.f32'):audio=np.fromfile(OUT/'browser_received'/file,dtype='<f4')
        else:
            if file not in cache:cache[file]=read(OUT/'natural'/file if file.endswith('.flac') else ROOT/'ml/detection/audio_samples'/file)
            x=cache[file]
            if variant=='raw':audio=x
            elif variant=='opus':audio=opus(x)
            elif variant=='benign_gain':audio=opus(x*.4)
            elif variant=='benign_lowpass':audio=opus(sosfilt(butter(2,3400,fs=16000,output='sos'),x).astype(np.float32))
            else:audio=opus(np.clip(effect(x,variant),-1,1))
        vectors.append(effect_features(np.clip(audio,-1,1)))
        if index%200==0:print(f'Temporal features {index+1}/{len(records)}',flush=True)
    external=[effect_features(np.clip(read(ROOT/f'ml/experiments/reference_backend/browser_speech_call/{name}.wav'),-1,1)) for name in ['clean','robotic-vocoder','deep-pitch-neural','synthetic-clone']]
    np.savez(OUT/'effect_features.npz',X=np.stack(vectors));np.savez(OUT/'effect_external_features.npz',X=np.stack(external));print('Temporal feature preparation complete.',flush=True)

if __name__=='__main__':main()
