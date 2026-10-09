"""Evaluate published true speech/spoof examples without using them for training."""
import hashlib,json,math
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from ml.detection.pretrained import PretrainedDetectors

def main():
    root=Path(__file__).parent;manifest=json.loads((root/'audio_samples/manifest.json').read_text());model=PretrainedDetectors();records=[];seen=set()
    try:
        for record in manifest['records']:
            if 'error' in record:continue
            if record['file'] in seen:continue
            seen.add(record['file']);path=root/'audio_samples'/record['file']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Audio sample checksum mismatch')
            rate,pcm=wavfile.read(path)
            if np.issubdtype(pcm.dtype,np.integer):pcm=pcm.astype(np.float32)/max(abs(np.iinfo(pcm.dtype).min),np.iinfo(pcm.dtype).max)
            if pcm.ndim==2:pcm=pcm.mean(1)
            duration=len(pcm)/rate
            if duration<3:pcm=np.tile(pcm,math.ceil(3*rate/len(pcm)))[:3*rate]
            if duration>6:pcm=pcm[:int(4.04*rate)]
            result=model.audio_score(pcm,rate)
            records.append(dict(file=record['file'],label=record['label'],source_url=record['url'],original_duration=duration,result=result))
            if len(records)%25==0:print(f'Audio showcase {len(records)} unique clips',flush=True)
        real=np.array([r['result']['score'] for r in records if r['label']=='real' and r['result']['score'] is not None]);fake=np.array([r['result']['score'] for r in records if r['label']=='spoof' and r['result']['score'] is not None])
        report=dict(scope='Curated official ASVspoof2019 LA evaluation showcase, not full/random corpus. Some speakers/texts shared across labels. No calibration or fine tuning on these examples.',
          real_count=len(real),spoof_count=len(fake),auc=float(((fake[:,None]>real[None,:])+.5*(fake[:,None]==real[None,:])).mean()),
          false_positives_at_uncalibrated_half=int((real>=.5).sum()),true_positives_at_uncalibrated_half=int((fake>=.5).sum()),real_mean=float(real.mean()),fake_mean=float(fake.mean()),records=records)
        (root/'evaluation/audio_showcase.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
    finally:model.close()

if __name__=='__main__':main()
