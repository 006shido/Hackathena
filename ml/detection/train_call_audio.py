"""Small speaker/method-disjoint call-domain experiment; not production certification."""
import hashlib,json,math,re,time
from fractions import Fraction
from pathlib import Path
import av,numpy as np,torch
from scipy.io import wavfile
from scipy.signal import resample_poly,stft,sosfilt,butter,sawtooth,iirpeak,lfilter
from ml.detection.pretrained import PretrainedDetectors

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'ml/detection/call_domain'
TEST={'p326','p233'};CAL={'p317','p274'}
HELD_METHODS={'sample-SS_9_16k','sample-VC_5'}

def read(path):
    if Path(path).suffix == '.flac':
        with av.open(str(path)) as container:
            frames=list(container.decode(audio=0))
        rate=frames[0].sample_rate
        if any(len(frame.layout.channels)!=1 for frame in frames):raise ValueError('Expected mono corpus audio')
        x=np.concatenate([frame.to_ndarray().reshape(-1) for frame in frames])
    else:
        rate,x=wavfile.read(path)
    if np.issubdtype(x.dtype,np.integer):x=x.astype(np.float32)/max(abs(np.iinfo(x.dtype).min),np.iinfo(x.dtype).max)
    if x.ndim>1:x=x.mean(1)
    x=resample_poly(x,16000//math.gcd(rate,16000),rate//math.gcd(rate,16000)).astype(np.float32)
    return np.tile(x,math.ceil(64600/len(x)))[:64600]

def opus(x):
    """Actual Opus encode/decode at 24 kbps, not a lowpass codec surrogate."""
    x=resample_poly(x,3,1).astype(np.float32)
    encoder=av.CodecContext.create('libopus','w');encoder.sample_rate=48000;encoder.layout='mono';encoder.format='flt';encoder.bit_rate=24000
    decoder=av.CodecContext.create('opus','r');chunks=[]
    for start in range(0,len(x)-959,960):
        frame=av.AudioFrame.from_ndarray(x[start:start+960].reshape(1,-1),format='flt',layout='mono')
        frame.sample_rate=48000;frame.pts=start;frame.time_base=Fraction(1,48000)
        for packet in encoder.encode(frame):
            chunks.extend(f.to_ndarray().reshape(-1) for f in decoder.decode(packet))
    for packet in encoder.encode(None):chunks.extend(f.to_ndarray().reshape(-1) for f in decoder.decode(packet))
    result=resample_poly(np.concatenate(chunks),1,3).astype(np.float32)
    return np.tile(result,math.ceil(64600/len(result)))[:64600]

def effect(x,name):
    """Approximate DSP augmentation; actual production call clips remain external tests."""
    if name=='deep':
        low=sosfilt(butter(2,1700,fs=16000,output='sos'),x)
        bass=sosfilt(butter(2,220,fs=16000,output='sos'),x)
        return (np.tanh((low+6*bass)*3)*.65).astype(np.float32)
    if name=='robotic':
        band=sosfilt(butter(2,[900,1900],btype='bandpass',fs=16000,output='sos'),x)
        return (x*.25+band*2.88*sawtooth(2*np.pi*160*np.arange(len(x))/16000)).astype(np.float32)
    b,a=iirpeak(1600,4,fs=16000);peak=lfilter(b,a,x)
    b,a=iirpeak(3200,3.5,fs=16000);peak2=lfilter(b,a,x)
    envelope=.35+np.sin(2*np.pi*38*np.arange(len(x))/16000)
    return (x*.35+(x+2*peak+1.5*peak2)*1.8*envelope).astype(np.float32)

@torch.inference_mode()
def features(model,x):
    x=np.clip(x,-1,1).astype(np.float32)
    hidden,logits=model.audio(torch.from_numpy(x.copy()).unsqueeze(0).to(model.device))
    _,_,z=stft(x,16000,nperseg=512,noverlap=256)
    power=np.abs(z)**2
    edges=np.geomspace(80,7600,33);freq=np.arange(power.shape[0])*16000/512
    bands=np.stack([power[(freq>=a)&(freq<b)].sum(0) for a,b in zip(edges[:-1],edges[1:])])
    bands=10*np.log10(np.maximum(bands/np.maximum(bands.sum(0),1e-10),1e-8))
    return np.concatenate([hidden[0].cpu().numpy(),logits[0].cpu().numpy(),bands.mean(1),bands.std(1)]).astype(np.float32)

def main():
    OUT.mkdir(exist_ok=True);started=time.time();torch.set_num_threads(4)
    model=PretrainedDetectors();records=[];vectors=[]
    manifest=json.loads((ROOT/'ml/detection/audio_samples/manifest.json').read_text())
    unique={r['file']:r for r in manifest['records'] if 'error' not in r}
    try:
        cache=np.load(OUT/'attempt2-features.npz')
        previous=json.loads((OUT/'attempt2-results.json').read_text())
        records=[{k:v for k,v in row.items() if k!='score'} for row in previous['records']]
        vectors=list(cache['X'])
        manifest=json.loads((OUT/'browser_received/manifest.json').read_text())
        if len(manifest['records']) != 65 or any('rms' not in row or row['rms'] < .0001 for row in manifest['records']):
            raise ValueError('Complete, non-silent browser collection is required')
        for index,record in enumerate(manifest['records']):
            audio=np.fromfile(OUT/'browser_received'/record['file'],dtype='<f4')
            if len(audio)!=64600 or not np.isfinite(audio).all() or np.sqrt(np.mean(audio**2))<.0001:raise ValueError('Invalid browser recording')
            vectors.append(features(model,audio));records.append(record)
            if index%10==0:print(f'Browser features {index+1}/{len(manifest["records"])}',flush=True)
        # Audit actual constructed records, not just the intended split rules.
        eval_speakers={speaker for row in records if row['split'] in ['test','calibration'] for speaker in row['speakers']}
        # Conversion files name both source and target speakers. Exclude either
        # participant from fitting when they occur in calibration or test.
        for row in records:
            if row['split']=='train' and set(row['speakers']) & eval_speakers:
                row['split']='excluded_identity_overlap'
        training_speakers={speaker for row in records if row['split']=='train' for speaker in row['speakers']}
        assert not training_speakers & eval_speakers
        X=np.stack(vectors);y=np.array([r['label'] for r in records]);train=np.array([r['split']=='train' for r in records])
        mean=X[train].mean(0);std=np.maximum(X[train].std(0),.01);Z=torch.tensor((X-mean)/std)
        torch.manual_seed(17);head=torch.nn.Linear(X.shape[1],1);optimizer=torch.optim.Adam(head.parameters(),lr=.015)
        labels=torch.tensor(y[train],dtype=torch.float32);weights=torch.where(labels>0,.5/(labels.mean()+1e-6),.5/(1-labels.mean()+1e-6))
        domain_weight=torch.tensor([20.0 if row['method']=='actual-browser-webrtc' else 1.0 for row in records])[train]
        weights=weights*domain_weight;weights=weights/weights.mean()
        for step in range(800):
            optimizer.zero_grad();logits=head(Z[train]).squeeze(1)
            loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,labels,reduction='none')*weights).mean()+.03*head.weight.square().sum()
            loss.backward();optimizer.step()
        with torch.no_grad():scores=head(Z).squeeze(1).sigmoid().numpy()
        calibration=np.array([r['split']=='calibration' and not r['label'] for r in records])
        threshold=float(min(.99999,max(.5,float(scores[calibration].max())+.01)))
        report=dict(scope='Curated ASVspoof plus 104 Mini LibriSpeech utterances, approximate DSP augmentations and actual browser VoiceTransformationPipeline/WebRTC receiver recordings. Speaker-disjoint fit/calibration/test. Old acceptance utterance excluded from training but reused for regression. Not fresh blind validation or physical live-device evidence.',threshold=threshold,records=[dict(r,score=float(s)) for r,s in zip(records,scores)])
        for split in ['test','method_test','browser_test']:
            mask=np.array([r['split']=='test' and r['method']=='actual-browser-webrtc' if split=='browser_test' else r['split']==split for r in records]);real=scores[mask&(y==0)];fake=scores[mask&(y==1)]
            report[split]=dict(real=len(real),spoof_or_dsp=len(fake),false_positives=int((real>=threshold).sum()),detected=int((fake>=threshold).sum()),real_median=float(np.median(real)) if len(real) else None,fake_median=float(np.median(fake)) if len(fake) else None)
        external=[];external_vectors=[]
        for name in ['clean','robotic-vocoder','deep-pitch-neural','synthetic-clone']:
            feature=features(model,read(ROOT/f'ml/experiments/reference_backend/browser_speech_call/{name}.wav'))
            external_vectors.append(feature)
            with torch.no_grad():score=float(head(torch.tensor((feature-mean)/std)).sigmoid().item())
            external.append(dict(name=name,score=score,above_threshold=score>=threshold))
        np.savez(OUT/'external_features.npz',X=np.stack(external_vectors))
        report['external_received_call']=external;report['seconds']=time.time()-started
        # Do not enable a candidate that fails the actual clean/filtered call regression.
        report['passed_local_gate']=bool(report['test']['false_positives']==0 and report['test']['detected']>=.8*report['test']['spoof_or_dsp'] and external[0]['score']<threshold and all(r['score']>=threshold for r in external[1:]))
        report['validated_for_live_calls']=False
        report['speaker_leakage_audit_passed']=True
        torch.save(dict(mean=torch.tensor(mean),std=torch.tensor(std),weight=head.weight.detach(),bias=head.bias.detach(),threshold=threshold),OUT/'candidate.pt')
        np.savez(OUT/'features.npz',X=X,y=y)
        (OUT/'results.json').write_text(json.dumps(report,indent=2));(OUT/'status.json').write_text(json.dumps(dict(stage='complete',seconds=report['seconds'],passed_local_gate=report['passed_local_gate'])))
        print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2),flush=True)
    finally:model.close()

if __name__=='__main__':main()
