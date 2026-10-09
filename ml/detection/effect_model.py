"""Load only a locally gated, checksum-matched DSP voice-effect candidate."""
import hashlib,json
from pathlib import Path
import numpy as np,torch
from scipy.signal import stft

def spectral_statistics(pcm):
    _,_,spectrum=stft(pcm,16000,nperseg=512,noverlap=256)
    power=np.abs(spectrum)**2;frequency=np.arange(power.shape[0])*16000/512;edges=np.geomspace(80,7600,33)
    bands=np.stack([power[(frequency>=a)&(frequency<b)].sum(0) for a,b in zip(edges[:-1],edges[1:])])
    bands=10*np.log10(np.maximum(bands/np.maximum(bands.sum(0),1e-10),1e-8))
    return np.concatenate([bands.mean(1),bands.std(1)]).astype(np.float32)

def effect_features(pcm):
    """Gain-invariant spectrum, envelope modulation and waveform shape."""
    pcm=np.asarray(pcm,dtype=np.float32);rms=max(float(np.sqrt(np.mean(pcm**2))),1e-6)
    envelope=np.abs(pcm[:len(pcm)//32*32]).reshape(-1,32).mean(1)
    modulation=np.abs(np.fft.rfft((envelope-envelope.mean())*np.hanning(len(envelope))))**2
    frequencies=np.fft.rfftfreq(len(envelope),1/500)
    total=max(float(modulation[(frequencies>=1)&(frequencies<230)].sum()),1e-12)
    edges=[1,4,10,20,30,45,70,110,150,180,230]
    modulation_bands=[10*np.log10(max(1e-8,float(modulation[(frequencies>=a)&(frequencies<b)].sum())/total)) for a,b in zip(edges[:-1],edges[1:])]
    frames=pcm[:len(pcm)//320*320].reshape(-1,320);energies=np.sqrt((frames**2).mean(1))/rms
    shape=[float(energies.mean()),float(energies.std()),*np.quantile(energies,[.1,.5,.9]),float(abs(pcm).max()/rms),float(np.mean(pcm**4)/rms**4),float(np.mean(pcm[1:]*pcm[:-1]<0)),float(np.mean(np.diff(pcm)**2)/rms**2),float(np.mean(abs(pcm))/rms),float(np.quantile(abs(pcm),.9)/rms)]
    return np.concatenate([spectral_statistics(pcm),modulation_bands,shape]).astype(np.float32)

class CallEffectModel:
    def __init__(self,path):
        path=Path(path);report=json.loads((path/'effect_results.json').read_text());weights=path/'effect_candidate.pt'
        if not report.get('passed_local_gate'):raise ValueError('Call effect candidate failed local validation')
        if hashlib.sha256(weights.read_bytes()).hexdigest()!=report['sha256']:raise ValueError('Effect checkpoint checksum mismatch')
        state=torch.load(weights,map_location='cpu',weights_only=True)
        for key in ['mean','std','weight','bias']:
            if not torch.isfinite(state[key]).all():raise ValueError('Invalid effect checkpoint')
        if state['mean'].shape!=(64,) or state['std'].shape!=(64,) or state['weight'].shape!=(1,64) or state['bias'].shape!=(1,) or (state['std']<=0).any():raise ValueError('Invalid effect checkpoint dimensions')
        self.mean=state['mean'].numpy();self.std=state['std'].numpy();self.weight=state['weight'].numpy().flatten();self.bias=float(state['bias'][0]);self.threshold=float(state['threshold'])
        if not .5<=self.threshold<1:raise ValueError('Invalid calibrated threshold')
    def score(self,pcm):
        features=spectral_statistics(pcm)
        value=float(((features-self.mean)/self.std)@self.weight+self.bias)
        return float(1/(1+np.exp(-np.clip(value,-40,40))))
