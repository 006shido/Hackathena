"""Shared training/inference features for the isolated call-domain candidate."""
import numpy as np
from scipy.signal import stft

def call_features(hidden, logits, pcm):
    _,_,spectrum=stft(pcm,16000,nperseg=512,noverlap=256)
    power=np.abs(spectrum)**2
    edges=np.geomspace(80,7600,33);frequency=np.arange(power.shape[0])*16000/512
    bands=np.stack([power[(frequency>=a)&(frequency<b)].sum(0) for a,b in zip(edges[:-1],edges[1:])])
    bands=10*np.log10(np.maximum(bands/np.maximum(bands.sum(0),1e-10),1e-8))
    return np.concatenate([hidden[0].detach().cpu().numpy(),logits[0].detach().cpu().numpy(),bands.mean(1),bands.std(1)]).astype(np.float32)
