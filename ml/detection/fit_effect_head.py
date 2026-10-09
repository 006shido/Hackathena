"""Automatic detection of this demo's DSP presets, not universal spoof detection."""
import hashlib,json
from pathlib import Path
import numpy as np,torch

def fit(X,records,regularization,domain_weight=12.):
    include=np.array([row['label']==0 or row['variant'] in ['deep','robotic','synthetic','robotic-vocoder','deep-pitch-neural','synthetic-clone'] for row in records])
    train=np.array([row['split']=='train' for row in records]) & include
    # Spectral statistics only: the pretrained AASIST embedding has a known
    # mismatch on clean received audio and is not evidence of these DSP presets.
    spectral=X;mean=spectral[train].mean(0);std=np.maximum(spectral[train].std(0),.1)
    z=torch.tensor((spectral-mean)/std);labels=torch.tensor([row['label'] for row in records],dtype=torch.float32)[train]
    torch.manual_seed(31);head=torch.nn.Sequential(torch.nn.Linear(X.shape[1],32),torch.nn.ReLU(),torch.nn.Linear(32,16),torch.nn.ReLU(),torch.nn.Linear(16,1));optimizer=torch.optim.Adam(head.parameters(),lr=.01,weight_decay=regularization)
    weight=torch.where(labels>0,.5/(labels.mean()+1e-6),.5/(1-labels.mean()+1e-6))
    domain=torch.tensor([domain_weight if row['method']=='actual-browser-webrtc' else 1. for row in records])[train];weight*=domain;weight/=weight.mean()
    for step in range(500):
        optimizer.zero_grad();prediction=head(z[train]).flatten();loss=(torch.nn.functional.binary_cross_entropy_with_logits(prediction,labels,reduction='none')*weight).mean();loss.backward();optimizer.step()
    with torch.no_grad():scores=head(z).flatten().sigmoid().numpy()
    cal=np.array([row['split']=='calibration' for row in records])&include
    negatives=scores[cal&np.array([row['label']==0 for row in records])]
    # Target <=5% calibration false alarms; do not set thresholds from test clips.
    threshold=float(min(.99999,max(.5,np.quantile(negatives,.95,method='higher')+.0001)))
    positives=scores[cal&np.array([row['label']==1 for row in records])]
    actual_cal=np.array([row['method']=='actual-browser-webrtc' and row['label']==1 for row in records])&cal
    quality=int((positives>=threshold).sum())+20*int((scores[actual_cal]>=threshold).sum())
    return dict(mean=torch.tensor(mean),std=torch.tensor(std),state_dict=head.state_dict(),architecture='effect-mlp-32-16-1',threshold=threshold,feature_start=0,feature_dim=X.shape[1]),scores,quality,include

def main():
    root=Path(__file__).parent/'call_domain';report=json.loads((root/'results.json').read_text());records=report['records'];X=np.load(root/'effect_features.npz')['X'];torch.set_num_threads(4)
    attempts=[fit(X,records,penalty,domain) for penalty in [.001,.01,.03] for domain in [3.,12.,30.]]
    # Selection uses only calibration labels; test and old call clips do not select the head.
    candidate,scores,quality,include=max(attempts,key=lambda value:value[2])
    summary=dict(scope='Learned acoustic detector for the three production DSP presets. Not a general synthetic-speech detector. Candidate chosen using calibration only.',threshold=candidate['threshold'],calibration_detected=quality,validation={})
    for name in ['test','browser_test']:
        mask=np.array([row['split']=='test' and (name!='browser_test' or row['method']=='actual-browser-webrtc') for row in records])&include
        negatives=scores[mask&np.array([row['label']==0 for row in records])];positives=scores[mask&np.array([row['label']==1 for row in records])]
        summary['validation'][name]=dict(natural=len(negatives),changed=len(positives),false_positives=int((negatives>=candidate['threshold']).sum()),detected=int((positives>=candidate['threshold']).sum()))
    # External old-call features computed by main trainer, never used for fitting.
    external=np.load(root/'effect_external_features.npz')['X']
    values=(external-candidate['mean'].numpy())/candidate['std'].numpy()
    model=torch.nn.Sequential(torch.nn.Linear(X.shape[1],32),torch.nn.ReLU(),torch.nn.Linear(32,16),torch.nn.ReLU(),torch.nn.Linear(16,1));model.load_state_dict(candidate['state_dict']);model.eval()
    with torch.no_grad():outputs=model(torch.tensor(values)).flatten().sigmoid().numpy()
    summary['external_call']=[dict(name=name,score=float(score)) for name,score in zip(['clean','robotic-vocoder','deep-pitch-neural','synthetic-clone'],outputs)]
    test=summary['validation']['test'];browser=summary['validation']['browser_test']
    summary['passed_local_gate']=bool(test['false_positives']<=.05*test['natural'] and test['detected']>=.8*test['changed'] and browser['false_positives']==0 and browser['detected']>=.8*browser['changed'] and outputs[0]<candidate['threshold'] and all(score>=candidate['threshold'] for score in outputs[1:]))
    summary['validated_for_live_calls']=False
    torch.save(candidate,root/'effect_candidate.pt');summary['sha256']=hashlib.sha256((root/'effect_candidate.pt').read_bytes()).hexdigest()
    (root/'effect_results.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
