"""Verify received, codec-decoded browser snapshots against both chosen sources."""
import sys,json,hashlib,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import cv2,torch
import torch.nn.functional as F
from PIL import Image
from ml.models.face_swap_model import ArcFaceIdentityExtractor
from ml.inference.face_correspondence import preprocess_single_face
from ml.training.face_preprocessing import RealFacePreprocessor

ROOT=Path(__file__).resolve().parents[3]
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path(__file__).parent/'browser_call')
output=parser.parse_args().output
cap=cv2.VideoCapture(str(Path(__file__).parent/'videos/faceocc2.webm'))
ok,bgr=cap.read();cap.release();assert ok
Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).save(output/'original-target.png')
weights=ROOT/'ml/models/weights/ms1mv2_iresnet50.pth'
assert hashlib.sha256(weights.read_bytes()).hexdigest().upper()=='2B75B93C48B01C78A4263F7295AB2DBF84F85190C51BE45B92C4C0B0AEDCEAA3'
model=ArcFaceIdentityExtractor(str(weights)).cuda().eval()
pre=RealFacePreprocessor(image_size=128)
paths={'source-a':ROOT/'ml/data/celeba/img_align_celeba/197935.jpg',
       'source-b':ROOT/'ml/data/celeba/img_align_celeba/039613.jpg','target':output/'original-target.png',
       **{name:output/(name+'.png') for name in ['remote-a','remote-b','remote-restored']}}
embeddings={}
with torch.inference_mode():
    for name,path in paths.items():
        _,tensor,*_=preprocess_single_face(pre,str(path))
        embeddings[name]=model(torch.from_numpy(tensor).unsqueeze(0).cuda())
results={}
for name in ['remote-a','remote-b','remote-restored']:
    results[name]={other:float(F.cosine_similarity(embeddings[name],embeddings[other])) for other in ['source-a','source-b','target']}
for suffix in ['a','b']:
    scores=results['remote-'+suffix];selected=scores['source-'+suffix]
    assert selected>=.3 and selected-scores['target']>=.05 and selected-scores['source-'+('b' if suffix=='a' else 'a')]>=.05,scores
assert results['remote-restored']['target']>.5,results['remote-restored']
assert results['remote-restored']['target']>max(results['remote-restored']['source-a'],results['remote-restored']['source-b'])+.1
report=dict(passed=True,scores=results,scope='Three received snapshots, original target ID from clip frame zero; independent of the generator, same scorer as the identity guard. Not an all-frame identity or speech test.')
(output/'identity_check.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
