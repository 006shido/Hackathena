"""Locate XSeg exclusion within genuine landmark-derived facial regions.

Sampled IMAGE landmarks are diagnostic only, not live VIDEO geometry or labels
that prove a pixel is an occluder. No acceptance thresholds change.
"""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from ml.training.face_preprocessing import RealFacePreprocessor,generate_facial_mask,MEDIAPIPE_FACE_OVAL_INDICES,estimate_umeyama_similarity_transform,FaceDetectionError
from ml.experiments.reference_backend.opencv_reference import CANONICAL
from ml.experiments.reference_backend.torch_occlusion import TorchOcclusion

def main():
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=TorchOcclusion();pre=RealFacePreprocessor(image_size=128)
    root=Path(__file__).parent;folder=root/'mask_coverage';folder.mkdir(exist_ok=True)
    cap=cv2.VideoCapture(str(root/'videos/faceocc2.webm'));records=[]
    picks=sorted(set(list(range(0,812,40))+[83,150,478,737,811]))
    try:
        for index in picks:
            cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,bgr=cap.read()
            if not ok:raise ValueError(f'Missing frame {index}')
            original=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            try:det=pre.detect_landmarks(Image.fromarray(original))
            except FaceDetectionError as error:
                records.append(dict(frame=index,detected=False,reason=str(error)));continue
            canonical=CANONICAL.copy();canonical[:,0]+=8
            affine=estimate_umeyama_similarity_transform(det.key_landmarks_5pts,canonical)
            crop=cv2.warpAffine(original,affine*2,(256,256))
            score=model(crop[:,:,::-1].astype(np.float32)[None]/255).cpu().numpy()
            visible=cv2.dilate((score>=.1).astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)))
            visible=cv2.resize(visible.astype(np.float32),(128,128),interpolation=cv2.INTER_AREA)>=.999
            dense=cv2.transform(det.dense_landmarks[None].astype(np.float32),affine)[0]
            oval=generate_facial_mask(dense[MEDIAPIPE_FACE_OVAL_INDICES],128,0).squeeze()>.99
            core=cv2.erode(oval.astype(np.uint8),np.ones((15,15),np.uint8))>0
            if not core.any():raise ValueError('Empty eroded face region')
            excluded=oval&~visible;core_excluded=core&~visible
            records.append(dict(frame=index,detected=True,face_pixels=int(oval.sum()),excluded_face_pixels=int(excluded.sum()),core_pixels=int(core.sum()),excluded_core_pixels=int(core_excluded.sum()),core_excluded_fraction=float(core_excluded.sum()/core.sum()),excluded_pixels_outside_core_fraction=float((excluded&~core).sum()/max(1,excluded.sum()))))
            if index in [0,150,478]:
                aligned=cv2.warpAffine(original,affine,(128,128));annotation=aligned.copy()
                annotation[excluded]=(255,80,0);annotation[core_excluded]=(255,0,255)
                cv2.drawContours(annotation,cv2.findContours(core.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(0,255,0),1)
                Image.fromarray(np.concatenate([aligned,annotation],1)).save(folder/f'frame{index}.png')
            print(f'Mask coverage frame {index}',flush=True)
    finally:cap.release()
    report=dict(scope='Sampled faceocc2 frames with independent IMAGE landmarks. Orange excludes face boundary, magenta excludes eroded core; neither is a ground-truth object label. Core uses 15px erosion in aligned128 coordinates.',records=records)
    (folder/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(records))
if __name__=='__main__':main()
