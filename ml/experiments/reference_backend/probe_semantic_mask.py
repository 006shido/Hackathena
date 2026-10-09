"""Isolated semantic composition probe using saved live VIDEO geometry."""
import argparse,hashlib,json,time
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from ml.training.face_preprocessing import RealFacePreprocessor,estimate_umeyama_similarity_transform,get_canonical_landmarks
from ml.experiments.reference_backend.opencv_reference import CANONICAL
from ml.models.face_swap_model import ArcFaceIdentityExtractor
from ml.inference.identity_guard import IdentityGuard

ROOT=Path(__file__).resolve().parents[3]
SHA='2218b6183c26ca5c83303232d682a536c670c13ea9695f716c777d1f244eefe9'
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--combine-xseg',action='store_true');parser.add_argument('--gpu',action='store_true');args=parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=Path(__file__).parent;asset=root/'assets/bisenet_resnet_18.onnx'
    if hashlib.sha256(asset.read_bytes()).hexdigest()!=SHA:raise ValueError('Parser checksum mismatch')
    net=cv2.dnn.readNetFromONNX(str(asset));cv2.setNumThreads(4)
    gpu=None
    if args.gpu:
        from ml.experiments.reference_backend.torch_semantic import TorchSemantic
        gpu=TorchSemantic()
    weights=ROOT/'ml/models/weights/ms1mv2_iresnet50.pth'
    if hashlib.sha256(weights.read_bytes()).hexdigest()!='2b75b93c48b01c78a4263f7295ab2dbf84f85190c51be45b92c4c0b0aedceaa3':raise ValueError('Scorer checksum mismatch')
    source=Image.open(ROOT/'ml/data/celeba/img_align_celeba/197935.jpg').convert('RGB');pre=RealFacePreprocessor(image_size=128)
    guard=IdentityGuard(ArcFaceIdentityExtractor(str(weights)).cuda().eval(),source,pre.detect_landmarks(source).key_landmarks_5pts)
    xseg=None
    if args.combine_xseg:
        from ml.experiments.reference_backend.torch_occlusion import TorchOcclusion
        xseg=TorchOcclusion()
    data=root/'current_baseline_occlusion_video';out=root/(('combined_mask_probe' if args.combine_xseg else 'semantic_mask_probe')+('_gpu' if args.gpu else ''));out.mkdir(exist_ok=True);records=[]
    for index in [0,150,478]:
        original=np.array(Image.open(data/f'raw_original_{index}.png'));baseline=np.array(Image.open(data/f'raw_output_{index}.png'))
        geometry=json.loads((data/f'raw_geometry_{index}.json').read_text());points=np.array(geometry['key_landmarks_5pts'],np.float32)
        canonical=CANONICAL.copy();canonical[:,0]+=8
        affine=estimate_umeyama_similarity_transform(points,canonical*4);crop=cv2.warpAffine(original,affine,(512,512))
        array=(crop.astype(np.float32)/255-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        input_array=array.transpose(2,0,1)[None]
        parity=None
        if gpu:
            net.setInput(input_array);reference=net.forward('output');candidate_logits=gpu(input_array).cpu().numpy();difference=np.abs(reference-candidate_logits)
            parity=dict(max_error=float(difference.max()),mean_error=float(difference.mean()),label_agreement=float(np.mean(reference.argmax(1)==candidate_logits.argmax(1))))
            if difference.max()>=.002:raise ValueError('Real semantic crop parity failed')
        timings=[]
        for _ in range(4):
            start=time.perf_counter()
            if gpu:logits=gpu(input_array).cpu().numpy()
            else:net.setInput(input_array);logits=net.forward('output')
            timings.append(1000*(time.perf_counter()-start))
        if logits.shape!=(1,19,512,512) or not np.isfinite(logits).all():raise ValueError('Invalid parser output')
        labels=logits[0].argmax(0).astype(np.uint8)
        selected=np.isin(labels,[1,2,3,4,5,10,11,12,13]).astype(np.float32)
        visibility=cv2.warpAffine(selected,cv2.invertAffineTransform(affine),(original.shape[1],original.shape[0]),borderValue=1)>=.999
        if xseg:
            small=cv2.resize(crop,(256,256))
            values=xseg(small[:,:,::-1].astype(np.float32)[None]/255).cpu().numpy()
            xvisible=cv2.dilate((values>=.1).astype(np.float32),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)))
            xvisible=cv2.warpAffine(xvisible,cv2.invertAffineTransform(affine/2),(original.shape[1],original.shape[0]),borderValue=1)>=.999
            hair=cv2.warpAffine((labels==17).astype(np.uint8),cv2.invertAffineTransform(affine),(original.shape[1],original.shape[0]),flags=cv2.INTER_NEAREST)>0
            # Retain baseline treatment of semantic hair; XSeg handles the rest.
            visibility=xvisible|hair
        candidate=baseline.copy();candidate[~visibility]=original[~visibility]
        common=estimate_umeyama_similarity_transform(points,get_canonical_landmarks(128));target=cv2.warpAffine(original,common,(128,128))
        scores={name:guard.check(target,cv2.warpAffine(image,common,(128,128))) for name,image in [('baseline',baseline),('candidate',candidate)]}
        if any(abs(scores['baseline'][key]-geometry[key])>1e-4 for key in ('source_similarity','target_similarity')):raise ValueError('Baseline score mismatch')
        Image.fromarray(labels).save(out/f'labels_{index}.png')
        Image.fromarray(np.concatenate([original,baseline,candidate],1)).save(out/f'comparison_{index}.png')
        record=dict(frame=index,warm_inference_ms_median=float(np.median(timings[1:])),gpu_parity=parity,identity=scores,class_pixel_counts={str(label):int((labels==label).sum()) for label in range(19)})
        if index==150:
            polygon=np.array([[110,110],[232,120],[222,203],[104,190]],np.int32);roi=np.zeros(original.shape[:2],np.uint8);cv2.fillPoly(roi,[polygon],255);roi=cv2.erode(roi,np.ones((9,9),np.uint8))>0
            record['book_candidate_mae']=float(np.abs(candidate.astype(float)-original)[roi].mean())
        records.append(record)
    report=dict(scope='Three lossless tuning frames with saved live VIDEO landmarks. No runtime or general-quality promotion.',execution='GPU' if gpu else 'CPU',method='XSeg visibility, with semantic hair retaining baseline pixels' if args.combine_xseg else 'Hard semantic facial classes, preserving hair/glasses/background',model_sha256=SHA,records=records)
    (out/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
