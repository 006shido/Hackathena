"""Isolated lossless XSeg compositing probe; never changes service defaults."""
import argparse,hashlib,json,time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from ml.training.face_preprocessing import RealFacePreprocessor,estimate_umeyama_similarity_transform,get_canonical_landmarks
from ml.experiments.reference_backend.opencv_reference import CANONICAL
from ml.inference.identity_guard import IdentityGuard
from ml.models.face_swap_model import ArcFaceIdentityExtractor

ROOT=Path(__file__).resolve().parents[3]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--sweep',action='store_true');parser.add_argument('--live-geometry',action='store_true');parser.add_argument('--boundary-sweep',action='store_true');parser.add_argument('--gpu',action='store_true');args=parser.parse_args()
    if args.boundary_sweep and not args.live_geometry:raise ValueError('Boundary sweep requires saved live geometry.')
    asset=Path(__file__).parent/'assets/xseg_1.onnx'
    expected='c4d1498b8a03b5fe2a3a5d2ef2a0402ab03bd51edaf5b2d8d5fb764702a97dd3'
    if hashlib.sha256(asset.read_bytes()).hexdigest()!=expected:raise ValueError('XSeg checksum mismatch')
    net=cv2.dnn.readNetFromONNX(str(asset));cv2.setNumThreads(4)
    gpu=None
    if args.gpu:
        import torch
        from ml.experiments.reference_backend.torch_occlusion import TorchOcclusion
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        gpu=TorchOcclusion()
    detector=RealFacePreprocessor(image_size=128)
    source=Image.open(ROOT/'ml/data/celeba/img_align_celeba/197935.jpg').convert('RGB')
    source_points=detector.detect_landmarks(source).key_landmarks_5pts
    weights=ROOT/'ml/models/weights/ms1mv2_iresnet50.pth'
    if hashlib.sha256(weights.read_bytes()).hexdigest()!='2b75b93c48b01c78a4263f7295ab2dbf84f85190c51be45b92c4c0b0aedceaa3':raise ValueError('ArcFace checksum mismatch')
    guard=IdentityGuard(ArcFaceIdentityExtractor(str(weights)).cuda().eval(),source,source_points)
    folder=Path(__file__).parent/'raw_occluder_probe';reports=[]
    artifact_folder=folder/('boundary_sweep' if args.boundary_sweep else 'live_alignment') if args.live_geometry else folder
    if args.gpu:artifact_folder=folder/(artifact_folder.name+'_gpu')
    artifact_folder.mkdir(exist_ok=True)
    first=np.array(Image.open(folder/'first_frame.png').convert('RGB'));width=first.shape[1]//2
    pairs=[('clear_frame0',first[:,:width],first[:,width:]),('book_frame150',np.array(Image.open(folder/'raw_original_150.png')),np.array(Image.open(folder/'raw_output_150.png')))]
    for name,original,baseline in pairs:
        if args.live_geometry:
            index=0 if name.startswith('clear') else 150
            geometry=json.loads((folder/f'raw_geometry_{index}.json').read_text())
            if not geometry.get('face_changed'):raise ValueError('Probe requires an accepted swapped frame.')
            points=np.asarray(geometry['key_landmarks_5pts'],dtype=np.float32)
            common=estimate_umeyama_similarity_transform(points,get_canonical_landmarks(128))
            reproduced=guard.check(cv2.warpAffine(original,common,(128,128)),cv2.warpAffine(baseline,common,(128,128)))
            if any(abs(reproduced[key]-geometry[key])>1e-4 for key in ('source_similarity','target_similarity')):
                raise ValueError('Saved geometry does not reproduce baseline identity scores.')
        else:points=detector.detect_landmarks(Image.fromarray(original)).key_landmarks_5pts
        canonical=CANONICAL.copy();canonical[:,0]+=8
        affine=estimate_umeyama_similarity_transform(points,canonical*2)
        crop=cv2.warpAffine(original,affine,(256,256))
        timings=[]
        input_array=crop[:,:,::-1].astype(np.float32)[None]/255
        real_parity=None
        if gpu:
            net.setInput(input_array);cpu_mask=net.forward()[0,:,:,0]
            gpu_mask=gpu(input_array).cpu().numpy();difference=np.abs(cpu_mask-gpu_mask)
            real_parity=dict(max_error=float(difference.max()),mean_error=float(difference.mean()))
            if difference.max()>=.001:raise ValueError('Real-crop CPU/GPU parity failed')
        for _ in range(6 if args.sweep else 1):
            start=time.perf_counter()
            if gpu:raw_mask=gpu(input_array).cpu().numpy()
            else:net.setInput(input_array);raw_mask=net.forward()[0,:,:,0]
            timings.append(1000*(time.perf_counter()-start))
        if not np.isfinite(raw_mask).all():raise ValueError('Nonfinite mask')
        if args.boundary_sweep:
            for dilation in [0,3,7,11]:
                evaluate_mask(name,original,baseline,points,affine,raw_mask,.1,timings,guard,artifact_folder,reports,True,hard=True,dilation=dilation,unknown_visible=True)
                reports[-1]['gpu_parity']=real_parity
            continue
        thresholds=[.1,.2,.3,.4,.5] if args.sweep else [.5]
        for threshold in thresholds:
            evaluate_mask(name,original,baseline,points,affine,raw_mask,threshold,timings,guard,artifact_folder,reports,args.sweep)
            if args.sweep:
                evaluate_mask(name,original,baseline,points,affine,raw_mask,threshold,timings,guard,artifact_folder,reports,True,hard=True)
    report=dict(scope='Two annotated lossless frames only. Threshold sweep is exploratory, not held-out validation or video runtime promotion.',alignment='saved live VIDEO landmarks with reproduced baseline scores' if args.live_geometry else 'independent IMAGE landmarks',model_sha256=expected,model_url='https://huggingface.co/facefusion/models-3.1.0/blob/main/xseg_1.onnx',license_metadata='GPL-3.0 per FaceFusion model metadata',records=reports)
    filename='xseg_boundary_sweep.json' if args.boundary_sweep else ('xseg_live_sweep.json' if args.live_geometry else ('xseg_sweep.json' if args.sweep else 'xseg_probe.json'))
    report['execution']='torch GPU' if args.gpu else 'OpenCV CPU'
    if args.gpu:filename=filename.replace('.json','_gpu.json')
    (folder/filename).write_text(json.dumps(report,indent=2));print(json.dumps(report))

def evaluate_mask(name,original,baseline,points,affine,raw_mask,threshold,timings,guard,folder,reports,sweep,hard=False,dilation=0,unknown_visible=False):
        mask=(raw_mask>=threshold).astype(np.float32) if hard else np.clip((cv2.GaussianBlur(np.clip(raw_mask,0,1),(0,0),5)-threshold)/(1-threshold),0,1)
        if dilation:mask=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*dilation+1,2*dilation+1)))
        full=cv2.warpAffine(mask,cv2.invertAffineTransform(affine),(original.shape[1],original.shape[0]),borderValue=1 if unknown_visible else 0)
        if hard:full=(full>=.999).astype(np.float32)
        candidate=np.clip(np.rint(original.astype(float)+(baseline.astype(float)-original)*full[:,:,None]),0,255).astype(np.uint8)
        common=estimate_umeyama_similarity_transform(points,get_canonical_landmarks(128))
        target=cv2.warpAffine(original,common,(128,128))
        scores={label:guard.check(target,cv2.warpAffine(output,common,(128,128))) for label,output in [('baseline',baseline),('candidate',candidate)]}
        stem=f'{name}_threshold{threshold:.1f}' if sweep else name
        if hard:stem+='_hard'
        if unknown_visible:stem+=f'_dilate{dilation}_unknown_visible'
        Image.fromarray(candidate).save(folder/f'{stem}_candidate.png')
        cv2.imwrite(str(folder/f'{stem}_mask.png'),np.rint(full*255).astype(np.uint8))
        Image.fromarray(np.concatenate([original,baseline,candidate],axis=1)).save(folder/f'{stem}_comparison.png')
        record=dict(name=name,threshold=threshold,hard=hard,dilation_radius=dilation,unknown_visible=unknown_visible,inference_ms=timings[0],warm_inference_ms_median=float(np.median(timings[1:] or timings)),identity=scores)
        if name.startswith('book'):
            polygon=np.array([[110,110],[232,120],[222,203],[104,190]],np.int32);roi=np.zeros(full.shape,np.uint8);cv2.fillPoly(roi,[polygon],255);roi=cv2.erode(roi,np.ones((9,9),np.uint8))>0
            record['book_changes']={label:dict(mae=float(np.abs(output.astype(float)-original)[roi].mean()),fraction_over8=float((np.abs(output.astype(float)-original).max(2)[roi]>8).mean())) for label,output in [('baseline',baseline),('candidate',candidate)]}
        reports.append(record)
if __name__=='__main__':main()
