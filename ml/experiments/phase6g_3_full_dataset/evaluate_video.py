"""Offline frame test on a real video or explicitly synthetic motion sequence."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from PIL import Image
from ml.inference.infer_phase6g import Phase6GInferenceEngine
from ml.inference.frame_pipeline import NeuralFrameSession


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=ROOT/'ml/data/celeba/img_align_celeba/197935.jpg')
    parser.add_argument('--target',type=Path,default=ROOT/'ml/data/celeba/img_align_celeba/098180.jpg')
    parser.add_argument('--video',type=Path)
    parser.add_argument('--checkpoint',type=Path,default=Path(__file__).parent/'run/best_model.pt')
    parser.add_argument('--correspondence',choices=['legacy','visibility'],default='legacy')
    parser.add_argument('--frames',type=int,default=60)
    parser.add_argument('--save-raw-frames',type=int,nargs='*',default=[],help='Save lossless original/output panels at selected zero-based frames')
    parser.add_argument('--reference-gpu',action='store_true',help='Offline research backend only')
    parser.add_argument('--lowlight-retry',action='store_true')
    parser.add_argument('--identity-guard',action='store_true')
    parser.add_argument('--video-landmarks',action='store_true',help='Experimental single-face temporal tracking, offline only')
    parser.add_argument('--occlusion-mask',action='store_true',help='Experimental GPU XSeg composition, offline only; requires identity guard')
    parser.add_argument('--occlusion-feather',type=float,default=0,help='Experimental visible-side feather radius in original-frame pixels')
    parser.add_argument('--occlusion-temporal',action='store_true',help='Experimental motion-compensated visibility smoothing')
    parser.add_argument('--semantic-hair',action='store_true',help='Experimental GPU semantic hair override for XSeg')
    parser.add_argument('--synthetic-size',type=int,nargs=2,metavar=('WIDTH','HEIGHT'))
    parser.add_argument('--motion-amplitude',type=float,default=1)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'video_probe')
    args=parser.parse_args()
    if args.lowlight_retry and not args.reference_gpu:raise ValueError('Low-light retry is currently a research-backend experiment.')
    if args.identity_guard and not args.reference_gpu:raise ValueError('Identity guard requires the research backend.')
    if args.video_landmarks and (not args.reference_gpu or not args.video or args.lowlight_retry):raise ValueError('Video tracking requires a real research video and no image retry.')
    if args.occlusion_mask and (not args.reference_gpu or not args.identity_guard):raise ValueError('Occlusion mask requires guarded research backend.')
    if not np.isfinite(args.occlusion_feather) or args.occlusion_feather<0 or (args.occlusion_feather and not args.occlusion_mask):raise ValueError('Invalid occlusion feather configuration.')
    if args.occlusion_temporal and not args.occlusion_mask:raise ValueError('Temporal occlusion requires the mask experiment.')
    if args.semantic_hair and not args.occlusion_mask:raise ValueError('Semantic hair override requires the mask experiment.')
    if args.motion_amplitude<=0 or not np.isfinite(args.motion_amplitude):raise ValueError('Motion amplitude must be positive.')
    args.output.mkdir(parents=True,exist_ok=True)
    if args.reference_gpu:
        from types import SimpleNamespace
        import torch
        from ml.experiments.reference_backend.opencv_reference import OpenCVReference
        from ml.experiments.reference_backend.frame_session import ReferenceFrameSession
        from ml.training.face_preprocessing import RealFacePreprocessor
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        engine=SimpleNamespace(reference=OpenCVReference(gpu=True),preprocessor=RealFacePreprocessor(image_size=128),lowlight_retry=args.lowlight_retry,video_landmarks=args.video_landmarks)
        if args.identity_guard:
            import hashlib
            from ml.models.face_swap_model import ArcFaceIdentityExtractor
            weights=ROOT/'ml/models/weights/ms1mv2_iresnet50.pth'
            assert hashlib.sha256(weights.read_bytes()).hexdigest().upper()=='2B75B93C48B01C78A4263F7295AB2DBF84F85190C51BE45B92C4C0B0AEDCEAA3'
            engine.identity_model=ArcFaceIdentityExtractor(str(weights)).cuda().eval()
            engine.identity_guard=True
        if args.occlusion_mask:
            from ml.experiments.reference_backend.torch_occlusion import TorchOcclusion
            engine.occlusion_model=TorchOcclusion()
            engine.occlusion_feather=args.occlusion_feather
            engine.occlusion_temporal=args.occlusion_temporal
            engine.occlusion_model(np.zeros((1,256,256,3),np.float32)).cpu().numpy()
        if args.semantic_hair:
            from ml.experiments.reference_backend.torch_semantic import TorchSemantic
            engine.semantic_model=TorchSemantic()
            engine.semantic_model(np.zeros((1,3,512,512),np.float32)).cpu().numpy()
        session=ReferenceFrameSession(engine,Image.open(args.source))
    else:
        engine=Phase6GInferenceEngine(checkpoint_path=str(args.checkpoint),model_variant='fullres')
        session=NeuralFrameSession(engine,Image.open(args.source),args.correspondence)
    capture=cv2.VideoCapture(str(args.video)) if args.video else None
    fps=capture.get(cv2.CAP_PROP_FPS) if capture else 15
    if not fps or not np.isfinite(fps):fps=15
    base=np.array(Image.open(args.target).convert('RGB')) if not capture else None
    if args.synthetic_size:
        if capture:raise ValueError('--synthetic-size is only for synthetic motion.')
        width,height=args.synthetic_size
        if width<128 or height<128 or width*height>1920*1080:raise ValueError('Invalid synthetic frame size.')
        scale=min(width/base.shape[1],height/base.shape[0])
        resized=cv2.resize(base,(round(base.shape[1]*scale),round(base.shape[0]*scale)))
        canvas=np.zeros((height,width,3),dtype=np.uint8)
        x=(width-resized.shape[1])//2;y=(height-resized.shape[0])//2
        canvas[y:y+resized.shape[0],x:x+resized.shape[1]]=resized
        base=canvas
    stability_mask=None
    if not capture:
        from ml.training.face_preprocessing import generate_facial_mask,MEDIAPIPE_FACE_OVAL_INDICES
        detection=engine.preprocessor.detect_landmarks(Image.fromarray(base))
        height,width=base.shape[:2]
        # Generate the same anatomical mask in normalized 128px coordinates,
        # then project it to the original (possibly non-square) synthetic frame.
        points=detection.dense_landmarks[MEDIAPIPE_FACE_OVAL_INDICES]*[128/width,128/height]
        small=generate_facial_mask(points,128,0).squeeze()
        stability_mask=cv2.resize(small,(width,height))>.99
        stability_mask=cv2.erode(stability_mask.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
        if not stability_mask.any():raise RuntimeError('No synthetic stability region.')
    writer=None;records=[];previous=None;previous_stable=None;previous_input=None
    try:
        for index in range(args.frames):
            if capture:
                ok,bgr=capture.read()
                if not ok:break
                frame=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            else:
                height,width=base.shape[:2]
                affine=cv2.getRotationMatrix2D((width/2,height/2),2*args.motion_amplitude*np.sin(index/10),1)
                affine[:,2]+=args.motion_amplitude*np.array([2*np.sin(index/7),np.cos(index/9)])
                frame=cv2.warpAffine(base,affine,(width,height),borderMode=cv2.BORDER_REFLECT_101)
            if args.reference_gpu:engine.capture_geometry=index in args.save_raw_frames
            output,diagnostic=session.process(Image.fromarray(frame),1000*index/fps)
            if index in args.save_raw_frames:
                Image.fromarray(frame).save(args.output/f'raw_original_{index}.png')
                Image.fromarray(output).save(args.output/f'raw_output_{index}.png')
                (args.output/f'raw_geometry_{index}.json').write_text(json.dumps(diagnostic,indent=2))
            diagnostic['frame']=index
            diagnostic['output_delta_mean']=None if previous is None else float(np.abs(output.astype(float)-previous).mean())
            previous=output.astype(float)
            if not capture:
                inverse=cv2.invertAffineTransform(affine)
                stable=cv2.warpAffine(output.astype(np.float32),inverse,(width,height),borderMode=cv2.BORDER_REFLECT_101)
                stable_input=cv2.warpAffine(frame.astype(np.float32),inverse,(width,height),borderMode=cv2.BORDER_REFLECT_101)
                diagnostic['motion_compensated_output_mae']=None if previous_stable is None else float(np.abs(stable-previous_stable)[stability_mask].mean())
                diagnostic['motion_compensated_input_mae']=None if previous_input is None else float(np.abs(stable_input-previous_input)[stability_mask].mean())
                previous_stable=stable;previous_input=stable_input
            records.append(diagnostic)
            if writer is None:
                writer=cv2.VideoWriter(str(args.output/'comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(frame.shape[1]*2,frame.shape[0]))
                if not writer.isOpened():raise RuntimeError('Could not create video artifact')
            writer.write(cv2.cvtColor(np.concatenate([frame,output],axis=1),cv2.COLOR_RGB2BGR))
            if index==0:Image.fromarray(np.concatenate([frame,output],axis=1)).save(args.output/'first_frame.png')
            if (index+1)%15==0:print(f'Processed {index+1} frames',flush=True)
    finally:
        if capture:capture.release()
        if writer:writer.release()
        if hasattr(session,'close'):session.close()
    if not records:raise RuntimeError('No frames processed')
    # Exclude cold first frame; include preprocessing, projection and transfer.
    warm=[r['pipeline_ms'] for r in records[1:]] or [records[0]['pipeline_ms']]
    report=dict(synthetic_motion=args.video is None,
        limitation='Synthetic translation/rotation is not a real expression or occlusion video test.' if args.video is None else 'Single supplied video; not a deployment acceptance suite.',
        correspondence=None if args.reference_gpu else args.correspondence,
        checkpoint='official-research-inswapper-128' if args.reference_gpu else str(args.checkpoint),
        frames=len(records),frame_size=[frame.shape[1],frame.shape[0]],
        face_detection_rate=100*np.mean([r['face_detected'] for r in records]),
        warm_pipeline_ms_median=float(np.median(warm)),warm_pipeline_ms_p95=float(np.percentile(warm,95)),records=records)
    report['lowlight_retry_enabled']=args.lowlight_retry
    report['lowlight_recovered_frames']=sum(bool(r.get('lowlight_retry')) for r in records)
    report['identity_guard_enabled']=args.identity_guard
    report['occlusion_mask_enabled']=args.occlusion_mask
    report['occlusion_feather']=args.occlusion_feather
    report['occlusion_temporal']=args.occlusion_temporal
    report['semantic_hair']=args.semantic_hair
    report['video_landmarks_enabled']=args.video_landmarks
    report['identity_rejected_frames']=sum(r.get('identity_accepted') is False for r in records)
    report['face_changed_frames']=sum(bool(r.get('face_changed',r['changed_pixels']>0)) for r in records)
    if not capture:
        report['synthetic_motion_amplitude']=args.motion_amplitude
        report['stability_metric']='Consecutive RGB MAE (0–255) inside eroded target face mask after inverse known global motion; interpolation residual remains. Not a real-expression flicker measurement.'
        for label in ['output','input']:
            values=[r[f'motion_compensated_{label}_mae'] for r in records if r[f'motion_compensated_{label}_mae'] is not None]
            if values:
                report[f'motion_compensated_{label}_mae_mean']=float(np.mean(values))
                report[f'motion_compensated_{label}_mae_p95']=float(np.percentile(values,95))
    (args.output/'results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)


if __name__=='__main__':main()
