"""Approximate real-video continuity on common accepted frames; no pass threshold."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np

def residuals(previous,current,old_outputs,new_outputs):
    old_gray=cv2.cvtColor(previous,cv2.COLOR_BGR2GRAY);new_gray=cv2.cvtColor(current,cv2.COLOR_BGR2GRAY)
    forward=cv2.calcOpticalFlowFarneback(old_gray,new_gray,None,.5,3,15,3,5,1.2,0)
    backward=cv2.calcOpticalFlowFarneback(new_gray,old_gray,None,.5,3,15,3,5,1.2,0)
    height,width=new_gray.shape;y,x=np.mgrid[:height,:width].astype(np.float32)
    mapx=x+backward[...,0];mapy=y+backward[...,1]
    warp=lambda image:cv2.remap(image,mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    consistent=np.linalg.norm(backward+warp(forward),axis=2)<1
    valid=consistent & (mapx>=1)&(mapx<width-2)&(mapy>=1)&(mapy<height-2)
    input_error=np.abs(current.astype(np.float32)-warp(previous).astype(np.float32)).mean(axis=2)
    valid &= input_error<20
    support=np.zeros((height,width),bool)
    for old,new in zip(old_outputs,new_outputs):
        support |= np.max(np.abs(new.astype(float)-current),axis=2)>8
        support |= warp((np.max(np.abs(old.astype(float)-previous),axis=2)>8).astype(np.uint8))>0
    support=cv2.erode(support.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
    valid &= support
    count=int(valid.sum())
    if count<100:return None
    output_errors=[float(np.abs(new.astype(np.float32)-warp(old).astype(np.float32)).mean(axis=2)[valid].mean()) for old,new in zip(old_outputs,new_outputs)]
    baseline=float(input_error[valid].mean())
    return dict(valid_pixels=count,input_mae=baseline,output_mae=output_errors,excess_mae=[value-baseline for value in output_errors])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('baseline',type=Path);parser.add_argument('candidate',type=Path);args=parser.parse_args()
    reports=[json.loads((folder/'results.json').read_text()) for folder in [args.baseline,args.candidate]]
    if reports[0]['frames']!=reports[1]['frames']:raise ValueError('Different frame counts')
    captures=[cv2.VideoCapture(str(folder/'comparison.mp4')) for folder in [args.baseline,args.candidate]]
    previous=None;records=[];common=0;input_mismatches=[]
    try:
        for index in range(reports[0]['frames']):
            frames=[]
            for capture in captures:
                ok,frame=capture.read()
                if not ok:raise RuntimeError(f'Missing encoded frame {index}')
                frames.append(frame)
            if frames[0].shape!=frames[1].shape or frames[0].shape[1]%2:
                raise ValueError('Comparison frame dimensions differ or panels are unequal')
            width=frames[0].shape[1]//2
            input_mismatches.append(float(np.abs(frames[0][:,:width].astype(np.float32)-frames[1][:,:width].astype(np.float32)).mean()))
            current=frames[0][:,:width];outputs=[frame[:,width:] for frame in frames]
            if previous is not None and all(reports[k]['records'][j]['face_changed'] for k in range(2) for j in [index-1,index]):
                common+=1
                result=residuals(previous[0],current,previous[1],outputs)
                if result:records.append(dict(frame=index,**result))
            previous=(current,outputs)
    finally:
        for capture in captures:capture.release()
    if not records:raise RuntimeError('No common valid stability samples')
    summary=dict(scope='Approximate optical-flow compensated encoded RGB residual on common consecutive accepted swaps. Forward/backward consistency <1 px, input residual <20/255, common union effect support eroded 5 px. Excludes fallback transitions and occlusion failures. Codec and flow errors remain; not a perceptual pass or calibrated flicker score.',
        common_accepted_pairs=common,evaluated_pairs=len(records),excluded_pairs=reports[0]['frames']-1-common,
        mean_input_mae=float(np.mean([r['input_mae'] for r in records])),
        encoded_original_panel_difference_mean=float(np.mean(input_mismatches)),
        encoded_original_panel_difference_max_frame=float(np.max(input_mismatches)),
        mean_output_mae=[float(np.mean([r['output_mae'][k] for r in records])) for k in range(2)],
        mean_excess_mae=[float(np.mean([r['excess_mae'][k] for r in records])) for k in range(2)],records=records)
    (args.candidate/'flow_stability.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({key:value for key,value in summary.items() if key!='records'}))

if __name__=='__main__':main()
