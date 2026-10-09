"""Inspect changed decisions side by side before promoting temporal tracking."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw

def main():
    parser=argparse.ArgumentParser();parser.add_argument('baseline',type=Path);parser.add_argument('candidate',type=Path);args=parser.parse_args()
    baseline=json.loads((args.baseline/'results.json').read_text());candidate=json.loads((args.candidate/'results.json').read_text())
    if len(baseline['records'])!=len(candidate['records']):raise ValueError('Frame counts differ.')
    gained=[];lost=[]
    for index,(old,new) in enumerate(zip(baseline['records'],candidate['records'])):
        a=old.get('face_changed',old['changed_pixels']>0);b=new.get('face_changed',new['changed_pixels']>0)
        if b and not a:gained.append(index)
        if a and not b:lost.append(index)
    report={'newly_swapped_frames':gained,'newly_fallback_frames':lost,'columns':['original','baseline output','candidate output'],
        'scope':'Selected frames where swap/fallback decisions differ; encoded comparison videos, not a perceptual quality pass.'}
    (args.candidate/'changed_decisions.json').write_text(json.dumps(report,indent=2))
    picks=[]
    for category,indices in [('new swap',gained),('new fallback',lost)]:
        if indices:
            picks += [(indices[i],category) for i in sorted(set(np.linspace(0,len(indices)-1,min(8,len(indices)),dtype=int).tolist()))]
    captures=[cv2.VideoCapture(str(folder/'comparison.mp4')) for folder in [args.baseline,args.candidate]]
    tiles=[]
    for index,category in picks:
        frames=[]
        for capture in captures:
            capture.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=capture.read()
            if not ok:raise RuntimeError(f'Comparison frame missing: {index}')
            frames.append(frame)
        width=frames[0].shape[1]//2
        joined=np.concatenate([frames[0][:,:width],frames[0][:,width:],frames[1][:,width:]],axis=1)
        tile=Image.new('RGB',(960,265),'white');tile.paste(Image.fromarray(cv2.cvtColor(cv2.resize(joined,(960,240)),cv2.COLOR_BGR2RGB)),(0,25))
        ImageDraw.Draw(tile).text((5,5),f'{index}: {category} | ORIGINAL / BASELINE / TRACKED',fill='black');tiles.append(tile)
    for capture in captures:capture.release()
    if tiles:
        sheet=Image.new('RGB',(960,265*len(tiles)),'white')
        for index,tile in enumerate(tiles):sheet.paste(tile,(0,index*265))
        sheet.save(args.candidate/'changed_decisions.jpg')
    print(json.dumps({'newly_swapped':len(gained),'newly_fallback':len(lost),'reviewed_indices':[index for index,_ in picks]}))

if __name__=='__main__':main()
