"""Summarize actual frame decisions and selected visuals without rerunning inference."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw

def stretches(values):
    result=[];start=None
    for index,value in enumerate(values+[False]):
        if value and start is None:start=index
        elif not value and start is not None:result.append([start,index-1,index-start]);start=None
    return sorted(result,key=lambda value:value[2],reverse=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args()
    data=json.loads((args.directory/'results.json').read_text());records=data['records']
    changed=[bool(record.get('face_changed',record['changed_pixels']>0)) for record in records]
    rejected=[record.get('identity_accepted') is False for record in records]
    missing=[not record['face_detected'] for record in records]
    decisions=[('swap' if changed[i] else 'identity rejected' if rejected[i] else 'face missing') for i in range(len(records))]
    transitions=[i for i in range(1,len(decisions)) if decisions[i]!=decisions[i-1]]
    summary={'frames':len(records),'changed_frames':sum(changed),'identity_rejected_frames':sum(rejected),'missing_face_frames':sum(missing),
        'decision_transitions':len(transitions),'transition_frames':transitions,'longest_identity_rejections':stretches(rejected)[:8],
        'longest_missing_face_runs':stretches(missing)[:8],
        'scope':'Observed identity/detection decisions and selected encoded comparison frames; not all-frame perceptual quality or a flicker score.'}
    accepted=[record for i,record in enumerate(records) if changed[i] and 'source_similarity' in record]
    if accepted:
        summary['accepted_source_similarity_mean']=float(np.mean([record['source_similarity'] for record in accepted]))
        summary['accepted_source_similarity_min']=min(record['source_similarity'] for record in accepted)
        summary['accepted_source_minus_target_min']=min(record['source_similarity']-record['target_similarity'] for record in accepted)
    (args.directory/'decision_summary.json').write_text(json.dumps(summary,indent=2))
    picks=set(np.linspace(0,len(records)-1,12,dtype=int).tolist())
    for index in transitions[:8]:picks.update([index-1,index])
    picks=sorted(picks)
    capture=cv2.VideoCapture(str(args.directory/'comparison.mp4'));tiles=[]
    for index in picks:
        capture.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=capture.read()
        if not ok:raise RuntimeError(f'Missing comparison frame {index}')
        tile=Image.new('RGB',(640,265),'white');tile.paste(Image.fromarray(cv2.cvtColor(cv2.resize(frame,(640,240)),cv2.COLOR_BGR2RGB)),(0,25))
        record=records[index];label=f"{index}: {decisions[index]}"
        if 'source_similarity' in record:label+=f" A={record['source_similarity']:.3f}"
        ImageDraw.Draw(tile).text((4,5),label,fill='black');tiles.append(tile)
    capture.release();sheet=Image.new('RGB',(1280,265*((len(tiles)+1)//2)),(220,220,220))
    for i,tile in enumerate(tiles):sheet.paste(tile,((i%2)*640,(i//2)*265))
    sheet.save(args.directory/'decision_contact.jpg');print(json.dumps(summary))

if __name__=='__main__':main()
