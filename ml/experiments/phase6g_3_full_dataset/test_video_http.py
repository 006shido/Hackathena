"""Actual localhost video API round trip, ownership, cleanup and timestamps."""
import argparse,io,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import requests
from PIL import Image


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:8001')
    parser.add_argument('--backend',choices=['trained-model','research-reference'],default='trained-model')
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'video_http_test')
    args=parser.parse_args()
    source=ROOT/'ml/data/celeba/img_align_celeba/197935.jpg'
    target=ROOT/'ml/data/celeba/img_align_celeba/098180.jpg'
    health=requests.get(args.url+'/health',timeout=10)
    assert health.status_code==200
    assert health.json()['model']=='fullres'
    assert health.json()['video_backend']==args.backend
    owner={'X-ML-Owner':'tester'}
    with source.open('rb') as image:
        created=requests.post(args.url+'/video/sessions',headers=owner,files={'source':('source.jpg',image,'image/jpeg')},timeout=30)
    assert created.status_code==200,created.text
    session=created.json()['session_id']
    assert created.json()['backend']==args.backend
    endpoint=args.url+'/video/sessions/'+session
    frame_times=[]
    try:
        with target.open('rb') as image:
            wrong=requests.post(endpoint+'/frame',headers={'X-ML-Owner':'other'},files={'frame':('frame.jpg',image,'image/jpeg')},data={'timestamp_ms':'1'},timeout=10)
        assert wrong.status_code==404
        for timestamp in [1,68,135]:
            with target.open('rb') as image:
                response=requests.post(endpoint+'/frame',headers=owner,files={'frame':('frame.jpg',image,'image/jpeg')},data={'timestamp_ms':str(timestamp)},timeout=30)
            assert response.status_code==200,response.text
            assert response.headers['content-type']=='image/jpeg'
            result=Image.open(io.BytesIO(response.content))
            assert result.size==Image.open(target).size
            assert response.headers['x-face-detected']=='true'
            assert response.headers['x-face-changed']=='true'
            assert response.headers['x-video-backend']==args.backend
            frame_times.append(float(response.headers['x-pipeline-ms']))
            assert frame_times[-1]<2000, 'A frame exceeded the browser request timeout after service readiness.'
        with target.open('rb') as image:
            stale=requests.post(endpoint+'/frame',headers=owner,files={'frame':('frame.jpg',image,'image/jpeg')},data={'timestamp_ms':'135'},timeout=10)
        assert stale.status_code==400
        output=args.output
        output.mkdir(exist_ok=True)
        result.save(output/'frame.png')
        (output/'results.json').write_text(json.dumps(dict(health=health.json(),frame_size=result.size,
            pipeline_ms=float(response.headers['x-pipeline-ms']),all_frame_pipeline_ms=frame_times,
            ownership_check=True,stale_timestamp_rejected=True),indent=2))
    finally:
        deleted=requests.delete(endpoint,headers=owner,timeout=10)
        assert deleted.status_code==200
    with target.open('rb') as image:
        missing=requests.post(endpoint+'/frame',headers=owner,files={'frame':('frame.jpg',image,'image/jpeg')},data={'timestamp_ms':'202'},timeout=10)
    assert missing.status_code==404
    if args.backend=='research-reference':
        bad_source=ROOT/'ml/data/celeba/img_align_celeba/186165.jpg'
        bad_target=ROOT/'ml/data/celeba/img_align_celeba/070860.jpg'
        with bad_source.open('rb') as image:
            rejected_session=requests.post(args.url+'/video/sessions',headers=owner,files={'source':('source.jpg',image,'image/jpeg')},timeout=30)
        assert rejected_session.status_code==200,rejected_session.text
        rejection_endpoint=args.url+'/video/sessions/'+rejected_session.json()['session_id']
        try:
            with bad_target.open('rb') as image:
                rejected=requests.post(rejection_endpoint+'/frame',headers=owner,files={'frame':('frame.jpg',image,'image/jpeg')},data={'timestamp_ms':'1'},timeout=30)
            assert rejected.status_code==200,rejected.text
            assert rejected.headers['x-face-detected']=='true'
            assert rejected.headers['x-face-changed']=='false'
            expected=io.BytesIO();Image.open(bad_target).convert('RGB').save(expected,format='JPEG',quality=90)
            assert rejected.content==expected.getvalue(),'Identity rejection did not return the encoded original camera frame.'
        finally:
            assert requests.delete(rejection_endpoint,headers=owner,timeout=10).status_code==200
    print('PASS: real neural HTTP frames, dimensions, ownership, timestamp rejection and session teardown.')


if __name__=='__main__':main()
