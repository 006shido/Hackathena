"""Download only the researchers' published ASVspoof evaluation showcase WAVs.

The page labels—not file names or detector scores—supply ground truth. These
curated examples are NOT the full evaluation corpus or independent training data.
"""
import hashlib,json,re,urllib.request,urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
BASE='https://nii-yamagishilab.github.io/samples-xin/'

def main():
    root=Path(__file__).parent;page=(root/'assets/asvspoof_samples_page.html').read_text()
    table=re.findall(r'<table\b[\s\S]*?</table>',page)[0]
    rows=[]
    for row_index,row in enumerate(re.findall(r'<tr\b[\s\S]*?</tr>',table)):
        sources=re.findall(r'<source\s+src="([^"]+)"',row)
        if not sources:continue
        if len(sources)!=15:raise ValueError('Unexpected published table columns; refusing to guess labels.')
        for column,source in enumerate(sources):
            rows.append(dict(row=row_index,column=column,url=BASE+source,label='real' if column in (10,14) else 'spoof',upstream_path=source))
    out=root/'audio_samples';out.mkdir(exist_ok=True)
    def download(row):
        name=row['upstream_path'].replace('/','__');path=out/name
        if not path.exists():
            try:
                with urllib.request.urlopen(row['url'],timeout=10) as response:data=response.read(5*1024*1024+1)
            except (urllib.error.URLError,TimeoutError) as error:
                return dict(**row,file=name,error=str(error))
            if len(data)>5*1024*1024 or data[:4] not in (b'RIFF',b'RF64'):raise ValueError('Unexpected WAV response')
            path.write_bytes(data)
        return dict(**row,file=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    unique={row['url']:row for row in rows}
    with ThreadPoolExecutor(max_workers=4) as pool:downloads={row['url']:row for row in pool.map(download,unique.values())}
    records=[dict(**row,**{k:v for k,v in downloads[row['url']].items() if k not in row}) for row in rows]
    report=dict(source_page=BASE+'main-asvspoof2019',license='Open Data Commons Attribution License; see source page and ASVspoof2019 database',
        scope='Curated published LA evaluation-set showcase, not random full benchmark; some original recordings repeat.',records=records)
    (out/'manifest.json').write_text(json.dumps(report,indent=2));print(f'Processed {len(records)} labeled references; {sum("error" in r for r in records)} unavailable references preserved in manifest.')

if __name__=='__main__':main()
