"""Fetch the official OpenSLR regression subset; bounded, checksum-verified extraction."""
import hashlib,json,tarfile,urllib.request,time
from pathlib import Path

def main():
    root=Path(__file__).parent/'call_domain/natural';root.mkdir(parents=True,exist_ok=True)
    url='https://openslr.trmal.net/resources/31/dev-clean-2.tar.gz'
    archive=root/'dev-clean-2.tar.gz';expected='6d7ab67ac6a1d2c993d050e16d61080d'
    if not archive.exists():
        partial=archive.with_suffix('.partial');total=0;last=time.time()
        with urllib.request.urlopen(url,timeout=30) as response,partial.open('wb') as output:
            while chunk:=response.read(1024*1024):
                total+=len(chunk)
                if total>150*1024*1024:raise ValueError('Unexpected archive size')
                output.write(chunk)
                if time.time()-last>10:print(f'Downloaded {total//1048576} MiB',flush=True);last=time.time()
        if hashlib.md5(partial.read_bytes()).hexdigest()!=expected:raise ValueError('Official archive checksum mismatch')
        partial.replace(archive)
    if hashlib.md5(archive.read_bytes()).hexdigest()!=expected:raise ValueError('Archive checksum mismatch')
    counts={};records=[]
    with tarfile.open(archive,'r:gz') as source:
        for member in source:
            if not member.isfile() or not member.name.endswith('.flac'):continue
            speaker=Path(member.name).name.split('-')[0]
            if counts.get(speaker,0)>=4:continue
            if member.size>10*1024*1024:raise ValueError('Unexpected audio file size')
            data=source.extractfile(member).read();name=Path(member.name).name
            (root/name).write_bytes(data);counts[speaker]=counts.get(speaker,0)+1
            records.append(dict(file=name,speaker='librispeech-'+speaker,sha256=hashlib.sha256(data).hexdigest()))
    (root/'manifest.json').write_text(json.dumps(dict(source=url,source_page='https://www.openslr.org/31/',license='CC BY 4.0',archive_md5=expected,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),selection='First four utterances per speaker in archive order',records=records),indent=2))
    print(f'Extracted {len(records)} clips across {len(counts)} speakers',flush=True)

if __name__=='__main__':main()
