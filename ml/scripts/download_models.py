"""Download and verify the research release's runtime model assets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2]

def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accept-research-terms', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if not args.verify_only and not args.accept_research_terms:
        parser.error('Read MODEL_NOTICES.md, then pass --accept-research-terms for noncommercial research use.')
    manifest = json.loads((ROOT / 'ml/model-manifest.json').read_text())
    for asset in manifest['assets']:
        target = ROOT / asset['path']
        if target.exists():
            if digest(target) != asset['sha256']:
                raise SystemExit(f'Existing file checksum mismatch: {target}; file was left untouched.')
            print(f"Verified {asset['name']}")
            continue
        if args.verify_only:
            raise SystemExit(f'Missing model: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.download')
        request = urllib.request.Request(asset['url'], headers={'User-Agent': 'DeepTrace-research-setup'})
        print(f"Downloading {asset['name']} ({asset['bytes'] / 1024**2:.1f} MiB)")
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if digest(temporary) != asset['sha256']:
            raise SystemExit(f'Download checksum mismatch: {temporary}; not installed.')
        os.replace(temporary, target)
    print('All runtime model assets verified.')

if __name__ == '__main__':
    main()
