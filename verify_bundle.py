#!/usr/bin/env python3
"""Check this locally supplied package; no downloads or malware execution."""
from pathlib import Path
import hashlib,json,sys
root=Path(__file__).resolve().parent
manifest=root/'SHA256_manifest.json'
if not manifest.is_file():
    raise SystemExit('Missing SHA256_manifest.json. Extract the entire ZIP, not only its .py file.')
failures=[]
for entry in json.loads(manifest.read_text())['files']:
    path=(root/entry['path']).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        failures.append(entry['path']+': missing/invalid path')
        continue
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):
            digest.update(chunk)
    if digest.hexdigest()!=entry['sha256']:
        failures.append(entry['path']+': checksum mismatch')
if failures:
    raise SystemExit('Package verification failed. Re-extract a clean copy:\n'+'\n'.join(failures))
print('Package verified: exact uploaded Disk, Memory and Network workbooks; script and guide.',flush=True)
