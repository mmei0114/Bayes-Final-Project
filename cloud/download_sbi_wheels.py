"""Download verified GPU wheels with bounded parallel HTTP range requests.

Pure standard library; PyPI publishes artifact SHA256/size. No code is executed.
"""
import concurrent.futures as cf
import hashlib
import json
import os
import re
import sys
import time
import urllib.request as ur
from pathlib import Path

DEST=Path(sys.argv[1]);DEST.mkdir(parents=True,exist_ok=True)
MIRROR='https://pypi.tuna.tsinghua.edu.cn'


def metadata(name,version):
    with ur.urlopen(f'https://pypi.org/pypi/{name}/{version}/json',timeout=20) as r:
        return json.load(r)


def artifact(doc):
    matches=[v for v in doc['urls'] if v['filename'].endswith('.whl') and
             ('manylinux' in v['filename']) and ('x86_64' in v['filename']) and
             ('cp311-cp311' in v['filename'] or 'py3-none-' in v['filename'])]
    if not matches: raise ValueError('No supported wheel: '+doc['info']['name'])
    return min(matches,key=lambda x:x['size'])


def download(item):
    path=DEST/item['filename'];size=item['size'];expected=item['digests']['sha256']
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==expected:
        return dict(file=path.name,status='cached',bytes=size)
    url=MIRROR+__import__('urllib.parse',fromlist=['urlparse']).urlparse(item['url']).path
    chunks=8;part=(size+chunks-1)//chunks;temp=path.with_suffix(path.suffix+'.tmp')
    fd=os.open(temp,os.O_RDWR|os.O_CREAT,0o600);os.ftruncate(fd,size);begin=time.monotonic()
    def piece(index):
        lo=index*part;hi=min(size,(index+1)*part)-1
        if lo>=size:return
        req=ur.Request(url,headers={'Range':f'bytes={lo}-{hi}'})
        for attempt in range(2):
            try:
                with ur.urlopen(req,timeout=25) as r:
                    if r.status!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {lo}-'):
                        raise ValueError('Server does not honor ranges')
                    offset=lo
                    while offset<=hi:
                        data=r.read(min(1024*1024,hi-offset+1))
                        if not data:raise IOError('Truncated range')
                        written=os.pwrite(fd,data,offset)
                        if written!=len(data):raise IOError('Short disk write')
                        offset+=written
                return
            except Exception:
                if attempt:raise
    try:
        with cf.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(piece,range(chunks)))
    finally:os.close(fd)
    sha=hashlib.sha256(temp.read_bytes()).hexdigest()
    if sha!=expected:raise ValueError('SHA256 mismatch: '+path.name)
    temp.replace(path)
    result=dict(file=path.name,bytes=size,seconds=time.monotonic()-begin,sha256=sha,status='verified')
    print(json.dumps(result),flush=True);return result


if len(sys.argv)>2:
    items=json.loads(Path(sys.argv[2]).read_text())
else:
    base=metadata('tensorflow','2.15.1');docs=[base]
    deps=[]
    for dep in base['info']['requires_dist']:
        if dep.startswith('nvidia-') and 'and-cuda' in dep:
            match=re.match(r'([A-Za-z0-9_-]+)\s*\(?\s*==\s*([0-9.]+)',dep)
            if not match:raise ValueError(dep)
            deps.append(match.groups())
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        docs+=list(pool.map(lambda pair:metadata(*pair),deps))
    items=[artifact(d) for d in docs]
(DEST/'artifacts.json').write_text(json.dumps(items,indent=2)+'\n')
print(json.dumps(dict(files=len(items),bytes=sum(v['size'] for v in items))),flush=True)
if os.environ.get('SBI_METADATA_ONLY')!='1':
    with cf.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(download,items))
    (DEST/'verified.json').write_text(json.dumps(results,indent=2)+'\n')
