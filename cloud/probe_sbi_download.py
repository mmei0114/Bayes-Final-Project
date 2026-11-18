"""Bounded bandwidth probes for identical TF wheels; no installation."""
import concurrent.futures as cf
import html
import json
import re
import time
import urllib.request as ur

FILENAME = 'tensorflow-2.15.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'

def probe(source):
    name, index = source
    begin = time.monotonic()
    try:
        if name == 'official':
            with ur.urlopen(index, timeout=12) as r:
                doc = json.load(r)
            url = next(v['url'] for v in doc['urls'] if v['filename'] == FILENAME)
        else:
            with ur.urlopen(index, timeout=12) as r:
                page = r.read().decode()
            url = html.unescape(next(v for v in re.findall(r'href=[\"\']([^\"\']+)', page)
                                     if FILENAME in v)).split('#')[0]
            url = ur.urljoin(index, url) if hasattr(ur, 'urljoin') else __import__('urllib.parse', fromlist=['urljoin']).urljoin(index,url)
        start = time.monotonic()
        with ur.urlopen(ur.Request(url, headers={'Range':'bytes=0-2097151'}), timeout=12) as r:
            payload = r.read(2097152)
        elapsed = time.monotonic()-start
        return dict(source=name, bytes=len(payload), MB_per_second=len(payload)/elapsed/1e6,
                    elapsed_total=time.monotonic()-begin, url=url)
    except Exception as e:
        return dict(source=name, error=str(e), elapsed_total=time.monotonic()-begin)

sources = [('official','https://pypi.org/pypi/tensorflow/2.15.1/json'),
           ('tsinghua','https://pypi.tuna.tsinghua.edu.cn/simple/tensorflow/'),
           ('huawei','https://repo.huaweicloud.com/repository/pypi/simple/tensorflow/')]
with cf.ThreadPoolExecutor(max_workers=3) as pool:
    print(json.dumps(list(pool.map(probe,sources)),indent=2),flush=True)
