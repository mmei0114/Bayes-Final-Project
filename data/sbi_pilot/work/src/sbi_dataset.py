"""Deterministic, lossless prior-predictive shards for the fixed-design SBI pilot."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from sbi_model import prior_sample,unconstrain,generate,PRIOR,DESIGN,atomic_json


def shard(task):
    destination,index,size,seed=task;destination=Path(destination)
    path=destination/f'shard_{index:04d}.npz'
    if path.exists():return path.name
    rng=np.random.default_rng(np.random.SeedSequence([seed,index]))
    p=prior_sample(rng,size);z=unconstrain(p).astype(np.float32);d=generate(rng,p)
    temp=path.with_suffix('.npz.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f,y=d['y'],u=d['u'],theta=z,
                            beta=p['beta'],var=p['var'],pi=p['pi'])
    temp.replace(path)
    return path.name


def create(output,split,size,seed,jobs,chunk=1000):
    destination=Path(output)/split;destination.mkdir(parents=True,exist_ok=True)
    begin=time.monotonic()
    spec=dict(split=split,size=size,seed=seed,chunk=chunk,prior=PRIOR,design=DESIGN,
              simulator_sha256=hashlib.sha256(Path(__file__).with_name('sbi_model.py').read_bytes()).hexdigest())
    if (destination/'spec.json').exists():assert json.loads((destination/'spec.json').read_text())==spec
    else:atomic_json(destination/'spec.json',spec)
    tasks=[(str(destination),i,min(chunk,size-i*chunk),seed) for i in range((size+chunk-1)//chunk)]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for name in pool.map(shard,tasks):print(split,name,flush=True)
    files={name:hashlib.sha256((destination/name).read_bytes()).hexdigest()
           for name in sorted(t[1] and f'shard_{t[1]:04d}.npz' or 'shard_0000.npz' for t in tasks)}
    atomic_json(destination/'manifest.json',dict(spec=spec,files=files,seconds=time.monotonic()-begin,
                                              status='COMPLETE'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--size',type=int,default=100000);ap.add_argument('--jobs',type=int,default=6)
    args=ap.parse_args()
    for split,size,seed in (('train',args.size,20261318),('validation',512,20261418),
                            ('calibration',128,20261518),('classifier',32,20261618)):
        create(args.output,split,size,seed,args.jobs)
