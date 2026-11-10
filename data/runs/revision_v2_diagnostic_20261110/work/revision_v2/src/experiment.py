import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
import argparse,json,sys,platform,hashlib,time,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from inference import generate,fit_profile,fit_mcem,evaluate

ROOT=Path(__file__).resolve().parents[1]
SCENARIOS={'baseline':{},'hard':{'separation':.5},'original':{'original':True},'n100':{'n':100},'n1000':{'n':1000},'T10':{'T':10},'T50':{'T':50},'K2':{'K':2},'K4':{'K':4},'var01':{'variance':.1},'var15':{'variance':1.5},'poisson':{'family':'poisson'},'nb':{'family':'nb'},'irregular':{'irregular':True}}

def atomic(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    with temp.open('w') as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

def execute(task):
    output,name,rep,method,starts,maxiter=task;output=Path(output)
    seed=20261110+10000*list(SCENARIOS).index(name)+rep;cfg=SCENARIOS[name];data=generate(seed,**cfg)
    tag=f'{name}_{rep}_{method}';ckpt=output/f'{tag}.checkpoint.json'
    def save(v):atomic(ckpt,dict(scenario=name,rep=rep,method=method,seed=seed,state=v))
    try:
        run=(fit_mcem(data,cfg.get('K',3),seed+200000,starts,maxiter,save) if method=='MCEM' else
             fit_profile(data,cfg.get('K',3),method,seed+200000,starts,maxiter,checkpoint=save))
        metrics=evaluate(data,run,cfg.get('K',3))
        atomic(output/f'{tag}.json',dict(scenario=name,rep=rep,method=method,seed=seed,config=cfg,metrics=metrics,fit=run))
        return tag,metrics
    except Exception:
        atomic(output/f'{tag}.error.json',dict(traceback=traceback.format_exc(),seed=seed))
        return tag,dict(error=traceback.format_exc())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reps',type=int,default=10);ap.add_argument('--starts',type=int,default=3)
    ap.add_argument('--jobs',type=int,default=8);ap.add_argument('--maxiter',type=int,default=300)
    ap.add_argument('--scenarios',nargs='+',default=list(SCENARIOS));ap.add_argument('--methods',nargs='+',default=['QML','GVA','MCEM'])
    ap.add_argument('--output',default=str(ROOT/'results'/'main'));args=ap.parse_args()
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    sources={name:hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest() for name in ('inference.py','legacy_model.py','experiment.py')}
    spec=dict(reps=args.reps,starts=args.starts,maxiter=args.maxiter,scenarios=args.scenarios,methods=args.methods,source=sources)
    mf=output/'metadata.json'
    if mf.exists() and json.loads(mf.read_text())['spec']!=spec:raise RuntimeError('Configuration/source changed: use a new output directory')
    if not mf.exists():atomic(mf,dict(spec=spec,python=sys.version,numpy=np.__version__,platform=platform.platform(),jobs=args.jobs))
    tasks=[]
    for name in args.scenarios:
        for rep in range(args.reps):
            seed=20261110+10000*list(SCENARIOS).index(name)+rep;data=generate(seed,**SCENARIOS[name])
            datapath=output/f'{name}_{rep}_data.npz'
            if not datapath.exists():
                temp=datapath.with_suffix('.npz.tmp')
                with temp.open('wb') as f:np.savez_compressed(f,**data)
                os.replace(temp,datapath)
            for method in args.methods:
                if not (output/f'{name}_{rep}_{method}.json').exists():tasks.append((str(output),name,rep,method,args.starts,args.maxiter))
    failures=0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        fs=[pool.submit(execute,t) for t in tasks]
        for f in as_completed(fs):
            tag,metric=f.result();failures+=int('error' in metric)
            print(tag,{k:round(float(v),4) if isinstance(v,(float,np.floating)) else v for k,v in metric.items() if k!='confusion'},flush=True)
    expected=args.reps*len(args.methods)*len(args.scenarios)
    complete=sum((output/f'{name}_{rep}_{method}.json').exists() for name in args.scenarios for rep in range(args.reps) for method in args.methods)
    atomic(output/'STATUS.json',dict(expected=expected,complete=complete,failed_in_this_run=failures,status='COMPLETE' if complete==expected else 'INCOMPLETE'))
    if complete!=expected:raise SystemExit(1)

if __name__=='__main__':main()
