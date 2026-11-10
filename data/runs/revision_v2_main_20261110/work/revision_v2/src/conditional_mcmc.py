"""Fixed-parameter Gibbs (U) / independent-t Metropolis (b) comparison.

This targets p(U,b|Y,theta_hat), NOT a parameter posterior. No extra packages.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp,gammaln
from inference import prepared,mode,ll,estep,local_vi
from experiment import atomic

def split_rhat(x):
    # x[chains,draws,subjects]; classical split R-hat, not rank-normalized.
    C,L,N=x.shape;H=L//2;v=np.concatenate([x[:,:H],x[:,-H:]],axis=0)
    W=v.var(1,ddof=1).mean(0);B=H*v.mean(1).var(0,ddof=1)
    return np.sqrt(((H-1)/H*W+B/H)/W)

def compare(data,p,seed,draws=8000,burn=2000,subjects=30,samples=None):
    begin=time.perf_counter();n=min(subjects,len(data['y']));indices=np.arange(n)
    d=prepared({k:(v[indices] if isinstance(v,np.ndarray) and v.ndim and v.shape[0]==len(data['y']) and k not in ('beta','variance','pi') else v) for k,v in data.items()})
    K=len(p['pi']);C=4;rng=np.random.default_rng(seed)
    means=[];scales=[]
    for k in range(K):
        m,v=mode(d,p['beta'][k],p['var'][k]);means.append(m);scales.append(np.sqrt(v*3/5))
    means=np.array(means).T;scales=np.array(scales).T
    u=np.tile(np.arange(C)%K,(n,1));b=np.take_along_axis(means,u,axis=1)
    bdraw=np.empty((C,draws,n));udraw=np.empty((C,draws,n),np.int8);accepted=0
    logprior=np.log(p['pi']);dfconst=gammaln(3)-gammaln(2.5)-.5*np.log(5*np.pi)
    for it in range(burn+draws):
        logjoint=np.stack([ll(d,p['beta'][k],b)+logprior[k]-.5*(np.log(2*np.pi*p['var'][k])+b*b/p['var'][k]) for k in range(K)],axis=2)
        prob=np.exp(logjoint-logsumexp(logjoint,axis=2,keepdims=True))
        u=np.minimum((rng.random((n,C,1))>np.cumsum(prob,axis=2)).sum(2),K-1)
        m=np.take_along_axis(means,u,axis=1);s=np.take_along_axis(scales,u,axis=1)
        z=rng.standard_t(5,(n,C));trial=m+s*z
        oldq=dfconst-np.log(s)-3*np.log1p(((b-m)/s)**2/5)
        newq=dfconst-np.log(s)-3*np.log1p(z*z/5)
        ratio=np.zeros_like(b)
        for k in range(K):
            val=ll(d,p['beta'][k],trial)-ll(d,p['beta'][k],b)-(trial*trial-b*b)/(2*p['var'][k])
            ratio=np.where(u==k,val,ratio)
        accept=np.log(rng.random((n,C)))<np.minimum(0,ratio+oldq-newq)
        accepted+=accept.sum();b=np.where(accept,trial,b)
        if it>=burn:bdraw[:,it-burn]=b.T;udraw[:,it-burn]=u.T
    L,r,bs,ws,_=estep(d,p,101)
    exactm=sum(r[:,k]*(ws[k]*bs[k]).sum(1) for k in range(K))
    exactv=sum(r[:,k]*(ws[k]*bs[k]**2).sum(1) for k in range(K))-exactm**2
    Fs=[];ms=[];ss=[]
    for k in range(K):
        F,m,s,_,_=local_vi(d,p['beta'][k],p['var'][k]);Fs.append(F);ms.append(m);ss.append(s)
    scores=np.array(Fs).T+logprior;rv=np.exp(scores-logsumexp(scores,axis=1)[:,None])
    vm=(rv*np.array(ms).T).sum(1)
    vv=(rv*(np.array(ss).T**2+np.array(ms).T**2)).sum(1)-vm**2
    rm=np.array([(udraw==k).mean((0,1)) for k in range(K)]).T
    mcmean=bdraw.mean((0,1));mcvar=bdraw.var((0,1),ddof=1)
    batches=bdraw.reshape(C,draws//100,100,n).mean(2).reshape(-1,n)
    mcse=batches.std(0,ddof=1)/np.sqrt(len(batches))
    if samples is not None:
        dest=Path(samples);tmp=dest.with_suffix('.npz.tmp')
        with tmp.open('wb') as f:np.savez_compressed(f,b=bdraw,u=udraw,indices=indices,**p)
        os.replace(tmp,dest)
    return dict(seed=seed,subjects=n,chains=C,burn=burn,draws=draws,
        acceptance=float(accepted/((draws+burn)*n*C)),max_split_rhat=float(np.max(split_rhat(bdraw))),
        mean_mcse=float(mcse.mean()),mcmc_mean_rmse=float(np.sqrt(np.mean((mcmean-exactm)**2))),
        gva_mean_rmse=float(np.sqrt(np.mean((vm-exactm)**2))),
        mcmc_responsibility_rmse=float(np.sqrt(np.mean((rm-r)**2))),
        gva_responsibility_rmse=float(np.sqrt(np.mean((rv-r)**2))),
        mcmc_variance_ratio=float(mcvar.sum()/exactv.sum()),gva_variance_ratio=float(vv.sum()/exactv.sum()),
        seconds=time.perf_counter()-begin,exact_mean=exactm.tolist(),mcmc_mean=mcmean.tolist(),gva_mean=vm.tolist())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',required=True);ap.add_argument('--scenario',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    results=Path(args.results);record=json.loads((results/f'{args.scenario}_0_QML.json').read_text())
    p={k:np.array(v) for k,v in record['fit']['params'].items()};data=dict(np.load(results/f'{args.scenario}_0_data.npz'));data['family']=str(data['family'])
    out=compare(data,p,20261110+900000+list(('baseline','poisson','nb')).index(args.scenario),samples=Path(args.output).with_suffix('.chains.npz'))
    atomic(args.output,out);print(json.dumps({k:v for k,v in out.items() if not k.endswith('_mean')},indent=2))

if __name__=='__main__':main()
