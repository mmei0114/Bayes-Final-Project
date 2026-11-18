"""Held-out diagnostics for the SBI pilot, never used to select checkpoints."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','1')
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import tensorflow as tf
import inference as core
from sbi_model import constrain,native,features,generate,atomic_json
from sbi_networks import make_models
from sbi_pilot import configure,load_split


def restore(output):
    cfg=json.loads((output/'configuration.json').read_text())
    for name,value in cfg['sources'].items():
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==value,name
    selection=json.loads((output/'selection.json').read_text())
    posterior,classifier=make_models(cfg['network_seed'])
    npe=output/f"checkpoint_{selection['best_npe_step']:06d}/state"
    cls=output/f"checkpoint_{selection['best_classifier_step']:06d}/state"
    tf.train.Checkpoint(posterior=posterior).restore(str(npe)).expect_partial()
    tf.train.Checkpoint(classifier=classifier).restore(str(cls)).expect_partial()
    standard=json.loads((output/'standardization.json').read_text())
    return posterior,classifier,np.array(standard['mean'],np.float32),np.array(standard['scale'],np.float32)


def evaluate(data,output,draws=512):
    begin=time.monotonic();posterior,classifier,mean,scale=restore(output)
    d,_=load_split(data,'calibration');rng=np.random.default_rng(20262018)
    n=len(d['theta']);samples=[];probs=[]
    for first in range(0,n,8):
        last=min(n,first+8);x=features(d['y'][first:last])
        base=rng.normal(size=(last-first,draws,11)).astype(np.float32)
        z=posterior.sample(x,base).numpy();samples.append(z*scale+mean)
        total=np.zeros((last-first,300,3),dtype=np.float64)
        # Bounded S x dataset expansion prevents GPU exhaustion at prediction time.
        for s in range(0,draws,32):
            zz=z[:,s:s+32];repeat=zz.shape[1]
            xx=np.repeat(x,repeat,axis=0);theta=zz.reshape(-1,11)
            logits=classifier((xx,theta),False)
            pp=tf.nn.softmax(logits,axis=-1).numpy().reshape(last-first,repeat,300,3)
            total+=pp.sum(1)
        probs.append(total/draws)
    samples=np.concatenate(samples);probs=np.concatenate(probs)
    p=constrain(samples);true=constrain(d['theta']);values=native(p);truth=native(true)
    if not np.isfinite(values).all():raise RuntimeError('Non-finite posterior samples')
    lo,med,hi=np.quantile(values,[.025,.5,.975],axis=1)
    ranks=(values<truth[:,None,:]).sum(1)
    cover95=((truth>=lo)&(truth<=hi)).mean(0)
    low50,high50=np.quantile(values,[.25,.75],axis=1)
    cover50=((truth>=low50)&(truth<=high50)).mean(0)
    prior,_=load_split(data,'train');prior_values=native(constrain(prior['theta']))
    prior_variance=prior_values.var(0,ddof=1)
    contraction=1-values.var(1,ddof=1).mean(0)/prior_variance
    rmse=np.sqrt(((values.mean(1)-truth)**2).mean(0))
    labels=np.argmax(probs,axis=-1);brier=np.mean(np.sum((probs-np.eye(3)[d['u']])**2,axis=-1))
    expected=np.maximum(probs[np.arange(n)[:,None],np.arange(300),d['u']],1e-30)
    avg_ce=float(-np.log(expected).mean())
    # Oracle integration isolates conditional classifier error from NPE error.
    cd,_=load_split(data,'classifier');targets=constrain(cd['theta']);x=features(cd['y'])
    pred=tf.nn.softmax(classifier((x,(cd['theta']-mean)/scale),False),axis=-1).numpy()
    oracle=[];quad=[]
    for i in range(len(x)):
        raw=dict(y=cd['y'][i],t=np.broadcast_to(np.linspace(-1,1,20),(300,20)),
                 mask=np.ones((300,20)),family='binary')
        prepared=core.prepared(raw);param={k:v[i] for k,v in targets.items()}
        ell,r,_,_,_=core.estep(prepared,param,61)
        high,r_high,_,_,_=core.estep(prepared,param,101)
        quad.append(dict(loglik_difference=float(abs(ell-high)),max_responsibility_difference=float(abs(r-r_high).max())))
        oracle.append(r_high)
    oracle=np.array(oracle)
    oracle_kl=float(np.mean(np.sum(oracle*(np.log(np.maximum(oracle,1e-30))-np.log(np.maximum(pred,1e-30))),axis=-1)))
    assert max(v['loglik_difference'] for v in quad)<1e-6
    # Predictive check for a representative prior-predictive dataset: new people/latent effects.
    pp_params={k:v[0,:64] for k,v in p.items()}
    rep=generate(rng,pp_params,'binary')['y']
    predictive=dict(observed_mean_by_time=d['y'][0].mean(0).tolist(),
                    replicated_mean_by_time_quantiles=np.quantile(rep.mean(1),[.05,.5,.95],axis=0).tolist())
    names=[f'beta{k}_{j}' for k in range(1,4) for j in ('intercept','slope')]
    names += [f'D{k}' for k in range(1,4)]+[f'pi{k}' for k in range(1,4)]
    result=dict(stage='pilot only; no full parameter-posterior MCMC reference yet',datasets=n,draws=draws,
                names=names,coverage_95=cover95.tolist(),coverage_50=cover50.tolist(),
                prior_scaled_RMSE=(rmse/np.sqrt(prior_variance)).tolist(),contraction=contraction.tolist(),
                marginal_rank_histograms=[np.histogram(ranks[:,j],bins=np.linspace(-.5,draws+.5,17))[0].tolist() for j in range(12)],
                sbc_interpretation='128 held-out datasets; pointwise histograms have limited power, not proof of correctness',
                posterior_cluster_brier=float(brier),posterior_cluster_cross_entropy=avg_ce,
                label_accuracy=float((labels==d['u']).mean()),
                conditional_classifier=dict(datasets=len(x),probability_RMSE=float(np.sqrt(np.mean((pred-oracle)**2))),
                                            average_KL_from_quadrature=oracle_kl,
                                            argmax_agreement=float((pred.argmax(-1)==oracle.argmax(-1)).mean()),
                                            quadrature_checks=quad),posterior_predictive=predictive,
                seconds=time.monotonic()-begin,original_formal_test_data_not_read=True)
    atomic_json(output/'evaluation.json',result)
    temp=output/'posterior_samples.npz.tmp'
    with temp.open('wb') as f:
        np.savez_compressed(f,unconstrained=samples,native=values,true=truth,ranks=ranks,
                            probabilities=probs,conditional_probabilities=pred,quadrature_probabilities=oracle)
    temp.replace(output/'posterior_samples.npz')
    print(json.dumps({k:v for k,v in result.items() if k not in ('conditional_classifier','posterior_predictive','marginal_rank_histograms')},indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--draws',type=int,default=512);args=ap.parse_args()
    configure();evaluate(args.data,args.output,args.draws)
