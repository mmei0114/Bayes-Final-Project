"""Independent mathematical checks and prior predictive audit before SBI training."""
import argparse
import json
from pathlib import Path
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.stats import chi2
from sbi_model import PRIOR, DESIGN, prior_sample, unconstrain, constrain, native, generate, sigmoid, log_abs_det_constrain, atomic_json


def checks(output):
    rng=np.random.default_rng(20261219)
    p=prior_sample(rng,2000);x=unconstrain(p);restored=constrain(x)
    error=float(np.max(abs(native(p)-native(restored))))
    assert error < 1e-11
    determinants=[]
    # Differentiate the inverse map numerically in an independent native chart.
    chart=lambda xx: np.r_[constrain(xx)['beta'][:,0],constrain(xx)['beta'][:,1],
                           constrain(xx)['var'],constrain(xx)['pi'][:2]]
    for xx in x[:5]:
        eps=1e-5;eye=np.eye(11)*eps
        J=np.column_stack([(chart(xx+e)-chart(xx-e))/(2*eps) for e in eye])
        actual=np.linalg.slogdet(J)[1];expected=log_abs_det_constrain(xx)
        determinants.append(float(abs(actual-expected)))
    assert max(determinants) < 1e-6
    # Analytic random-intercept moments, not a second implementation of the simulator.
    n=100000;fixed=dict(beta=np.array([[[.3,0.]]*3]),var=np.array([[.1,.5,1.]]),pi=np.array([[.5,.3,.2]]))
    moment_checks={}
    nodes, weights=hermgauss(101);weights=weights/np.sqrt(np.pi)
    for family in ('binary','poisson','nb'):
        d=generate(rng,fixed,family,n=n,T=2);y=d['y'][0];u=d['u'][0]
        count=np.bincount(u,minlength=3);stat=float(((count-n*fixed['pi'][0])**2/(n*fixed['pi'][0])).sum())
        assert stat < chi2.ppf(.9999,2)
        errors=[]
        for k in range(3):
            a=y[u==k].astype(float);v=fixed['var'][0,k]
            if family=='binary':
                prob=sigmoid(.3+np.sqrt(2*v)*nodes);mean=float(prob@weights)
                second=float(prob**2@weights);variance=mean*(1-mean)
            else:
                mean=np.exp(.3+v/2);second=np.exp(.6+2*v)
                variance=mean+(second/5 if family=='nb' else 0)+second-mean**2
            mean_error=abs(a[:,0].mean()-mean)
            assert mean_error < 8*np.sqrt(variance/len(a)),(family,k,mean_error)
            # Within-person covariance verifies that b is shared across time.
            products=(a[:,0]-mean)*(a[:,1]-mean)
            covariance=second-mean**2
            cov_error=abs(products.mean()-covariance)
            assert cov_error < 8*products.std(ddof=1)/np.sqrt(len(a)),(family,k,cov_error)
            errors.append(dict(cluster=k,mean_error=float(mean_error),covariance_error=float(cov_error)))
        moment_checks[family]=dict(class_count_chisquare=stat,per_cluster=errors)
    # Prior predictive settings are assessed independently of the old test datasets.
    audit={}
    pp=prior_sample(rng,512)
    for family in ('binary','poisson','nb'):
        d=generate(rng,pp,family);y=d['y'];unique_counts=(np.eye(3)[d['u']]).sum(1)
        audit[family]=dict(response_quantiles=np.quantile(y,[0,.5,.9,.99,.999,1]).tolist(),
                           dataset_max_quantiles=np.quantile(y.max((1,2)),[.5,.9,.99,1]).tolist(),
                           all_zero_trajectory_fraction=float((y==0).all(-1).mean()),
                           all_one_trajectory_fraction=float((y==1).all(-1).mean()),
                           minimum_cluster_count_quantiles=np.quantile(unique_counts.min(1),[0,.1,.5,1]).tolist())
    result=dict(prior=PRIOR,design=DESIGN,transform_roundtrip_max_error=error,
                inverse_log_jacobian_max_error=max(determinants),moment_checks=moment_checks,
                prior_predictive=audit,seed=20261219,passed=True)
    atomic_json(output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path('qa/sbi_pilot/model_checks.json'))
    checks(ap.parse_args().output)
