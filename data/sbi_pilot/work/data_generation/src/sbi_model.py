"""Prior-predictive mixture GLMM simulator and standard constraint transforms.

The likelihood is identical to inference.py. The ordered/centered stick-breaking
coordinates follow the Amortized Bayesian Mixture Models author implementation;
the bounded-variance logit is a standard interval transform. No training dependency.
"""
import json
from pathlib import Path

import numpy as np

PRIOR = dict(beta_normal_sd=1.5, logvar_normal_mean=float(np.log(.5)),
             logvar_normal_sd=.7, variance_lower=.005, variance_upper=5.,
             dirichlet_alpha=[2., 2., 2.], intercept_order='ascending',
             coefficient_support='real, proper Normal prior; not ML optimization bounds')
DESIGN = dict(n=300, T=20, K=3, times=[-1., 1.], nb_size=5.)


def sigmoid(x):
    return np.exp(-np.logaddexp(0., -np.asarray(x)))


def prior_sample(rng, batch):
    beta = rng.normal(0., PRIOR['beta_normal_sd'], (batch, 3, 2))
    lo, hi = np.log(PRIOR['variance_lower']), np.log(PRIOR['variance_upper'])
    logvar = rng.normal(PRIOR['logvar_normal_mean'], PRIOR['logvar_normal_sd'], (batch, 3))
    bad = (logvar <= lo) | (logvar >= hi)
    while bad.any():
        logvar[bad] = rng.normal(PRIOR['logvar_normal_mean'], PRIOR['logvar_normal_sd'], bad.sum())
        bad = (logvar <= lo) | (logvar >= hi)
    var = np.exp(logvar)
    pi = rng.dirichlet(PRIOR['dirichlet_alpha'], batch)
    order = np.argsort(beta[:, :, 0], axis=1)
    return dict(beta=np.take_along_axis(beta, order[:, :, None], axis=1),
                var=np.take_along_axis(var, order, axis=1),
                pi=np.take_along_axis(pi, order, axis=1))


def unconstrain(p):
    b = np.asarray(p['beta']); v = np.asarray(p['var']); pi = np.asarray(p['pi'])
    gap = np.diff(b[..., 0], axis=-1)
    if np.any(gap <= 0) or np.any(pi <= 0):
        raise ValueError('Need ordered intercepts and interior simplex')
    x = np.concatenate((b[..., :1, 0], np.log(gap), b[..., 1]), axis=-1)
    lo, hi = PRIOR['variance_lower'], PRIOR['variance_upper']
    if np.any(v <= lo) or np.any(v >= hi):
        raise ValueError('Variance outside open bounds')
    zv = np.log(v-lo)-np.log(hi-v)
    remaining = np.concatenate((np.ones_like(pi[..., :1]), 1.-np.cumsum(pi[..., :-2], axis=-1)), axis=-1)
    stick = pi[..., :-1]/remaining
    adjust = np.log(np.array([.5, 1.]))
    zp = np.log(stick)-np.log1p(-stick)-adjust
    return np.concatenate((x, zv, zp), axis=-1)


def constrain(x):
    x = np.asarray(x, dtype=np.float64)
    if x.shape[-1] != 11 or not np.isfinite(x).all():
        raise ValueError('Expected finite unconstrained 11-vector')
    with np.errstate(over='raise', invalid='raise'):
        intercept = np.concatenate((x[..., :1], x[..., :1]+np.cumsum(np.exp(x[..., 1:3]),axis=-1)),axis=-1)
        beta = np.stack((intercept, x[..., 3:6]),axis=-1)
        lo, hi = PRIOR['variance_lower'], PRIOR['variance_upper']
        var = lo+(hi-lo)*sigmoid(x[..., 6:9])
        stick = sigmoid(x[..., 9:11]+np.log(np.array([.5, 1.])))
        pi = np.stack((stick[...,0],(1-stick[...,0])*stick[...,1],
                       (1-stick[...,0])*(1-stick[...,1])),axis=-1)
    return dict(beta=beta, var=var, pi=pi)


def log_abs_det_constrain(x):
    """log |d(beta0,beta1,D,pi_1:2)/dx| in the canonical 11-dimensional chart."""
    x = np.asarray(x)
    log_sig = lambda z: -np.logaddexp(0.,-z)
    lv = (np.log(PRIOR['variance_upper']-PRIOR['variance_lower'])+
          log_sig(x[...,6:9])+log_sig(-x[...,6:9])).sum(-1)
    a = x[...,9:11]+np.log(np.array([.5,1.]))
    lp = (log_sig(a)+log_sig(-a)).sum(-1)+log_sig(-a[...,0])
    return x[...,1:3].sum(-1)+lv+lp


def native(p):
    return np.concatenate((np.asarray(p['beta']).reshape((*np.asarray(p['var']).shape[:-1],6)),
                           p['var'],p['pi']),axis=-1)


def generate(rng, p, family='binary', n=300, T=20):
    """One independent cluster per person, one shared random intercept per trajectory."""
    beta, var, pi = (np.asarray(p[k]) for k in ('beta','var','pi'))
    batch = len(beta)
    u = (rng.random((batch,n,1)) > np.cumsum(pi,axis=1)[:,None,:-1]).sum(-1)
    rows = np.arange(batch)[:,None]
    b = rng.normal(size=(batch,n))*np.sqrt(var[rows,u])
    t = np.linspace(-1.,1.,T)
    eta = beta[rows,u,0,None]+beta[rows,u,1,None]*t+b[...,None]
    if family == 'binary':
        y = rng.binomial(1,sigmoid(eta)).astype(np.uint8)
    elif family == 'poisson':
        y = rng.poisson(np.exp(eta))
    elif family == 'nb':
        y = rng.negative_binomial(5.,sigmoid(np.log(5.)-eta))
    else:
        raise ValueError(family)
    return dict(y=y,u=u.astype(np.uint8),b=b,t=t,mask=np.ones(T,dtype=np.uint8))


def features(y, family='binary'):
    """Entire ordered response vector followed by t and mask, never true latent inputs."""
    y = np.asarray(y); T = y.shape[-1]
    values = (2*y.astype(np.float32)-1) if family == 'binary' else np.log1p(y).astype(np.float32)
    time = np.broadcast_to(np.linspace(-1.,1.,T,dtype=np.float32),values.shape)
    mask = np.ones_like(values)
    return np.concatenate((values,time,mask),axis=-1)


def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)
