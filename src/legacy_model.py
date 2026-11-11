"""Finite mixtures of random-intercept GLMMs. NumPy/SciPy only.

MCEM: independent t importance draws, two independent E/M replicates for
Monte Carlo step-error assessment. VEM: structured Gaussian latent VI.
"""
import time
import numpy as np
from scipy.special import expit, logsumexp, gammaln
from scipy.optimize import minimize, linear_sum_assignment
from scipy.cluster.vq import kmeans2
from numpy.polynomial.hermite import hermgauss

PI = np.array([.5, .3, .2])
BETA = np.array([[-1.5, -1.8], [.5, 1.8], [1.5, -1.8]])
ORIGINAL = np.array([[-1.5, .2], [.5, -.2], [1., .3]])

def generate(seed, n=300, T=20, family='binary', variance=.5,
             separation=1., irregular=False, original=False, **unused):
    rng = np.random.default_rng(seed)
    beta = ORIGINAL.copy() if original else BETA.mean(0) + separation*(BETA-BETA.mean(0))
    t = np.broadcast_to(np.linspace(-1, 1, T), (n, T)).copy()
    mask = np.ones((n, T))
    if irregular:
        t = np.sort(rng.uniform(-1, 1, (n, T)), axis=1)
        lengths = rng.integers(max(5, T//2), T+1, n)
        mask = (np.arange(T)[None, :] < lengths[:, None]).astype(float)
    u = rng.choice(3, n, p=PI)
    b = rng.normal(0, np.sqrt(variance), n)
    eta = beta[u, 0, None] + beta[u, 1, None]*t + b[:, None]
    if family == 'binary': y = rng.binomial(1, expit(eta))
    elif family == 'poisson': y = rng.poisson(np.exp(eta))
    elif family == 'nb': y = rng.negative_binomial(5., 5./(5.+np.exp(eta)))
    else: raise ValueError(family)
    return dict(y=y, t=t, mask=mask, u=u, b=b, beta=beta,
                variance=np.full(3, variance), pi=PI.copy(), family=family)

def likelihood(data, beta, b, derivatives=False):
    """Subject log likelihood at b[n,M]; no n*T*M tensor is retained."""
    y, t, mask = data['y'], data['t'], data['mask']
    ell = np.zeros_like(b)
    if derivatives:
        s0 = np.zeros_like(b); s1 = np.zeros_like(b)
        h00 = np.zeros_like(b); h01 = np.zeros_like(b); h11 = np.zeros_like(b)
    for j in range(y.shape[1]):
        tj = t[:, j, None]; yy = y[:, j, None]; w = mask[:, j, None]
        eta = beta[0]+beta[1]*tj+b
        if data['family'] == 'binary':
            p = expit(eta)
            ll = yy*eta-np.logaddexp(0, eta)
            if derivatives: s=yy-p; h=p*(1-p)
        elif data['family'] == 'poisson':
            mu = np.exp(np.minimum(eta, 50))
            ll = yy*eta-mu-gammaln(yy+1)
            if derivatives: s=yy-mu; h=mu
        else:
            a=5.; p=expit(eta-np.log(a))
            ll = (gammaln(yy+a)-gammaln(a)-gammaln(yy+1)
                  +yy*(eta-np.log(a))-(yy+a)*np.logaddexp(0, eta-np.log(a)))
            if derivatives: s=yy-(yy+a)*p; h=(yy+a)*p*(1-p)
        ell += w*ll
        if derivatives:
            s0 += w*s; s1 += w*s*tj
            h00 += w*h; h01 += w*h*tj; h11 += w*h*tj*tj
    if derivatives: return ell,s0,s1,h00,h01,h11
    return ell

def mode(data, beta, var):
    m=np.zeros((len(data['y']), 1))
    for _ in range(30):
        _,s,_,h,_,_=likelihood(data,beta,m,True)
        step=np.clip((s-m/var)/(h+1/var),-2,2)
        m+=step
        if np.max(np.abs(step))<1e-9: break
    return m[:,0], 1/(h[:,0]+1/var)

def initialize(data,K,seed):
    rng=np.random.default_rng(seed)
    mask=data['mask']; t=data['t']; y=data['y']; N=mask.sum(1)
    mean=(y*mask).sum(1)/N
    center=(t*mask).sum(1)/N
    slope=((t-center[:,None])*y*mask).sum(1)/np.maximum(((t-center[:,None])**2*mask).sum(1),1e-6)
    if data['family']=='binary':
        p=(mean*N+.5)/(N+1)
        features=np.c_[np.log(p/(1-p)),np.clip(slope/(p*(1-p)),-4,4)]
    else: features=np.c_[np.log(mean+.1),np.clip(slope/(mean+.1),-4,4)]
    features+=rng.normal(0,.08,features.shape)
    cent,lab=kmeans2(features,K,minit='++',iter=30,seed=rng)
    return dict(beta=cent, var=np.full(K,.5), pi=np.array([(lab==k).mean() for k in range(K)]))

def e_is(data,p,M,rng):
    n=len(data['y']); K=len(p['pi']); bs=[]; ws=[]; logs=[]; ess=[]
    for k in range(K):
        m,v=mode(data,p['beta'][k],p['var'][k])
        # t_5 scale chosen so its variance equals inverse posterior curvature.
        scale=np.sqrt(v*3/5)
        z=rng.standard_t(5,(n,M)); b=m[:,None]+scale[:,None]*z
        logq=(gammaln(3)-gammaln(2.5)-.5*np.log(5*np.pi)-np.log(scale[:,None])
              -3*np.log1p(z*z/5))
        logw=likelihood(data,p['beta'][k],b)-.5*(np.log(2*np.pi*p['var'][k])+b*b/p['var'][k])-logq
        norm=logsumexp(logw,axis=1)
        w=np.exp(logw-norm[:,None]); logs.append(norm-np.log(M))
        bs.append(b); ws.append(w); ess.append(1/(w*w).sum(1))
    L=np.array(logs).T+np.log(p['pi'])
    r=np.exp(L-logsumexp(L,axis=1)[:,None])
    return r,bs,ws,float(np.quantile(np.array(ess)/M,.1))

def e_vi(data,p,previous=None,order=15):
    z,w=hermgauss(order); z=z*np.sqrt(2); w=w/np.sqrt(np.pi)
    n=len(data['y']); K=len(p['pi']); bs=[]; ws=[]; scores=[]; local=[]; ok=True
    for k in range(K):
        m,v=mode(data,p['beta'][k],p['var'][k])
        initial=np.r_[m,np.sqrt(v)] if previous is None else previous[k]
        def fun(x):
            mu=x[:n]; sd=x[n:]; b=mu[:,None]+sd[:,None]*z
            ll,s,_,_,_,_=likelihood(data,p['beta'][k],b,True)
            val=(ll@w-.5*(mu*mu+sd*sd)/p['var'][k]+np.log(sd)).sum()
            gm=s@w-mu/p['var'][k]
            gs=(s*z)@w-sd/p['var'][k]+1/sd
            return -val,-np.r_[gm,gs]
        opt=minimize(fun,initial,jac=True,method='L-BFGS-B',
                     bounds=[(-20,20)]*n+[(.001,5)]*n,
                     options=dict(maxiter=60,ftol=1e-11,gtol=1e-6,maxcor=10))
        ok &= opt.success
        mu=opt.x[:n]; sd=opt.x[n:]; b=mu[:,None]+sd[:,None]*z
        score=likelihood(data,p['beta'][k],b)@w-.5*((mu*mu+sd*sd)/p['var'][k]-1+np.log(p['var'][k])-2*np.log(sd))
        scores.append(score); bs.append(b); ws.append(np.broadcast_to(w,b.shape)); local.append(opt.x)
    L=np.array(scores).T+np.log(p['pi'])
    elbo=logsumexp(L,axis=1).sum()
    r=np.exp(L-logsumexp(L,axis=1)[:,None])
    return r,bs,ws,elbo,local,ok

def m_step(data,p,r,bs,ws):
    n,K=r.shape; new=dict(beta=p['beta'].copy(),var=p['var'].copy(),pi=np.maximum(r.mean(0),1e-8))
    new['pi']/=new['pi'].sum(); ok=True
    for k in range(K):
        ww=r[:,k,None]*ws[k]; b=bs[k]
        new['var'][k]=np.clip((ww*b*b).sum()/max(r[:,k].sum(),1e-12),.005,5.)
        def fun(beta):
            ll,s0,s1,_,_,_=likelihood(data,beta,b,True)
            return -(ww*ll).sum(),-np.array([(ww*s0).sum(),(ww*s1).sum()])
        opt=minimize(fun,p['beta'][k],jac=True,method='L-BFGS-B',bounds=[(-8,8),(-8,8)],
                     options=dict(ftol=1e-10,gtol=1e-6,maxiter=40))
        new['beta'][k]=opt.x; ok &= opt.success
    return new,ok

def vector(p): return np.r_[p['beta'].ravel(),p['var'],p['pi']]

def quadrature(data,p,order=61,moments=False):
    """Adaptive Gaussian quadrature, separate from estimation."""
    z,w=hermgauss(order); z*=np.sqrt(2); w/=np.sqrt(np.pi)
    logs=[]; means=[]; vs=[]
    for k in range(len(p['pi'])):
        m,v=mode(data,p['beta'][k],p['var'][k]); sd=np.sqrt(v)
        b=m[:,None]+sd[:,None]*z
        logq=-.5*z*z-.5*np.log(2*np.pi)-np.log(sd[:,None])
        a=(likelihood(data,p['beta'][k],b)-.5*(np.log(2*np.pi*p['var'][k])+b*b/p['var'][k])
           -logq+np.log(w))
        norm=logsumexp(a,axis=1); post=np.exp(a-norm[:,None])
        logs.append(norm); mu=(post*b).sum(1); means.append(mu)
        vs.append((post*(b-mu[:,None])**2).sum(1))
    L=np.array(logs).T+np.log(p['pi']); ll=logsumexp(L,axis=1)
    r=np.exp(L-ll[:,None])
    if moments: return ll.sum(),r,np.array(means).T,np.array(vs).T
    return ll.sum(),r

def fit(data,K,method,seed,starts=3,maxiter=160,adaptive=True,fixed_M=128):
    begin=time.perf_counter(); runs=[]; best=None
    for start in range(starts):
        p=initialize(data,K,seed+start*1009); rng=np.random.default_rng(seed+start*1009+73)
        M=64 if adaptive else fixed_M; previous=None; trace=[]; stable=0; oldobj=None; success=True
        for it in range(maxiter):
            old=vector(p)
            if method=='VEM':
                r,bs,ws,obj,previous,ok=e_vi(data,p,previous)
                new,ok2=m_step(data,p,r,bs,ws); noise=0.; ess=1.
            else:
                r,bs,ws,ess=e_is(data,p,M,rng); a,ok=m_step(data,p,r,bs,ws)
                r2,b2,w2,ess2=e_is(data,p,M,rng); c,ok2=m_step(data,p,r2,b2,w2)
                new={key:(a[key]+c[key])/2 for key in p}
                # Error of average of two independent parameter updates.
                noise=np.max(np.abs(vector(a)-vector(c))/(2*(1+np.abs(old))))
                obj=quadrature(data,new,order=21)[0]; ess=min(ess,ess2)
            success &= ok and ok2
            step=np.max(np.abs(vector(new)-old)/(1+np.abs(old)))
            rel=np.inf if oldobj is None else abs(obj-oldobj)/(1+abs(oldobj))
            trace.append(dict(iteration=it+1,objective=float(obj),step=float(step),noise=float(noise),M=M,ess=ess))
            if method=='VEM': good=step<.005 and rel<1e-6
            else: good=step<.015 and rel<2e-5 and noise<.0075 and (M>=256 or not adaptive)
            stable=stable+1 if good else 0
            if method=='MCEM' and adaptive and (step<max(2*noise,.03) or ess<.4): M=min(512,M*2)
            p=new; oldobj=obj
            if stable>=3: break
        # Re-evaluate VI ELBO after final M-step and select by its own objective.
        ll,rq=quadrature(data,p)
        if method=='VEM':
            r,bs,ws,finalobj,previous,ok=e_vi(data,p,previous)
            pred_r=r
        else: finalobj=ll; pred_r=rq
        run=dict(params={k:v.tolist() for k,v in p.items()},trace=trace,converged=stable>=3,
                 optimizer_success=bool(success),objective=float(finalobj),loglik=float(ll),start=start,
                 responsibility=pred_r.tolist(),local=None if previous is None else [v.tolist() for v in previous])
        runs.append(run)
        if best is None or finalobj>best['objective']: best=run
    best=best.copy(); best['seconds']=time.perf_counter()-begin
    best['starts']=[{k:v for k,v in r.items() if k not in ('responsibility','local')} for r in runs]
    return best

def ari(a,b):
    tab=np.zeros((a.max()+1,b.max()+1),int); np.add.at(tab,(a,b),1)
    c2=lambda v:(v*(v-1)/2).sum()
    total=len(a)*(len(a)-1)/2; A=c2(tab); B=c2(tab.sum(0)); C=c2(tab.sum(1))
    expected=B*C/total; den=(B+C)/2-expected
    return float((A-expected)/den) if den else 1.

def evaluate(data,run,K):
    r=np.array(run['responsibility']); pred=r.argmax(1); u=data['u']
    tab=np.zeros((3,K),int); np.add.at(tab,(u,pred),1)
    row,col=linear_sum_assignment(-tab)
    acc=tab[row,col].sum()/len(u)
    p={k:np.array(v) for k,v in run['params'].items()}
    # Parameter alignment uses beta-distance, independent of confusion matching.
    if K==3:
        rr,cc=linear_sum_assignment(((data['beta'][:,None,:]-p['beta'][None,:,:])**2).sum(2))
        beta_rmse=np.sqrt(np.mean((data['beta'][rr]-p['beta'][cc])**2))
        var_rmse=np.sqrt(np.mean((data['variance'][rr]-p['var'][cc])**2))
        pi_rmse=np.sqrt(np.mean((data['pi'][rr]-p['pi'][cc])**2))
    else: beta_rmse=var_rmse=pi_rmse=None
    return dict(ari=ari(u,pred),error=1-acc,beta_rmse=beta_rmse,var_rmse=var_rmse,pi_rmse=pi_rmse,
                seconds=run['seconds'],converged=run['converged'],iterations=len(run['trace']),
                loglik=run['loglik'],confusion=tab.tolist())
