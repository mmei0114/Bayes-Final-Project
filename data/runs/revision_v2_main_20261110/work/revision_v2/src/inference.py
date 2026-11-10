"""Likelihood and Gaussian variational profile inference; independent-IS MCEM.

References and exact scope are documented in ../README.md and theory.md.
No complete-data parameter averaging and no clipped Poisson mean.
"""
import time
import numpy as np
from scipy.special import expit, logsumexp, gammaln
from scipy.optimize import minimize
from scipy.stats import norm
from numpy.polynomial.hermite import hermgauss
from legacy_model import generate, initialize, evaluate

def weighted(w,v):
    return np.multiply(w,v,out=np.zeros_like(v),where=w>0)

def prepared(data):
    d=dict(data); d['family']=str(d['family'])
    d['const']=gammaln(d['y']+1)
    if d['family']=='nb':
        d['const']=gammaln(d['y']+5)-gammaln(5)-d['const']
    d['sumy']=(d['y']*d['mask']).sum(1)[:,None]
    d['sumty']=(d['y']*d['t']*d['mask']).sum(1)[:,None]
    d['sumconst']=(d['const']*d['mask']).sum(1)[:,None]
    return d

def ll(d,beta,b,deriv=False):
    """b has shape (n, samples); d must be prepared. Overflow -> -inf, not a new model."""
    if d['family']=='poisson':
        e=np.exp(beta[1]*d['t'])*d['mask'];A=e.sum(1)[:,None]
        with np.errstate(over='ignore'):eb=np.exp(b+beta[0])
        mu=eb*A
        out=d['sumy']*(beta[0]+b)+d['sumty']*beta[1]-mu-d['sumconst']
        if not deriv:return out
        mu1=eb*(e*d['t']).sum(1)[:,None];mu2=eb*(e*d['t']**2).sum(1)[:,None]
        return out,d['sumy']-mu,d['sumty']-mu1,mu,mu1,mu2
    out=np.zeros_like(b)
    if deriv:
        s0=np.zeros_like(b);s1=np.zeros_like(b)
        h0=np.zeros_like(b);h1=np.zeros_like(b);h2=np.zeros_like(b)
    for j in range(d['y'].shape[1]):
        t=d['t'][:,j,None]; y=d['y'][:,j,None]; w=d['mask'][:,j,None]
        eta=beta[0]+beta[1]*t+b
        if d['family']=='binary':
            p=expit(eta); val=y*eta-np.logaddexp(0,eta)
            if deriv: s=y-p;h=p*(1-p)
        elif d['family']=='poisson':
            with np.errstate(over='ignore'):
                mu=np.exp(eta)
            val=y*eta-mu-d['const'][:,j,None]
            if deriv:s=y-mu;h=mu
        else:
            z=eta-np.log(5.);p=expit(z)
            val=d['const'][:,j,None]+y*z-(y+5)*np.logaddexp(0,z)
            if deriv:s=y-(y+5)*p;h=(y+5)*p*(1-p)
        # Avoid 0 * inf at padded missing observations.
        out+=np.where(w>0,val,0.)
        if deriv:
            ss=np.where(w>0,s,0.);hh=np.where(w>0,h,0.)
            s0+=ss;s1+=ss*t;h0+=hh;h1+=hh*t;h2+=hh*t*t
    return (out,s0,s1,h0,h1,h2) if deriv else out

def mode(d,beta,var):
    m=np.zeros((len(d['y']),1))
    for _ in range(80):
        val,s,_,h,_,_=ll(d,beta,m,True)
        step=np.clip((s-m/var)/(h+1/var),-3.,3.)
        target=val-m*m/(2*var)
        rate=np.ones_like(m)
        for back in range(30):
            trial=m+rate*step
            good=ll(d,beta,trial)-trial*trial/(2*var)>=target-1e-12
            if good.all(): break
            rate=np.where(good,rate,.5*rate)
        m=trial
        if np.max(abs(rate*step))<1e-9:break
    _,s,_,h,_,_=ll(d,beta,m,True)
    if np.max(abs(s-m/var))>1e-5: raise RuntimeError('Conditional mode residual too large')
    return m[:,0],1/(h[:,0]+1/var)

def nodes(order):
    z,w=hermgauss(order);return z*np.sqrt(2),w/np.sqrt(np.pi)

def estep(d,p,order=61,M=None,rng=None):
    """IS uses independent M draws in each subject-component stratum."""
    bs=[];ws=[];logs=[];ess=[]
    for k in range(len(p['pi'])):
        m,v=mode(d,p['beta'][k],p['var'][k])
        if M is None:
            z,w=nodes(order);sd=np.sqrt(v)
            b=m[:,None]+sd[:,None]*z
            logq=-.5*z*z-.5*np.log(2*np.pi)-np.log(sd[:,None])
            logbase=np.log(w)
        else:
            sd=np.sqrt(v*3/5);z=rng.standard_t(5,(len(m),M))
            b=m[:,None]+sd[:,None]*z
            logq=gammaln(3)-gammaln(2.5)-.5*np.log(5*np.pi)-np.log(sd[:,None])-3*np.log1p(z*z/5)
            logbase=-np.log(M)
        a=ll(d,p['beta'][k],b)-.5*(np.log(2*np.pi*p['var'][k])+b*b/p['var'][k])-logq+logbase
        normed=logsumexp(a,axis=1);w=np.exp(a-normed[:,None])
        logs.append(normed);bs.append(b);ws.append(w);ess.append(1/(w*w).sum(1))
    L=np.array(logs).T+np.log(p['pi']);s=logsumexp(L,axis=1)
    return s.sum(),np.exp(L-s[:,None]),bs,ws,float(np.quantile(ess,.1))

def gaussian(d,beta,m,s,order=31):
    """Exact Poisson Gaussian expectation, otherwise positive-weight GH."""
    if d['family']=='poisson':
        t=d['t'];y=d['y'];w=d['mask']
        eta=beta[0]+beta[1]*t+m[:,None]
        with np.errstate(over='ignore'): mu=np.exp(eta+.5*s[:,None]**2)
        val=((y*eta-mu-d['const'])*w).sum(1)
        a=((y-mu)*w).sum(1);c=(mu*w).sum(1)
        # Gradients and negative Hessian w.r.t (m,s).
        gm=a;gs=-s*c
        h00=c;h01=s*c;h11=(1+s*s)*c
        gb=np.c_[a,((y-mu)*t*w).sum(1)]
        return val,gm,gs,h00,h01,h11,gb
    z,w=nodes(order);b=m[:,None]+s[:,None]*z
    val,a,c,h,_,_=ll(d,beta,b,True)
    return val@w,a@w,(a*z)@w,h@w,(h*z)@w,(h*z*z)@w,np.c_[a@w,c@w]

def local_vi(d,beta,var,order=31):
    """Independent, strictly concave (m,s) objectives; vectorized damped Newton."""
    m,v=mode(d,beta,var);s=np.sqrt(v);ok=np.zeros(len(m),bool)
    for it in range(100):
        val,gm,gs,a,c,h,_=gaussian(d,beta,m,s,order)
        target=val-(m*m+s*s)/(2*var)+np.log(s)
        gm-=m/var;gs+=1/s-s/var
        a+=1/var;h+=1/var+1/(s*s)
        ok=np.maximum(abs(gm),abs(gs))<1e-7
        if ok.all():break
        det=a*h-c*c
        dm=(h*gm-c*gs)/det;ds=(a*gs-c*gm)/det
        rate=np.minimum(1.,np.where(ds<0,-.9*s/np.minimum(ds,-1e-300),1.))
        dm=np.where(ok,0.,dm);ds=np.where(ok,0.,ds)
        for back in range(35):
            mt=m+rate*dm;st=s+rate*ds
            vt=gaussian(d,beta,mt,st,order)[0]-(mt*mt+st*st)/(2*var)+np.log(st)
            good=np.isfinite(vt)&(vt>=target+1e-4*rate*(gm*dm+gs*ds)-1e-12)
            if good.all():break
            rate=np.where(good,rate,.5*rate)
        m=mt;s=st
    val,gm,gs,_,_,_,gb=gaussian(d,beta,m,s,order)
    residual=np.max(np.maximum(abs(gm-m/var),abs(gs+1/s-s/var)))
    F=val-.5*((m*m+s*s)/var-1+np.log(var)-2*np.log(s))
    return F,m,s,gb,float(residual)

def pack(p):return np.r_[p['beta'].ravel(),np.log(p['var']),np.log(p['pi'][:-1]/p['pi'][-1])]
def unpack(x,K):
    return dict(beta=x[:2*K].reshape(K,2),var=np.exp(x[2*K:3*K]),pi=np.exp(np.r_[x[3*K:],0.]-logsumexp(np.r_[x[3*K:],0.])))
def bounds(K):return [(-8,8)]*(2*K)+[(np.log(.005),np.log(5.))]*K+[(-16,16)]*(K-1)
def projected(x,g,bnd):
    gg=g.copy()
    for j,(a,b) in enumerate(bnd):
        if (x[j]<=a+1e-8 and gg[j]>0) or (x[j]>=b-1e-8 and gg[j]<0):gg[j]=0
    return float(np.max(abs(gg)))

def profile(d,x,K,method,order=61,details=False):
    p=unpack(x,K);gb=[];gv=[];mus=[];sds=[];res=0.
    if method=='GVA':
        Fs=[]
        for k in range(K):
            F,m,s,g,rr=local_vi(d,p['beta'][k],p['var'][k],order)
            Fs.append(F);gb.append(g);gv.append(.5*((m*m+s*s)/p['var'][k]-1));mus.append(m);sds.append(s);res=max(res,rr)
        L=np.array(Fs).T+np.log(p['pi']);val=logsumexp(L,axis=1)
        r=np.exp(L-val[:,None]);value=val.sum()
    else:
        value,r,bs,ws,_=estep(d,p,order)
        for k in range(K):
            _,a,c,_,_,_=ll(d,p['beta'][k],bs[k],True)
            gb.append(np.c_[(a*ws[k]).sum(1),(c*ws[k]).sum(1)])
            gv.append(.5*((ws[k]*bs[k]**2).sum(1)/p['var'][k]-1))
    gradb=np.array([(r[:,k,None]*gb[k]).sum(0) for k in range(K)]).ravel()
    gradv=np.array([(r[:,k]*gv[k]).sum() for k in range(K)])
    gradpi=r.sum(0)[:-1]-len(r)*p['pi'][:-1]
    grad=np.r_[gradb,gradv,gradpi]
    if details:return value,r,dict(mean=np.array(mus).T.tolist(),sd=np.array(sds).T.tolist(),residual=res)
    return -value/len(r),-grad/len(r)

def fit_profile(data,K,method,seed,starts=3,maxiter=300,order=None,checkpoint=None):
    begin=time.perf_counter();d=prepared(data);runs=[]
    order=order or (31 if method=='GVA' else 61)
    for start in range(starts):
        init=initialize(data,K,seed+start*1009);x0=pack(init);trace=[]
        def fun(x):return profile(d,x,K,method,order)
        def callback(x):
            f,g=fun(x);trace.append(dict(iteration=len(trace)+1,objective=float(-f*len(d['y'])),gradient=projected(x,g,bounds(K))))
            if checkpoint and len(trace)%10==0:checkpoint(dict(active_start=start,packed_parameters=x.tolist(),trace=trace,completed_starts=runs))
        opt=minimize(fun,x0,jac=True,bounds=bounds(K),method='L-BFGS-B',callback=callback,
                     options=dict(maxiter=maxiter,ftol=1e-12,gtol=2e-6,maxls=30,maxcor=15))
        f,g=fun(opt.x);value,r,local=profile(d,opt.x,K,method,order,True)
        pg=projected(opt.x,g,bounds(K));p=unpack(opt.x,K)
        loglik=estep(d,p,101)[0];low=estep(d,p,61)[0]
        run=dict(params={k:v.tolist() for k,v in p.items()},trace=trace,
            objective=float(value),loglik=float(loglik),quadrature_error=float(abs(loglik-low)),
            responsibility=r.tolist(),local=local,start=start,gradient=pg,
            converged=bool(pg<1e-4 and (method!='GVA' or local['residual']<1e-5)),
            optimizer_success=bool(opt.success),message=str(opt.message),iterations=int(opt.nit))
        runs.append(run)
        if checkpoint:checkpoint(dict(completed_starts=runs))
    best=max(runs,key=lambda v:v['objective']).copy();best['seconds']=time.perf_counter()-begin
    best['starts']=[{k:v for k,v in r.items() if k not in ('responsibility','local')} for r in runs]
    return best

def qvalue(d,p,r,bs,ws):
    vals=[]
    for k in range(len(p['pi'])):
        b=bs[k]
        vals.append(ll(d,p['beta'][k],b)+np.log(p['pi'][k])-.5*(np.log(2*np.pi*p['var'][k])+b*b/p['var'][k]))
    return float(sum(weighted(r[:,k,None]*ws[k],vals[k]).sum() for k in range(len(vals)))),vals

def mstep(d,p,r,bs,ws):
    K=len(p['pi']);new=dict(beta=p['beta'].copy(),var=p['var'].copy(),pi=r.mean(0))
    if np.min(new['pi'])<1e-12: raise RuntimeError('Numerically empty component')
    ok=True;residual=0.
    for k in range(K):
        w=r[:,k,None]*ws[k];N=r[:,k].sum()
        new['var'][k]=np.clip((w*bs[k]**2).sum()/N,.005,5.)
        def f(beta):
            v,a,c,_,_,_=ll(d,beta,bs[k],True)
            return -weighted(w,v).sum()/N,-np.array([weighted(w,a).sum(),weighted(w,c).sum()])/N
        opt=minimize(f,p['beta'][k],jac=True,bounds=[(-8,8)]*2,method='L-BFGS-B',options=dict(maxiter=80,ftol=1e-12,gtol=1e-7))
        new['beta'][k]=opt.x
        rr=projected(opt.x,f(opt.x)[1],[(-8,8)]*2);residual=max(residual,rr);ok &= bool(opt.success)
    oldQ,_=qvalue(d,p,r,bs,ws);newQ,_=qvalue(d,new,r,bs,ws)
    if newQ<oldQ-1e-7:raise RuntimeError('M-step did not increase sampled Q')
    return new,bool(ok),residual

def increment(d,old,new,r,bs,ws):
    """Delta-method MCSE of stratified self-normalized IS ratio, summed over subjects.

    Center by the SUBJECT joint expectation, not its component expectation;
    this includes Monte Carlo variation in estimated class responsibilities.
    """
    _,a=qvalue(d,old,r,bs,ws);_,b=qvalue(d,new,r,bs,ws)
    joint=[r[:,k,None]*ws[k] for k in range(len(a))]
    deltas=[np.subtract(v,u,out=np.zeros_like(v),where=w>0) for u,v,w in zip(a,b,joint)]
    per=sum(weighted(w,v).sum(1) for w,v in zip(joint,deltas));variance=0.
    for w,delta in zip(joint,deltas):
        M=w.shape[1];z=w*(delta-per[:,None]);z-=z.mean(1)[:,None]
        variance+=M/(M-1)*(z*z).sum()
    return float(per.sum()),float(np.sqrt(variance))

def fit_mcem(data,K,seed,starts=3,maxiter=250,checkpoint=None):
    begin=time.perf_counter();d=prepared(data);runs=[];n=len(d['y'])
    for start in range(starts):
        p=initialize(data,K,seed+1009*start);rng=np.random.default_rng(seed+1009*start+73)
        M=128;trace=[];stable=0;reason='iteration_limit';inner_ok=True
        for it in range(maxiter):
            _,r,bs,ws,ess=estep(d,p,M=M,rng=rng);new,ok,mres=mstep(d,p,r,bs,ws);inner_ok &= ok
            _,rv,bv,wv,ev=estep(d,p,M=M,rng=rng)
            delta,se=increment(d,p,new,rv,bv,wv)
            lo=delta-1.96*se;hi=delta+1.96*se
            step=np.max(abs(pack(new)-pack(p))/(1+abs(pack(p))))
            # Numerical target per subject, calibrated before the frozen main suite.
            small=hi/n<5e-5 and se/n<2e-5 and step<.01 and mres<1e-4
            accept=lo>0
            if accept:p=new
            stable=stable+1 if small else 0
            trace.append(dict(iteration=it+1,M=M,delta_Q=delta,mcse=se,lower=lo,upper=hi,step=float(step),accepted=bool(accept),ess_fraction=min(ess,ev)/M,mstep_gradient=mres))
            if checkpoint and (it+1)%5==0:checkpoint(dict(active_start=start,params={k:v.tolist() for k,v in p.items()},trace=trace,M=M,stable=stable,rng_state=rng.bit_generator.state,completed_starts=runs))
            if stable>=3:reason='small_Q_increment';break
            if (not accept and not small) or min(ess,ev)/M<.4:
                if M>=8192:reason='mc_budget';break
                M*=2
        L,r,_,_,_=estep(d,p,101);Lg=estep(d,p,61)[0]
        _,g=profile(d,pack(p),K,'QML',61);pg=projected(pack(p),g,bounds(K))
        run=dict(params={k:v.tolist() for k,v in p.items()},trace=trace,objective=float(L),loglik=float(L),
                 responsibility=r.tolist(),converged=reason=='small_Q_increment',optimizer_success=bool(inner_ok),
                 gradient=pg,message=reason,start=start,iterations=len(trace),quadrature_error=float(abs(L-Lg)))
        runs.append(run)
        if checkpoint:checkpoint(dict(completed_starts=runs))
    best=max(runs,key=lambda x:x['loglik']).copy();best['seconds']=time.perf_counter()-begin
    best['starts']=[{k:v for k,v in r.items() if k!='responsibility'} for r in runs]
    return best
