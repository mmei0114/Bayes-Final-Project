import json
from pathlib import Path
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from inference import *

root=Path(__file__).resolve().parents[1];out={}
for family in ('binary','poisson','nb'):
    d=prepared(generate(42,n=12,T=8,family=family));p=initialize(d,3,43);x=pack(p)
    res={}
    for method in ('QML','GVA'):
        f,g=profile(d,x,3,method,61 if method=='QML' else 31)
        fd=approx_derivative(lambda xx:profile(d,xx,3,method,61 if method=='QML' else 31)[0],x,method='3-point').ravel()
        error=np.max(abs(g-fd));assert error<2e-5,(family,method,error)
        res[method+'_gradient_error']=float(error)
    F,m,s,gb,rr=local_vi(d,p['beta'][0],p['var'][0]);assert rr<1e-5,rr
    res['local_residual']=rr
    if family=='poisson':
        # Independent high-order numerical expectation vs analytic Gaussian mgf.
        z,w=nodes(61);analytic=gaussian(d,p['beta'][0],m,s)[0]
        numeric=ll(d,p['beta'][0],m[:,None]+s[:,None]*z)@w
        assert np.max(abs(analytic-numeric))<1e-8
        res['analytic_GH_error']=float(np.max(abs(analytic-numeric)))
        tiny=prepared(generate(9,n=2,T=2,family='poisson'))
        b=np.full((2,1),51.)
        val,score,*_=ll(tiny,np.zeros(2),b,True)
        fd=(ll(tiny,np.zeros(2),b+1e-5)-ll(tiny,np.zeros(2),b-1e-5))/(2e-5)
        relative=np.max(abs((score-fd)/score));assert relative<1e-8
        res['eta51_gradient_relative_error']=float(relative)
    # Validate independent-IS Q-increment MCSE against empirical variation.
    new={k:v.copy() for k,v in p.items()};new['beta']+=.03
    estimates=[];errors=[]
    for rep in range(80):
        _,r,bs,ws,_=estep(d,p,M=256,rng=np.random.default_rng(9090+rep))
        delta,se=increment(d,p,new,r,bs,ws);estimates.append(delta);errors.append(se)
    ratio=np.std(estimates,ddof=1)/np.sqrt(np.mean(np.array(errors)**2))
    assert .65<ratio<1.45,(family,ratio)
    res['empirical_to_delta_sd_ratio']=float(ratio)
    out[family]=res
(root/'qa/validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
