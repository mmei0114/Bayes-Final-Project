"""Audit and plot saved SBI pilot outputs, without TensorFlow or new fitting."""
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'build/matplotlib'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binom

DATA=ROOT/'data/sbi_pilot'


def save(fig,name):
    for ext in ('pdf','png'):
        fig.savefig(ROOT/'figures'/f'{name}.{ext}',dpi=170,bbox_inches='tight',
                    metadata={'Creator':'Bayes Final Project','CreationDate':None} if ext=='pdf' else None)
    plt.close(fig)


def main():
    for line in (DATA/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1)
        assert hashlib.sha256((DATA/name).read_bytes()).hexdigest()==digest,name
    out=DATA/'output';ev=json.loads((out/'evaluation.json').read_text())
    history=json.loads((out/'history.json').read_text());selection=json.loads((out/'selection.json').read_text())
    arrays=np.load(out/'posterior_samples.npz');values=arrays['native'];truth=arrays['true'];ranks=arrays['ranks']
    assert values.shape==(ev['datasets'],ev['draws'],12)
    assert np.isfinite(values).all() and np.all(values[...,6:9]>.005) and np.all(values[...,6:9]<5.)
    assert np.all(np.diff(values[...,[0,2,4]],axis=-1)>0)
    assert np.allclose(values[...,9:12].sum(-1),1.)
    assert np.array_equal(ranks,(values<truth[:,None,:]).sum(1))
    for level,quant in ((95,[.025,.975]),(50,[.25,.75])):
        lo,hi=np.quantile(values,quant,axis=1)
        assert np.allclose(((truth>=lo)&(truth<=hi)).mean(0),ev[f'coverage_{level}'])
    for key in ('probabilities','conditional_probabilities','quadrature_probabilities'):
        assert np.isfinite(arrays[key]).all() and np.min(arrays[key])>=0
        assert np.allclose(arrays[key].sum(-1),1.,atol=2e-6),key
    assert selection['best_npe_step'] in [v['step'] for v in history]
    assert selection['best_classifier_step']==min(history,key=lambda v:v['validation_ce'])['step']
    assert np.isclose(selection['best_validation_nll'],min(v['validation_nll'] for v in history),atol=1e-4)
    hist=np.array([np.histogram(ranks[:,j],bins=np.linspace(-.5,ev['draws']+.5,17))[0] for j in range(12)])
    assert np.array_equal(hist,np.array(ev['marginal_rank_histograms']))
    assert np.all(hist.sum(1)==ev['datasets'])
    result=ROOT/'results/sbi_pilot_metrics.csv'
    with result.open('w',newline='') as file:
        writer=csv.writer(file);writer.writerow(['parameter','coverage_50','coverage_95','prior_scaled_RMSE','contraction'])
        for j,name in enumerate(ev['names']):
            writer.writerow([name]+[ev[k][j] for k in ('coverage_50','coverage_95','prior_scaled_RMSE','contraction')])
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    step=np.array([v['step'] for v in history])
    fig,axes=plt.subplots(1,2,figsize=(10,3.4))
    for ax,train,val,title,best in zip(axes,['train_nll','train_ce'],['validation_nll','validation_ce'],
                                     ['Global posterior NPE','Conditional classifier'],
                                     [selection['best_npe_step'],selection['best_classifier_step']]):
        ax.plot(step,[v[train] for v in history],label='Training window mean')
        ax.plot(step,[v[val] for v in history],label='Validation (512 datasets)')
        ax.axvline(best,ls=':',color='.4',label='Selected checkpoint')
        ax.set(xlabel='Optimizer step',ylabel='Negative log density' if train=='train_nll' else 'Cross entropy',title=title)
    axes[1].legend(fontsize=8);fig.suptitle('Binary SBI pilot: training and validation losses')
    fig.tight_layout();save(fig,'sbi_pilot_training')
    fig,axes=plt.subplots(3,4,figsize=(11.5,7))
    lower,upper=binom.ppf([.025,.975],ev['datasets'],1/16)
    for j,ax in enumerate(axes.flat):
        ax.axhspan(lower,upper,color='.9',label='Pointwise 95% binomial band')
        ax.bar(np.arange(16),hist[j],width=.85,color='#34749a')
        ax.axhline(ev['datasets']/16,color='black',lw=.7)
        ax.set(title=ev['names'][j],xticks=[0,7.5,15],xticklabels=['0','1/2','1'],xlabel='Normalized rank',ylabel='Datasets')
    fig.suptitle(f"Binary pilot SBC: {ev['datasets']} held-out datasets, {ev['draws']} posterior draws\nGray bands show pointwise 95% binomial limits")
    fig.tight_layout();save(fig,'sbi_pilot_sbc')
    fig,axes=plt.subplots(1,2,figsize=(11,3.6))
    x=np.arange(12)
    for target,color in ((50,'#34749a'),(95,'#bf7837')):
        axes[0].plot(x,ev[f'coverage_{target}'],'o-',label=f'{target}% intervals',color=color)
        axes[0].axhline(target/100,color=color,ls=':',lw=1)
    axes[0].set(ylabel='Empirical coverage',ylim=(0,1.03));axes[0].legend(fontsize=8)
    axes[1].bar(x,ev['contraction'],color='#34749a');axes[1].axhline(0,color='.3',lw=.8)
    axes[1].set(ylabel='Posterior contraction')
    labels=[rf'$\beta_{{{k},{j}}}$' for k in range(1,4) for j in (0,1)]
    labels += [rf'$D_{k}$' for k in range(1,4)] + [rf'$\pi_{k}$' for k in range(1,4)]
    for ax in axes:ax.set_xticks(x,labels)
    fig.suptitle('Binary pilot: interval coverage and posterior contraction')
    fig.tight_layout();save(fig,'sbi_pilot_recovery')
    pp=ev['posterior_predictive'];q=np.array(pp['replicated_mean_by_time_quantiles']);t=np.linspace(-1,1,20)
    fig,ax=plt.subplots(figsize=(6,3.5));ax.fill_between(t,q[0],q[2],alpha=.22,label='90% predictive band (new people)')
    ax.plot(t,q[1],label='Predictive median');ax.plot(t,pp['observed_mean_by_time'],'o-',label='Observed mean')
    ax.set(xlabel='Time',ylabel='Mean binary response',title='Pilot predictive check: first calibration dataset')
    ax.legend(fontsize=8);fig.tight_layout();save(fig,'sbi_pilot_predictive')
    audit=dict(artifact_hashes_verified=True,posterior_constraints_verified=True,
               coverage_recomputed=True,rank_histograms_recomputed=True,
               class_probabilities_normalized=True,validation_checkpoint_selection_verified=True,
               datasets=ev['datasets'],draws=ev['draws'],formal_comparison_completed=False,
               full_parameter_posterior_MCMC_reference_completed=False)
    (ROOT/'qa/sbi_pilot/evaluation_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
