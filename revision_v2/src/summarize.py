"""Consolidate only the frozen cloud main suite; do not mix pilots/v1 results."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
from scipy.special import expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import atomic,SCENARIOS
from inference import prepared,profile,pack

ROOT=Path(__file__).resolve().parents[1]
LABELS={'baseline':'基准','hard':'分离减半','original':'作业建议参数','n100':r'$n=100$','n1000':r'$n=1000$','T10':r'$T=10$','T50':r'$T=50$','K2':r'$K=2$','K4':r'$K=4$','var01':r'$D=0.1$','var15':r'$D=1.5$','poisson':'Poisson','nb':'负二项','irregular':'不规则时间'}

def main():
    for folder in ('results','figures','qa'):(ROOT/folder).mkdir(parents=True,exist_ok=True)
    ap=argparse.ArgumentParser();ap.add_argument('--run-root',default=str(ROOT.parent/'data/runs/revision_v2_main_20261110'));ap.add_argument('--mcmc-root',default=str(ROOT.parent/'data/runs/revision_v2_diagnostic_20261110'));args=ap.parse_args();runroot=Path(args.run_root);mcroot=Path(args.mcmc_root)
    records=[]
    for folder,expected in [('robustness',140),('comparison',15)]:
        status=json.loads((runroot/folder/'STATUS.json').read_text());assert status['status']=='COMPLETE' and status['complete']==expected
        metadata=json.loads((runroot/folder/'metadata.json').read_text())
        for name,digest in metadata['spec']['source'].items():
            frozen=runroot/'work/revision_v2/src'/name
            assert hashlib.sha256(frozen.read_bytes()).hexdigest()==digest,name
            if name!='experiment.py':assert frozen.read_bytes()==(ROOT/'src'/name).read_bytes(),name
        assert not list((runroot/folder).glob('*.error.json')),'Task error files present'
        subset=[json.loads(p.read_text()) for p in sorted((runroot/folder).glob('*.json')) if not any(s in p.name for s in ('.checkpoint.','.error.'))]
        subset=[v for v in subset if 'fit' in v];assert len(subset)==expected
        records+=subset
    assert len(records)==155
    groups={};warnings=[];maxnorm=0.;maxquad=0.;maxlocal=0.;min_elbo_gap=float('inf')
    for record in records:
        fit=record['fit'];key=(record['scenario'],record['method']);groups.setdefault(key,[]).append(record)
        assert record['seed']==20261110+10000*list(SCENARIOS).index(record['scenario'])+record['rep']
        assert len(fit['starts'])==3
        r=np.array(fit['responsibility']);params={k:np.array(v) for k,v in fit['params'].items()}
        assert np.isfinite(r).all() and np.isfinite(fit['loglik']) and abs(params['pi'].sum()-1)<1e-10
        norm=float(np.max(abs(r.sum(1)-1)));assert norm<1e-10;maxnorm=max(maxnorm,norm)
        assert np.array(record['metrics']['confusion']).sum()==len(r)
        maxquad=max(maxquad,fit['quadrature_error'])
        if record['method']=='GVA':
            maxlocal=max(maxlocal,fit['local']['residual'])
            gap=fit['loglik']-fit['objective'];assert gap>=-1e-6
            min_elbo_gap=min(min_elbo_gap,gap)
        if not fit['converged']:warnings.append([*key,record['rep'],'not_converged',fit['message']])
        if not fit['optimizer_success']:warnings.append([*key,record['rep'],'optimizer_status'])
        if np.any(params['var']<.00500001) or np.any(params['var']>4.99999) or np.any(abs(params['beta'])>7.99999):warnings.append([*key,record['rep'],'parameter_boundary'])
        if record['method']=='MCEM':
            assert all((not t['accepted']) or t['lower']>0 for t in fit['trace'])
    assert len(groups)==31 and all(len(v)==5 for v in groups.values())
    for group in groups.values():assert sorted(r['rep'] for r in group)==list(range(5))
    for name in ('baseline','poisson','nb'):
        for rep in range(5):
            with np.load(runroot/f'robustness/{name}_{rep}_data.npz') as a,np.load(runroot/f'comparison/{name}_{rep}_data.npz') as b:
                assert set(a.files)==set(b.files)
                assert all(np.array_equal(a[k],b[k]) for k in a.files),'Methods used different data'
    rows=[];summaries=[]
    for rec in records:
        row={k:rec[k] for k in ('scenario','rep','method','seed')};row.update({k:v for k,v in rec['metrics'].items() if k!='confusion'})
        ref=next(r for r in groups[rec['scenario'],'QML'] if r['rep']==rec['rep'])
        row['likelihood_gap_per_subject']=(ref['fit']['loglik']-rec['fit']['loglik'])/len(rec['fit']['responsibility'])
        rows.append(row)
    for key,recs in groups.items():
        g=dict(scenario=key[0],method=key[1],R=len(recs),converged=sum(r['fit']['converged'] for r in recs))
        for field in ('ari','error','beta_rmse','var_rmse','pi_rmse','seconds','likelihood_gap_per_subject'):
            vals=[r[field] for r in rows if (r['scenario'],r['method'])==key and r[field] is not None]
            g[field]=None if not vals else float(np.mean(vals));g[field+'_sd']=None if len(vals)<2 else float(np.std(vals,ddof=1))
        summaries.append(g)
    for name,data in [('replicates',rows),('summary',summaries)]:
        atomic(ROOT/f'results/{name}.json',data)
        with (ROOT/f'results/{name}.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    G={(r['scenario'],r['method']):r for r in summaries}
    fmt=lambda g,k:f"{g[k]:.3f}" if g[k] is not None else '--'
    ari=lambda g:f"{g['ari']:.3f}$\\pm${g['ari_sd']:.3f}"
    t=[r'\begin{table}[H]\centering\small',r'\caption{全部14场景的5次重复。ARI为均值$\pm$标准差；RMSE为GVA跨重复均值，误设$K$不比较参数。收敛数按QML/GVA，分母各为5。}',r'\begin{tabular}{lcccccc}\toprule',r'场景 & ARI(QML) & ARI(GVA) & $\beta$ RMSE & $D$ RMSE & $\pi$ RMSE & 收敛\\\midrule']
    for name in SCENARIOS:
        q,g=G[name,'QML'],G[name,'GVA'];t.append(f"{LABELS[name]} & {ari(q)} & {ari(g)} & {fmt(g,'beta_rmse')} & {fmt(g,'var_rmse')} & {fmt(g,'pi_rmse')} & {q['converged']}/{g['converged']} \\\\")
    t+=[r'\bottomrule\end{tabular}\end{table}'];(ROOT/'results/robustness_table.tex').write_text('\n'.join(t)+'\n')
    t=[r'\begin{table}[H]\centering\small',r'\caption{三类响应的同数据比较。各格为5次均值；耗时为三起点总墙钟秒数。$\Delta\ell/n=(\ell_{\mathrm{QML}}-\ell_{\mathrm{fit}})/n$，fit表示对应方法，可因不同局部解出现负值，不是KL散度。}',r'\begin{tabular}{llcccccc}\toprule',r'响应 & 方法 & ARI & 错分率 & $\beta$ RMSE & 秒 & 收敛/5 & $\Delta\ell/n$\\\midrule']
    for name in ('baseline','poisson','nb'):
        for method in ('QML','GVA','MCEM'):
            g=G[name,method];t.append(f"{LABELS[name]} & {method} & {ari(g)} & {g['error']:.3f} & {g['beta_rmse']:.3f} & {g['seconds']:.1f} & {g['converged']} & {g['likelihood_gap_per_subject']:.5f} \\\\")
    t+=[r'\bottomrule\end{tabular}\end{table}'];(ROOT/'results/comparison_table.tex').write_text('\n'.join(t)+'\n')
    b=G['baseline','GVA'];hard=G['hard','GVA'];orig=G['original','GVA']
    (ROOT/'results/abstract_result.tex').write_text(f"基准中GVA的平均ARI为{b['ari']:.3f}，积分参照为{G['baseline','QML']['ari']:.3f}；分离减半时GVA降至{hard['ari']:.3f}。")
    text=(f"基准GVA的平均ARI为{b['ari']:.3f}、错分率{b['error']:.3f}；分离减半和作业建议参数时ARI为{hard['ari']:.3f}、{orig['ari']:.3f}。"
          f"全部155次拟合中，{sum(r['fit']['converged'] for r in records)}次满足各自停止标准；未收敛和边界记录完整保留。"
          "各因素的实测结果见表，不从5次重复推断普遍优势。不同目标的差异、同一目标的积分误差和参数恢复误差分别评价。")
    text+=f"Poisson场景GVA三起点平均耗时{G['poisson','GVA']['seconds']:.1f}秒、收敛{G['poisson','GVA']['converged']}/5；其解析期望与剖面优化的实现不再采用旧版高维局部数值优化，但新旧硬件/容差不同，不能直接用旧耗时计算加速比。"
    scores=[r['fit']['gradient'] for r in records if r['method']=='MCEM']
    text+=f"MCEM所选解的每个体观测目标投影得分最大范数范围为{min(scores):.4g}至{max(scores):.4g}；小Q增量并不直接保证观测得分为零，表中的剩余似然差也须同时查看。"
    text+='MCEM在二元、Poisson、负二项下的方差参数RMSE依次为'+ '、'.join(f"{G[n,'MCEM']['var_rmse']:.3f}" for n in ('baseline','poisson','nb'))+'，混合比例RMSE依次为'+ '、'.join(f"{G[n,'MCEM']['pi_rmse']:.3f}" for n in ('baseline','poisson','nb'))+'。'
    boundary=sum(1 for w in warnings if w[-1]=='parameter_boundary')
    inner=sum(1 for w in warnings if w[-1]=='optimizer_status')
    if inner:text+=f"另有{inner}次所选起点记录过内部求解器非成功状态，与外层停止标记分开保存。"
    text+=(f"有{boundary}次所选解触及预设方差或固定效应边界，参数解释受约束影响。" if boundary else "所选参数未触及预设方差或固定效应边界。")
    text+=(f"样本量100至1000时，GVA固定效应RMSE由{G['n100','GVA']['beta_rmse']:.3f}变为{G['n1000','GVA']['beta_rmse']:.3f}；"
           f"序列10至50次时ARI由{G['T10','GVA']['ari']:.3f}变为{G['T50','GVA']['ari']:.3f}。"
           f"方差0.1与1.5的ARI为{G['var01','GVA']['ari']:.3f}/{G['var15','GVA']['ari']:.3f}；"
           f"拟合2簇与4簇为{G['K2','GVA']['ari']:.3f}/{G['K4','GVA']['ari']:.3f}，不规则时间为{G['irregular','GVA']['ari']:.3f}。")
    start_spreads={}
    for method in ('QML','GVA','MCEM'):
        vals=[np.ptp([s['objective'] for s in r['fit']['starts']])/len(r['fit']['responsibility']) for r in groups['baseline',method]]
        start_spreads[method]=dict(minimum=float(min(vals)),maximum=float(max(vals)))
    text+='基准5份数据中，每个体终目标的三起点极差最大值，QML、GVA、MCEM依次为'+ '、'.join(f"{start_spreads[m]['maximum']:.4g}" for m in ('QML','GVA','MCEM'))+'；各方法按自己的目标衡量，MCEM差异还含MC波动，不能据此证明全局最优。'
    (ROOT/'results/findings.tex').write_text(text)
    mc=[]
    for name in ('baseline','poisson','nb'):
        v=json.loads((mcroot/f'mcmc/{name}.json').read_text());assert (mcroot/f'mcmc/{name}.chains.npz').exists()
        assert v['kernel']=='joint_independence_MH'
        assert v['target_record_sha256']==hashlib.sha256((runroot/f'robustness/{name}_0_QML.json').read_bytes()).hexdigest()
        assert v['sampler_source_sha256']==hashlib.sha256((mcroot/'work/revision_v2/src/conditional_mcmc.py').read_bytes()).hexdigest()
        target=next(r for r in groups[name,'QML'] if r['rep']==0)['fit']['params']
        with np.load(mcroot/f'mcmc/{name}.chains.npz') as samples:
            assert samples['b'].shape==(4,8000,30) and samples['u'].shape==(4,8000,30)
            assert all(np.array_equal(samples[k],target[k]) for k in ('beta','var','pi'))
        mc.append((name,v))
    t=[r'\begin{table}[H]\centering\small',r'\caption{同一固定QML参数下，30个体条件分布的核验。误差对101点积分计算；成对数值为MCMC/GVA。方差比为各个体边缘$b_i$方差之和相对于积分的比。}',r'\begin{tabular}{lccccc}\toprule',r'响应 & MH接受率 & 最大split $\widehat R$ & 均值RMSE & 责任度RMSE & 方差比\\\midrule']
    for name,v in mc:
        t.append(f"{LABELS[name]} & {v['acceptance']:.3f} & {v['max_split_rhat']:.4f} & {v['mcmc_mean_rmse']:.4f}/{v['gva_mean_rmse']:.4f} & {v['mcmc_responsibility_rmse']:.4f}/{v['gva_responsibility_rmse']:.4f} & {v['mcmc_variance_ratio']:.3f}/{v['gva_variance_ratio']:.3f} \\\\")
    t+=[r'\bottomrule\end{tabular}\end{table}'];(ROOT/'results/mcmc_table.tex').write_text('\n'.join(t)+'\n')
    checks=json.loads((runroot/'work/revision_v2/qa/validation.json').read_text());atomic(ROOT/'qa/validation.json',checks)
    gh_errors=[]
    for name in SCENARIOS:
        rec=next(v for v in groups[name,'GVA'] if v['rep']==0);p={k:np.array(v) for k,v in rec['fit']['params'].items()}
        data=dict(np.load(runroot/f'robustness/{name}_0_data.npz'));data['family']=str(data['family']);d=prepared(data)
        value,_,_=profile(d,pack(p),len(p['pi']),'GVA',61,True)
        gh_errors.append(dict(scenario=name,absolute_ELBO_difference=float(abs(value-rec['fit']['objective']))))
    atomic(ROOT/'qa/gh_order_checks.json',gh_errors)
    ratios=[checks[k]['empirical_to_delta_sd_ratio'] for k in ('binary','poisson','nb')]
    txt=f"在小规模80次独立IS检验中，经验标准差与delta方法预测标准差之比分别为{ratios[0]:.3f}、{ratios[1]:.3f}、{ratios[2]:.3f}。这是误差公式的有限验证，不是所有场景的覆盖率证明。全部所选参数的61与101点积分log似然最大差为{maxquad:.3g}。"
    actual=0;ceiling=0
    for rec in records:
        if rec['method']!='MCEM':continue
        n=len(rec['fit']['responsibility']);K=len(rec['fit']['params']['pi'])
        for start in rec['fit']['starts']:
            actual+=2*n*K*sum(t['M'] for t in start['trace']);ceiling+=2*n*K*8192*len(start['trace'])
    txt+=f"全部MCEM起点实际生成{actual:,}个提议值；若对相同的已发生迭代都固定使用每层8192，则为{ceiling:,}个，前者占{actual/ceiling:.3f}。这仅比较相同迭代数的抽样量，不是固定样本算法达到同等精度的时间比。"
    txt+=f"14场景第一次重复的GVA，以61点替换31点后重新求局部最优，剖面ELBO最大绝对差为{max(v['absolute_ELBO_difference'] for v in gh_errors):.3g}。"
    txt+='条件MCMC的个体均值批均值MCSE（对30人再平均）依次为'+ '、'.join(f"{v['mean_mcse']:.4f}" for _,v in mc)+'。'
    if any(v['max_split_rhat']>1.05 for _,v in mc):txt+='部分链的split诊断偏大，应延长链或改进混合，不能将该MCMC结果视为已收敛真值。'
    txt+='本例GVA与积分的差异很小，有限MCMC仍有抽样误差；这不表示变分推断普遍比充分计算的MCMC更准确。这些结果针对固定参数的潜变量目标，未构造全贝叶斯参数区间。'
    (ROOT/'results/validation_text.tex').write_text(txt)
    # Figures use frozen baseline replicate zero, no truth-based initialization.
    bymethod={m:next(r for r in groups['baseline',m] if r['rep']==0) for m in ('QML','GVA','MCEM')}
    data=dict(np.load(runroot/'robustness/baseline_0_data.npz'))
    from scipy.optimize import linear_sum_assignment
    fitbeta=np.array(bymethod['GVA']['fit']['params']['beta']);truth=data['beta'];rr,cc=linear_sum_assignment(((truth[:,None]-fitbeta[None])**2).sum(2));tt=np.linspace(-1,1,100)
    fig,ax=plt.subplots(figsize=(7,3))
    for k,j in zip(rr,cc):
        line=ax.plot(tt,expit(truth[k,0]+truth[k,1]*tt),label=f'True {k+1}')[0]
        ax.plot(tt,expit(fitbeta[j,0]+fitbeta[j,1]*tt),'--',color=line.get_color(),label=f'GVA {k+1}')
    ax.set(xlabel='Time',ylabel='P(Y=1 | b=0)');ax.legend(ncol=3,fontsize=8);fig.tight_layout();fig.savefig(ROOT/'figures/trajectories.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(9,3))
    for method in ('QML','GVA'):
        tr=bymethod[method]['fit']['trace'];y=np.array([v['objective'] for v in tr]);axes[0].plot(np.arange(1,len(y)+1),y-y[-1],label=method)
    tr=bymethod['MCEM']['fit']['trace'];x=np.arange(1,len(tr)+1)
    axes[1].errorbar(x,[v['delta_Q'] for v in tr],yerr=[1.96*v['mcse'] for v in tr],fmt='.',capsize=1);axes[1].axhline(0,color='k',lw=.5);axes[1].set_yscale('symlog',linthresh=.001)
    axes[2].step(x,[v['M'] for v in tr],where='post')
    axes[0].set_ylabel('Objective minus final');axes[0].legend();axes[1].set_ylabel('Validation Q increment');axes[2].set_ylabel('Draws per stratum')
    for ax in axes:ax.set_xlabel('Iteration');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(ROOT/'figures/diagnostics.pdf');plt.close(fig)
    # Same theta, same selected subject; show why acceptance alone is insufficient.
    old=np.load(runroot/'probe_mcmc/poisson.chains.npz');new=np.load(mcroot/'mcmc/poisson.chains.npz')
    assert all(np.array_equal(old[k],new[k]) for k in ('beta','var','pi','indices'))
    from conditional_mcmc import split_rhat
    subject=int(np.argmax(split_rhat(old['b'])))
    fig,axes=plt.subplots(1,2,figsize=(8,2.6),sharey=True)
    for ax,sample,title in zip(axes,(old['b'],new['b']),('Conditional updates','Joint independence MH')):
        for c in range(4):ax.plot(np.arange(0,sample.shape[1],20),sample[c,::20,subject],lw=.5,label=f'Chain {c+1}')
        ax.set(title=title,xlabel='Saved draw',ylabel='Random intercept')
    axes[1].legend(ncol=2,fontsize=7);fig.tight_layout();fig.savefig(ROOT/'figures/mcmc_trace.pdf');plt.close(fig)
    audit_subject=subject
    audit=dict(fits=155,distinct_datasets=70,converged={m:sum(r['fit']['converged'] for r in records if r['method']==m) for m in ('QML','GVA','MCEM')},warnings=warnings,max_responsibility_error=maxnorm,max_quadrature_difference=maxquad,max_local_vi_residual=maxlocal,conditional_mcmc=[dict(scenario=n,**v) for n,v in mc],source_snapshot=(runroot/'source_snapshot.txt').read_text().strip(),mcmc_snapshot=(mcroot/'source_snapshot.txt').read_text().strip())
    audit['mcmc_trace_subject_index']=audit_subject
    audit['minimum_loglik_minus_approximate_ELBO']=min_elbo_gap
    audit['same_data_comparisons_verified']=15
    audit['baseline_multistart_objective_spread_per_subject']=start_spreads
    atomic(ROOT/'qa/audit.json',audit);print(json.dumps({k:v for k,v in audit.items() if k not in ('warnings','conditional_mcmc')},indent=2));print('Warning records:',len(warnings))

if __name__=='__main__':main()
