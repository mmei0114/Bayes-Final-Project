"""Consolidate only the frozen cloud main suite; do not mix pilots/v1 results."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
from scipy.special import expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import atomic,SCENARIOS

ROOT=Path(__file__).resolve().parents[1]
LABELS={'baseline':'基准','hard':'分离减半','original':'作业建议参数','n100':r'$n=100$','n1000':r'$n=1000$','T10':r'$T=10$','T50':r'$T=50$','K2':r'$K=2$','K4':r'$K=4$','var01':r'$D=0.1$','var15':r'$D=1.5$','poisson':'Poisson','nb':'负二项','irregular':'不规则时间'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-root',default=str(ROOT.parent/'cloud/received/runs/revision_v2_main_20261110'));ap.add_argument('--mcmc-root',default=str(ROOT.parent/'cloud/received/runs/revision_v2_diagnostic_20261110'));args=ap.parse_args();runroot=Path(args.run_root);mcroot=Path(args.mcmc_root)
    records=[]
    for folder,expected in [('robustness',140),('comparison',15)]:
        status=json.loads((runroot/folder/'STATUS.json').read_text());assert status['status']=='COMPLETE' and status['complete']==expected
        metadata=json.loads((runroot/folder/'metadata.json').read_text())
        for name,digest in metadata['spec']['source'].items():assert hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest()==digest,name
        subset=[json.loads(p.read_text()) for p in sorted((runroot/folder).glob('*.json')) if not any(s in p.name for s in ('.checkpoint.','.error.'))]
        subset=[v for v in subset if 'fit' in v];assert len(subset)==expected
        records+=subset
    assert len(records)==155
    groups={};warnings=[];maxnorm=0.;maxquad=0.;maxlocal=0.
    for record in records:
        fit=record['fit'];key=(record['scenario'],record['method']);groups.setdefault(key,[]).append(record)
        r=np.array(fit['responsibility']);params={k:np.array(v) for k,v in fit['params'].items()}
        assert np.isfinite(r).all() and np.isfinite(fit['loglik']) and abs(params['pi'].sum()-1)<1e-10
        norm=float(np.max(abs(r.sum(1)-1)));assert norm<1e-10;maxnorm=max(maxnorm,norm)
        assert np.array(record['metrics']['confusion']).sum()==len(r)
        maxquad=max(maxquad,fit['quadrature_error'])
        if record['method']=='GVA':maxlocal=max(maxlocal,fit['local']['residual'])
        if not fit['converged']:warnings.append([*key,record['rep'],'not_converged',fit['message']])
        if not fit['optimizer_success']:warnings.append([*key,record['rep'],'optimizer_status'])
        if np.any(params['var']<.00500001) or np.any(params['var']>4.99999) or np.any(abs(params['beta'])>7.99999):warnings.append([*key,record['rep'],'parameter_boundary'])
        if record['method']=='MCEM':
            assert all((not t['accepted']) or t['lower']>0 for t in fit['trace'])
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
    t=[r'\begin{table}[H]\centering\small',r'\caption{三类响应的同数据比较。各格为5次均值；耗时为三起点总墙钟秒数。$\Delta\ell/n=(\ell_{QML}-\ell_{方法})/n$，可因不同局部解出现负值，不是KL散度。}',r'\begin{tabular}{llcccccc}\toprule',r'响应 & 方法 & ARI & 错分率 & $\beta$ RMSE & 秒 & 收敛/5 & $\Delta\ell/n$\\\midrule']
    for name in ('baseline','poisson','nb'):
        for method in ('QML','GVA','MCEM'):
            g=G[name,method];t.append(f"{LABELS[name]} & {method} & {ari(g)} & {g['error']:.3f} & {g['beta_rmse']:.3f} & {g['seconds']:.1f} & {g['converged']} & {g['likelihood_gap_per_subject']:.5f} \\\\")
    t+=[r'\bottomrule\end{tabular}\end{table}'];(ROOT/'results/comparison_table.tex').write_text('\n'.join(t)+'\n')
    b=G['baseline','GVA'];hard=G['hard','GVA'];orig=G['original','GVA']
    text=(f"基准GVA的平均ARI为{b['ari']:.3f}、错分率{b['error']:.3f}；分离减半和作业建议参数时ARI为{hard['ari']:.3f}、{orig['ari']:.3f}。"
          f"全部155次拟合中，{sum(r['fit']['converged'] for r in records)}次满足各自停止标准；未收敛和边界记录完整保留。"
          "各因素的实测结果见表，不从5次重复推断普遍优势。不同目标的差异、同一目标的积分误差和参数恢复误差分别评价。")
    text+=f"Poisson场景GVA三起点平均耗时{G['poisson','GVA']['seconds']:.1f}秒、收敛{G['poisson','GVA']['converged']}/5；其解析期望与剖面优化的实现不再采用旧版高维局部数值优化，但新旧硬件/容差不同，不能直接用旧耗时计算加速比。"
    (ROOT/'results/findings.tex').write_text(text)
    mc=[]
    for name in ('baseline','poisson','nb'):
        v=json.loads((mcroot/f'mcmc/{name}.json').read_text());assert (mcroot/f'mcmc/{name}.chains.npz').exists()
        assert v['kernel']=='joint_independence_MH'
        assert v['target_record_sha256']==hashlib.sha256((runroot/f'robustness/{name}_0_QML.json').read_bytes()).hexdigest()
        mc.append((name,v))
    t=[r'\begin{table}[H]\centering\small',r'\caption{同一固定QML参数下，30个体条件分布的核验。误差对101点积分计算；成对数值为MCMC/GVA。方差比为各个体边缘$b_i$方差之和相对于积分的比。}',r'\begin{tabular}{lccccc}\toprule',r'响应 & MH接受率 & 最大split $\widehat R$ & 均值RMSE & 责任度RMSE & 方差比\\\midrule']
    for name,v in mc:
        t.append(f"{LABELS[name]} & {v['acceptance']:.3f} & {v['max_split_rhat']:.4f} & {v['mcmc_mean_rmse']:.4f}/{v['gva_mean_rmse']:.4f} & {v['mcmc_responsibility_rmse']:.4f}/{v['gva_responsibility_rmse']:.4f} & {v['mcmc_variance_ratio']:.3f}/{v['gva_variance_ratio']:.3f} \\\\")
    t+=[r'\bottomrule\end{tabular}\end{table}'];(ROOT/'results/mcmc_table.tex').write_text('\n'.join(t)+'\n')
    checks=json.loads((runroot/'work/revision_v2/qa/validation.json').read_text());atomic(ROOT/'qa/validation.json',checks)
    ratios=[checks[k]['empirical_to_delta_sd_ratio'] for k in ('binary','poisson','nb')]
    txt=f"在小规模80次独立IS检验中，经验标准差与delta方法预测标准差之比分别为{ratios[0]:.3f}、{ratios[1]:.3f}、{ratios[2]:.3f}。这是误差公式的有限验证，不是所有场景的覆盖率证明。全部所选参数的61与101点积分log似然最大差为{maxquad:.3g}。"
    txt+='条件MCMC的个体均值批均值MCSE（对30人再平均）依次为'+ '、'.join(f"{v['mean_mcse']:.4f}" for _,v in mc)+'。'
    if any(v['max_split_rhat']>1.05 for _,v in mc):txt+='部分链的split诊断偏大，应延长链或改进混合，不能将该MCMC结果视为已收敛真值。'
    txt+='这些结果针对固定参数的潜变量目标，未构造全贝叶斯参数区间。'
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
    fig,axes=plt.subplots(1,3,figsize=(11,3))
    for method in ('QML','GVA'):
        tr=bymethod[method]['fit']['trace'];y=np.array([v['objective'] for v in tr]);axes[0].plot(np.arange(1,len(y)+1),y-y[-1],label=method)
    tr=bymethod['MCEM']['fit']['trace'];x=np.arange(1,len(tr)+1)
    axes[1].errorbar(x,[v['delta_Q'] for v in tr],yerr=[1.96*v['mcse'] for v in tr],fmt='.',capsize=1);axes[1].axhline(0,color='k',lw=.5);axes[1].set_yscale('symlog',linthresh=.001)
    axes[2].step(x,[v['M'] for v in tr],where='post')
    axes[0].set_ylabel('Objective minus final');axes[0].legend();axes[1].set_ylabel('Validation Q increment');axes[2].set_ylabel('Draws per stratum')
    for ax in axes:ax.set_xlabel('Iteration');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(ROOT/'figures/diagnostics.pdf');plt.close(fig)
    audit=dict(fits=155,distinct_datasets=70,converged={m:sum(r['fit']['converged'] for r in records if r['method']==m) for m in ('QML','GVA','MCEM')},warnings=warnings,max_responsibility_error=maxnorm,max_quadrature_difference=maxquad,max_local_vi_residual=maxlocal,conditional_mcmc=[dict(scenario=n,**v) for n,v in mc],source_snapshot=(runroot/'source_snapshot.txt').read_text().strip(),mcmc_snapshot=(mcroot/'source_snapshot.txt').read_text().strip())
    atomic(ROOT/'qa/audit.json',audit);print(json.dumps({k:v for k,v in audit.items() if k not in ('warnings','conditional_mcmc')},indent=2));print('Warning records:',len(warnings))

if __name__=='__main__':main()
