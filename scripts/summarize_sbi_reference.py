"""Summarize saved Stan draws against the corresponding BayesFlow inputs."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'build/matplotlib'))
sys.path.insert(0, str(ROOT/'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from sbi_model import atomic_json, constrain

LABELS = {'binary':'二元', 'poisson':'Poisson', 'nb':'负二项'}
NAMES = [f'{name}.{k}' for k in range(1,4) for name in ('intercept','slope')]
NAMES += [f'{name}.{k}' for name in ('variance','pi') for k in range(1,4)]


def load_chain(path):
    with path.open() as stream:
        lines = [line for line in stream if line.strip() and not line.startswith('#')]
    header = lines[0].strip().split(',')
    values = np.loadtxt(lines[1:], delimiter=',', ndmin=2)
    return {name:values[:,i] for i,name in enumerate(header)}


def study_case(run, summary, config, family):
    folder = run/family
    paths = [folder/f'chain_{i}.csv' for i in range(1,5)]
    diagnostics = folder/'stats.csv'
    with tempfile.TemporaryDirectory(dir=folder) as temporary, (folder/'stats.log').open('w') as log:
        new_diagnostics = Path(temporary)/'stats.csv'
        subprocess.run([str(summary), '--sig_figs=10', f'--csv_filename={new_diagnostics}', *map(str,paths)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
        new_diagnostics.replace(diagnostics)
    with diagnostics.open() as handle:
        indexed = {row['name']:row for row in csv.DictReader(line for line in handle if line.strip() and not line.startswith('#'))}
    chains = [load_chain(path) for path in paths]
    n = config['draws_per_chain']
    assert all(len(c['lp__'])==n for c in chains)
    draws = np.stack([np.stack([c[name] for name in NAMES], axis=-1) for c in chains])
    assert np.isfinite(draws).all()
    names = [f'{name[:-2]}[{name[-1]}]' for name in NAMES]
    components = {name: {k:float(indexed[name][k]) for k in ('R_hat','ESS_bulk','ESS_tail')} for name in names}
    max_rhat = max(v['R_hat'] for v in components.values())
    min_bulk = min(v['ESS_bulk'] for v in components.values())
    min_tail = min(v['ESS_tail'] for v in components.values())
    divergences = int(sum(c['divergent__'].sum() for c in chains))
    depth_hits = int(sum((c['treedepth__']>=config['max_depth']).sum() for c in chains))
    bfmi = [float(np.square(np.diff(c['energy__'])).mean()/c['energy__'].var(ddof=1)) for c in chains]
    qualified = max_rhat < 1.01 and min(min_bulk,min_tail) >= 400 and divergences == 0 and depth_hits == 0 and min(bfmi) >= .3
    saved = ROOT/'data/sbi_extension/comparison'/family
    source_hash = json.loads((saved/'metadata.json').read_text())['comparison_inputs_sha256'][config['cases'][family]['name']]
    assert source_hash == config['cases'][family]['sha256']
    with np.load(saved/'comparison_predictions.npz') as sbi:
        d2_sbi = constrain(sbi['theta'][config['rep']])['var'][:,1]
    record = dict(dataset=config['cases'][family]['name'], max_global_rhat=max_rhat, min_global_bulk_ess=min_bulk,
                  min_global_tail_ess=min_tail, divergences=divergences, at_max_tree_depth=depth_hits, e_bfmi=bfmi,
                  qualified=qualified, d2_means_by_chain=draws[:,:,7].mean(1).tolist(),
                  d2_sbi_95_interval=np.quantile(d2_sbi,[.025,.975]).tolist(), diagnostics=components,
                  chain_info=[json.loads((folder/f'status_{i}.json').read_text()) for i in range(1,5)],
                  diagnostic_csv_sha256=hashlib.sha256(diagnostics.read_bytes()).hexdigest(),
                  chain_csv_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    if qualified:
        record['d2_nuts_95_interval'] = np.quantile(draws[:,:,7],[.025,.975]).tolist()
    return record,draws


def write_table(cases):
    lines = [r'\begin{table}[H]\centering\small',
             r'\caption{同先验的 NUTS 诊断，每族第一次重复，共四链。$\widehat R$ 和 ESS 在 12 个全局分量上汇总。}',
             r'\begin{tabular}{lrrrrrrl}\toprule',
             r'响应 & $\max\widehat R$ & $\min$ bulk ESS & $\min$ tail ESS & 发散 & 达最大树深度 & $\min$ E-BFMI & 通过 \\\midrule']
    for family, case in cases.items():
        fields = [LABELS[family], f"{case['max_global_rhat']:.3f}", f"{case['min_global_bulk_ess']:.1f}",
                  f"{case['min_global_tail_ess']:.1f}", str(case['divergences']), str(case['at_max_tree_depth']),
                  f"{min(case['e_bfmi']):.2f}", '是' if case['qualified'] else '否']
        lines.append(' & '.join(fields)+r' \\')
    lines += [r'\bottomrule\end{tabular}\end{table}']
    (ROOT/'results/sbi_reference_table.tex').write_text('\n'.join(lines)+'\n')


def plot_trace(draws):
    fig, axes = plt.subplots(1,3, figsize=(10.6,2.5), layout='constrained')
    for axis,(family,values) in zip(axes,draws.items()):
        x = np.arange(1,values.shape[1]+1,5)
        for i,chain in enumerate(values[:,:,7]):
            axis.plot(x, chain[::5], linewidth=.55, alpha=.8, label=f'Chain {i+1}')
        axis.set_title({'binary':'Binary','poisson':'Poisson','nb':'Negative binomial'}[family])
        axis.set_xlabel('Retained iteration')
        axis.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel(r'$D_2$')
    axes[2].legend(loc='upper right', fontsize=7, ncol=2)
    fig.savefig(ROOT/'figures/sbi_reference_trace.pdf',metadata={'CreationDate':None})


def main(run, stansummary):
    config = json.loads((run/'configuration.json').read_text())
    cases, draws = {}, {}
    for family in config['cases']:
        cases[family],draws[family] = study_case(run, stansummary, config, family)
    atomic_json(ROOT/'results/sbi_reference_audit.json', {'configuration':config,'cases':cases})
    np.savez_compressed(ROOT/'data/sbi_reference/global_draws.npz', names=np.array(NAMES), **draws)
    write_table(cases)
    plot_trace(draws)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('stansummary', type=Path)
    args = parser.parse_args()
    main(args.run, args.stansummary)
