"""Validate and plot actual supplementary parameter/likelihood trajectories."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment
import inference as core
from convergence import PROTOCOL, load_case, native, window_diagnostics, replay_comparison
from experiment import atomic

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ('baseline', 'poisson', 'nb')
METHODS = ('QML', 'GVA', 'MCEM')
LABEL = dict(baseline='二元', poisson='Poisson', nb='负二项')


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def validate(record):
    original, data, provenance = load_case(record['scenario'], record['rep'], record['method'])
    assert provenance == record['provenance']
    assert record['protocol'] == PROTOCOL
    points = record['points']; assert [p['iteration'] for p in points] == list(range(len(points)))
    ell = np.array([p['loglik'] for p in points]); assert np.isfinite(ell).all()
    parameters = [{k: np.array(v) for k, v in p['params'].items()} for p in points]
    vectors = np.array([native(p) for p in parameters]); assert np.isfinite(vectors).all()
    for p in parameters:
        assert abs(p['pi'].sum()-1) < 1e-10 and (p['pi'] > 0).all()
        assert (p['var'] >= .005-1e-10).all() and (p['var'] <= 5+1e-10).all()
    rho = abs(np.diff(ell))/(1+abs(ell[:-1]))
    change = np.max(abs(np.diff(vectors, axis=0))/(1+abs(vectors[:-1])), axis=1)
    assert np.allclose(rho, [p['likelihood_relative_change'] for p in points[1:]], atol=1e-15, rtol=1e-10)
    assert np.allclose(change, [p['parameter_scaled_change'] for p in points[1:]], atol=1e-15, rtol=1e-10)
    for i, p in enumerate(points[1:], 1):
        if record['method'] == 'MCEM':
            assert p['accepted'] == (p['delta_Q']-1.96*p['mcse'] > 0)
            if not p['accepted']:
                assert np.array_equal(vectors[i], vectors[i-1])
        if record['method'] == 'GVA':
            assert p['objective'] <= p['loglik']+1e-6
    diagnostic = window_diagnostics(points)
    assert all(record[k] == v for k, v in diagnostic.items())
    # Independent high-order likelihood checks at initial, original-length, and final points.
    d = core.prepared(data); errors = []; high_order = {}
    for i in sorted({0, min(record['original_iteration'], len(points)-1), len(points)-1}):
        value = float(core.estep(d, parameters[i], 101)[0])
        high_order[i] = value
        errors.append(abs(value-ell[i])); assert errors[-1] < 1e-6
    assert abs(high_order[len(points)-1]-record['final_loglik_101']) < 1e-7
    original_index = record['original_iteration'] if record['method'] == 'MCEM' else len(points)-1
    if original_index < len(points):
        comparison = replay_comparison(parameters[original_index], high_order[original_index], original, d)
        saved = record['at_original_iteration'] if record['method'] == 'MCEM' else record['replay_comparison']
        for key, value in comparison.items():
            assert np.isclose(saved[key], value, rtol=1e-7, atol=1e-12), key
        if record['method'] == 'MCEM':
            assert all(saved[k] == v for k, v in window_diagnostics(points[:original_index+1]).items())
    return dict(high_order_loglik_max_difference=max(errors),
                observed_likelihood_decreases=int((np.diff(ell) < -1e-7).sum()),
                input_files_verified=True)


def parameter_figure(records, scenario, destination):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:3]
    fig, axes = plt.subplots(3, 3, figsize=(7.8, 6.2))
    for row, method in enumerate(METHODS):
        record = records[scenario, 0, method]; points = record['points']
        _, data, _ = load_case(scenario, 0, method)
        final_beta = np.array(points[-1]['params']['beta'])
        rr, cc = linear_sum_assignment(((data['beta'][:, None]-final_beta[None])**2).sum(2))
        permutation = cc[np.argsort(rr)]
        x = np.arange(len(points))
        beta = np.array([p['params']['beta'] for p in points])[:, permutation]
        var = np.array([p['params']['var'] for p in points])[:, permutation]
        pi = np.array([p['params']['pi'] for p in points])[:, permutation]
        for k, color in enumerate(colors):
            axes[row, 0].plot(x, beta[:, k, 0], color=color, lw=1)
            axes[row, 0].plot(x, beta[:, k, 1], color=color, ls='--', lw=1)
            axes[row, 1].plot(x, var[:, k], color=color, lw=1)
            axes[row, 2].plot(x, pi[:, k], color=color, lw=1)
        axes[row, 0].set_ylabel(method)
        if method == 'MCEM':
            original = record['original_iteration']
            for ax in axes[row]:
                ax.axvline(original, color='0.4', ls=':', lw=.8)
                if x[-1] > original:
                    ax.axvspan(original, x[-1], color='0.93', zorder=-10)
        for ax in axes[row]:
            ax.grid(alpha=.2); ax.tick_params(labelsize=8)
    for ax, title in zip(axes[0], ('Fixed effects', 'Variance D', 'Mixing proportions')):
        ax.set_title(title, fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel('Iteration', fontsize=9)
    handles = [Line2D([0], [0], color=c, label=f'Cluster {i+1}') for i, c in enumerate(colors)]
    handles += [Line2D([0], [0], color='k', label='Intercept'), Line2D([0], [0], color='k', ls='--', label='Slope')]
    fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, 1)); fig.savefig(destination); plt.close(fig)


def likelihood_figure(records, destination):
    colors = dict(QML='tab:blue', GVA='tab:orange', MCEM='tab:green')
    fig, axes = plt.subplots(3, 2, figsize=(7.8, 6.7))
    for row, scenario in enumerate(FAMILIES):
        for method in METHODS:
            record = records[scenario, 0, method]; points = record['points']
            x = np.arange(len(points)); values = np.array([p['loglik'] for p in points])
            axes[row, 0].plot(x, values/record['n'], color=colors[method], label=method, lw=1)
            axes[row, 1].plot(x[1:], [p['likelihood_relative_change'] for p in points[1:]],
                              color=colors[method], label=method, lw=1)
        original = records[scenario, 0, 'MCEM']['original_iteration']
        for ax in axes[row]:
            ax.axvline(original, color='0.4', ls=':', lw=.8); ax.grid(alpha=.2); ax.tick_params(labelsize=8)
        axes[row, 0].set_ylabel(f'{scenario}\nlog likelihood / n', fontsize=8)
        axes[row, 1].set_ylabel('Relative likelihood change', fontsize=8)
        axes[row, 1].set_yscale('symlog', linthresh=1e-8)
        axes[row, 1].set_ylim(bottom=0)
        axes[row, 1].axhline(PROTOCOL['likelihood_relative_tolerance'], color='tab:red', ls='--', lw=.9)
    axes[0, 0].legend(fontsize=8, frameon=False)
    axes[0, 0].set_title('Observed-data log likelihood', fontsize=10)
    axes[0, 1].set_title('Threshold: 1e-6', fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel('Iteration', fontsize=9)
    fig.tight_layout(); fig.savefig(destination); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--input', type=Path, default=ROOT/'data/convergence')
    ap.add_argument('--output-root', type=Path, default=ROOT); args = ap.parse_args()
    source = args.input.resolve(); output = args.output_root.resolve()
    status = json.loads((source/'STATUS.json').read_text())
    assert status['status'] == 'COMPLETE' and status['complete'] == status['expected'] == 21
    metadata = json.loads((source/'metadata.json').read_text())
    for name, value in metadata['spec']['sources'].items():
        assert digest(ROOT/'src'/name) == value, name
    expected = [(s, r, m) for s in FAMILIES for m in METHODS for r in (range(5) if m == 'MCEM' else range(1))]
    records = {}; checks = {}; hashes = {}
    for key in expected:
        path = source/f'{key[0]}_{key[1]}_{key[2]}.json'; record = json.loads(path.read_text())
        assert (record['scenario'], record['rep'], record['method']) == key
        records[key] = record; checks[path.name] = validate(record); hashes[path.name] = digest(path)
    local = json.loads((source/'local_execution.json').read_text())
    remote = json.loads((source/'server_execution.json').read_text())
    assert local['environment'] == 'yellow' and remote['environment'] == 'base'
    assert remote['completed_before_remote_execution'] == local['completed_files']
    assert len(local['completed_files']) == 11 and len(remote['scheduled_remote_tasks']) == 10
    assert set(local['completed_files']) | set(remote['scheduled_remote_tasks']) == set(hashes)
    for name, value in local['completed_files'].items():
        assert hashes[name] == value
    for folder in ('results', 'figures', 'qa'):
        (output/folder).mkdir(parents=True, exist_ok=True)
    rows = []
    for key, record in records.items():
        old = record.get('at_original_iteration') or {}
        row = dict(scenario=key[0], rep=key[1], method=key[2],
                   original_iterations=record['original_iteration'], final_iterations=len(record['points'])-1,
                   additional_iterations=max(0, len(record['points'])-1-record['original_iteration']),
                   at_original_likelihood_stable=old.get('likelihood_stable'),
                   at_original_max_relative_change=old.get('window_max_likelihood_relative_change'),
                   likelihood_stable=record['likelihood_stable'], parameter_stable=record['parameter_stable'],
                   max_relative_change=record.get('window_max_likelihood_relative_change'),
                   max_scaled_parameter_change=record.get('window_max_parameter_scaled_change'),
                   accepted_updates_in_final_window=record.get('window_accepted_updates'),
                   reason=record['reason'], seconds=record['seconds'])
        rows.append(row)
    atomic(output/'results/convergence_summary.json', rows)
    with (output/'results/convergence_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    table = [r'\begin{table}[H]\centering\small',
             r'\caption{15次MCEM的补充诊断。通过指连续3轮相对似然变化不超过$10^{-6}$，且末窗至少接受一次更新；参数平稳用缩放变化$10^{-3}$，仅为辅助。每个响应5份数据，延长最多30轮；全拒绝窗另列，预算耗尽不记通过。}',
             r'\begin{tabular}{lccccc}\toprule',
             r'响应 & 原轮数处通过 & 补充末窗通过 & 参数平稳 & 全拒绝窗 & 延长轮数\\\midrule']
    for scenario in FAMILIES:
        group = [r for r in rows if r['scenario'] == scenario and r['method'] == 'MCEM']
        before = sum(bool(r['at_original_likelihood_stable']) for r in group)
        after = sum(r['likelihood_stable'] for r in group); params = sum(r['parameter_stable'] for r in group)
        rejected = sum(r['accepted_updates_in_final_window'] == 0 for r in group)
        lo, hi = min(r['additional_iterations'] for r in group), max(r['additional_iterations'] for r in group)
        table.append(f"{LABEL[scenario]} & {before}/5 & {after}/5 & {params}/5 & {rejected} & {lo}--{hi} \\\\")
    table += [r'\bottomrule\end{tabular}\end{table}']
    (output/'results/convergence_table.tex').write_text('\n'.join(table)+'\n')
    mcem = [r for r in rows if r['method'] == 'MCEM']; profiles = [r for r in rows if r['method'] != 'MCEM']
    comparisons = [v.get('at_original_iteration') or {} for k, v in records.items() if k[2] == 'MCEM']
    matches = sum(bool(v.get('numerically_matches_saved_fit')) for v in comparisons)
    text = (f"6次积分/GVA代表性补充轨迹中，{sum(r['likelihood_stable'] for r in profiles)}次通过本次观测似然窗口检查。"
            f"15次MCEM中，原记录轮数处{sum(bool(r['at_original_likelihood_stable']) for r in mcem)}次、"
            f"补充末窗{sum(r['likelihood_stable'] for r in mcem)}次通过；"
            f"其余{sum(not r['likelihood_stable'] for r in mcem)}次保留为未通过该严格口径，不因曲线视觉平坦直接改记收敛。"
            f"原轮数处有{matches}/15次与保存参数和似然达到预设数值一致性（参数最大差小于$10^{{-4}}$、似然差/个体小于$10^{{-6}}$）。"
            "这些是同初始化的补充重跑与有限延长，不回填原始缺失历史，也不替换原表中的155次拟合或耗时。")
    failed = [r for r in mcem if not r['likelihood_stable'] and r['accepted_updates_in_final_window'] > 0]
    if failed:
        lo = min(r['max_relative_change'] for r in failed); hi = max(r['max_relative_change'] for r in failed)
        text += f"未通过且末窗有接受更新者的最大$\\rho_t$在{lo:.3g}至{hi:.3g}之间；全拒绝窗数量另见表。"
    (output/'results/convergence_text.tex').write_text(text+'\n')
    for scenario in FAMILIES:
        suffix = '' if scenario == 'baseline' else '_'+scenario
        parameter_figure(records, scenario, output/f'figures/parameter_convergence{suffix}.pdf')
    likelihood_figure(records, output/'figures/likelihood_convergence.pdf')
    audit = dict(protocol=PROTOCOL, records=21, mcem_records=15, input_record_sha256=hashes,
                 individual_checks=checks, matching_original_iteration_states=matches,
                 mcem_likelihood_stable=sum(r['likelihood_stable'] for r in mcem),
                 mcem_parameter_stable=sum(r['parameter_stable'] for r in mcem),
                 execution_environment_file_mapping_verified=True,
                 original_results_untouched=True)
    atomic(output/'qa/convergence_audit.json', audit)
    print(json.dumps({k: v for k, v in audit.items() if k not in ('input_record_sha256', 'individual_checks')}, indent=2))


if __name__ == '__main__':
    main()
