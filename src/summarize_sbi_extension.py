"""Summarize the archived three-family SBI outputs using only NumPy and SciPy."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import inference as core
from legacy_model import evaluate
from sbi_model import atomic_json, constrain, native, prior_sample


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {'binary': 'baseline', 'poisson': 'poisson', 'nb': 'nb'}
NAMES = [f'beta{k}_{j}' for k in range(1, 4) for j in ('intercept', 'slope')]
NAMES += [f'{parameter}{k}' for parameter in ('D', 'pi') for k in range(1, 4)]
METRICS = ('ari', 'error', 'beta_rmse', 'var_rmse', 'pi_rmse')


def read_npz(path):
    with np.load(path) as data:
        return dict(data)


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict_inputs(root, split, expected):
    split_dir = root / split
    manifest = json.loads((split_dir / 'manifest.json').read_text())
    assert manifest == expected
    entries = []
    for name, digest in sorted(manifest['files'].items()):
        path = split_dir / name
        assert hash_file(path) == digest, path
        entries.append(read_npz(path))
    return {name: np.concatenate([entry[name] for entry in entries]) for name in entries[0]}


def convert_draws(array, datasets, draws):
    assert array.shape == (datasets, draws, 11)
    params = constrain(array)
    values = native(params)
    assert np.isfinite(values).all()
    gaps = np.diff(params['beta'][..., 0], axis=-1)
    assert (gaps >= 0.).all()
    assert ((params['var'] > .005) & (params['var'] < 5.)).all()
    assert np.allclose(params['pi'].sum(-1), 1.) and (params['pi'] > 0.).all()
    # The unconstrained coordinates still encode strictly positive gaps. Very
    # small gaps can round to zero after adding them to a float64 intercept.
    rounded = (gaps == 0.).any(axis=-1)
    rounding = dict(draws_with_rounded_intercept_ties=int(rounded.sum()),
                    datasets_with_rounded_intercept_ties=int(rounded.any(axis=1).sum()))
    return params, values, rounding


def check_probabilities(p, size):
    assert p.shape == (size, 300, 3)
    assert np.isfinite(p).all() and (p >= 0.).all() and np.allclose(p.sum(-1), 1., atol=1e-6)


def calibration_metrics(pred, original, draws, reference_variance):
    truths = native(dict(beta=original['beta'], var=original['var'], pi=original['pi']))
    _, values, rounding = convert_draws(pred['theta'], len(truths), draws)
    ranks = (values < truths[:, None, :]).sum(1)
    check_probabilities(pred['probabilities'], len(truths))
    rows = []
    for j, name in enumerate(NAMES):
        points = values[..., j]
        empirical = []
        for quantiles in ((.25, .75), (.025, .975)):
            lower, upper = np.quantile(points, quantiles, axis=1)
            empirical.append(float(np.mean((lower <= truths[:, j]) & (truths[:, j] <= upper))))
        rows.append(dict(parameter=name, coverage_50=empirical[0], coverage_95=empirical[1],
                         contraction=float(1. - points.var(1, ddof=1).mean() / reference_variance[j]),
                         prior_scaled_rmse=float(np.sqrt(np.mean((points.mean(1) - truths[:, j])**2) / reference_variance[j]))))
    bins = np.linspace(-.5, draws + .5, 21)
    p = pred['probabilities']
    encoded = np.eye(3)[original['u']]
    observed_prob = (p * encoded).sum(-1)
    audit = dict(size=len(truths), draws=draws, rounding=rounding,
                 rank_histograms={name: np.histogram(ranks[:, j], bins=bins)[0].tolist() for j, name in enumerate(NAMES)},
                 rank_bin_edges=bins.tolist(),
                 brier=float(np.mean(np.sum((p - encoded)**2, axis=-1))),
                 cross_entropy=float(-np.log(np.maximum(observed_prob, 1e-30)).mean()),
                 ordered_label_accuracy=float(np.mean(p.argmax(-1) == original['u'])))
    return rows, audit


def conditional_audit(pred, original, family):
    values = constrain(original['theta'])
    y = original['y']
    check_probabilities(pred, len(y))
    time = np.broadcast_to(np.linspace(-1., 1., y.shape[-1]), y.shape[1:])
    passed, omitted, deltas = [], [], []
    for i, response in enumerate(y):
        data = core.prepared(dict(y=response.astype(np.int64), t=time, mask=np.ones_like(time), family=family))
        params = {k: v[i] for k, v in values.items()}
        try:
            ll_61, p_61, *_ = core.estep(data, params, 61)
            ll_101, p_101, *_ = core.estep(data, params, 101)
            ll_delta = float(abs(ll_61 - ll_101))
            prob_delta = float(abs(p_61 - p_101).max())
            if not (ll_delta < 1e-6 and prob_delta < 1e-6):
                raise ValueError(f'61/101 node difference: {ll_delta:g}, {prob_delta:g}')
        except (RuntimeError, FloatingPointError, ValueError) as exc:
            omitted.append(dict(dataset=i, reason=str(exc)))
        else:
            passed.append(p_101)
            deltas.append((i, ll_delta, prob_delta))
    report = dict(checked=len(y), with_verified_reference=len(passed), omitted=omitted)
    if passed:
        gh = np.stack(passed)
        classifier = pred[[i for i, _, _ in deltas]]
        report.update(
            probability_rmse=float(np.sqrt(np.mean((classifier - gh)**2))),
            argmax_agreement=float(np.mean(classifier.argmax(-1) == gh.argmax(-1))),
            largest_log_likelihood_difference=max(value[1] for value in deltas),
            largest_probability_difference=max(value[2] for value in deltas),
        )
    return report


def classical_metrics(run, scenario):
    rows = []
    for rep in range(5):
        for method in ('MCEM', 'GVA', 'QML'):
            folder = 'comparison' if method == 'MCEM' else 'robustness'
            path = run / folder / f'{scenario}_{rep}_{method}.json'
            fit = json.loads(path.read_text())
            rows.append(dict(scenario=scenario, rep=rep, method=method,
                             **{name: fit['metrics'][name] for name in METRICS}))
    return rows


def comparison_metrics(pred, metadata, run, scenario):
    source = run / 'comparison'
    params, _, rounding = convert_draws(pred['theta'], 5, metadata['draws'])
    check_probabilities(pred['probabilities'], 5)
    rows = []
    for rep in range(5):
        name = f'{scenario}_{rep}_data.npz'
        path = source / name
        assert hash_file(path) == metadata['comparison_inputs_sha256'][name]
        classic_path = source.parent / 'robustness' / name
        data = read_npz(path)
        classic = read_npz(classic_path)
        assert set(data) == set(classic) and all(np.array_equal(data[k], classic[k]) for k in data), name
        fit = dict(responsibility=pred['probabilities'][rep],
                   params={k: v[rep].mean(axis=0) for k, v in params.items()},
                   seconds=pred['seconds'][rep], converged=None, trace=[], loglik=None)
        metrics = evaluate(data, fit, K=3)
        rows.append(dict(scenario=scenario, rep=rep, method='SBI', **{name: metrics[name] for name in METRICS}))
    return rows, rounding


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(received, destination, classic_root):
    prediction = received / 'sbi_extension_predict_20261118'
    count_data = received / 'sbi_extension_data_20261118/datasets'
    assert (prediction / 'STATE').read_text().strip() == 'COMPLETE'
    assert (prediction / 'EXIT_CODE').read_text().strip() == '0'
    reference_variance = native(prior_sample(np.random.default_rng(20263618), 100000)).var(0, ddof=1)
    calibration_rows, comparison_rows, calibration_audit = [], [], {}
    for family, scenario in FAMILIES.items():
        folder = prediction / family
        meta = json.loads((folder / 'metadata.json').read_text())
        assert meta['family'] == family and meta['calibration_datasets'] == 500 and meta['draws'] == 1024
        inputs = prediction / 'binary_data' if family == 'binary' else count_data / family
        original = predict_inputs(inputs, 'calibration', meta['calibration'])
        pred = read_npz(folder / 'calibration_predictions.npz')
        rows, audit = calibration_metrics(pred, original, meta['draws'], reference_variance)
        conditional_root = received / 'sbi_data_20261118/datasets' if family == 'binary' else inputs
        conditional_input = predict_inputs(conditional_root, 'classifier', meta['conditional'])
        conditional = read_npz(folder / 'conditional_predictions.npz')['probabilities']
        conditional_check = conditional_audit(conditional, conditional_input, family)
        calibration_rows.extend(dict(family=family, **row) for row in rows)
        calibration_audit[family] = dict(audit, selection=meta['selection'], gpu_inference_seconds=meta['comparison_seconds'],
                                         conditional_classifier=conditional_check)
        comparison = read_npz(folder / 'comparison_predictions.npz')
        comparison_rows.extend(classical_metrics(classic_root, scenario))
        rows, rounding = comparison_metrics(comparison, meta, classic_root, scenario)
        comparison_rows.extend(rows)
        calibration_audit[family]['comparison_rounding'] = rounding
    groups = []
    for scenario in FAMILIES.values():
        for method in ('MCEM', 'GVA', 'QML', 'SBI'):
            group = [r for r in comparison_rows if r['scenario'] == scenario and r['method'] == method]
            assert len(group) == 5
            groups.append(dict(scenario=scenario, method=method, datasets=len(group),
                               **{f'{k}_{label}': float(stat([r[k] for r in group]))
                                  for k in METRICS for label, stat in (('mean', np.mean), ('sd', lambda x: np.std(x, ddof=1)))}))
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / 'sbi_extension_calibration.csv', calibration_rows)
    write_csv(destination / 'sbi_extension_comparison_replicates.csv', comparison_rows)
    write_csv(destination / 'sbi_extension_comparison_summary.csv', groups)
    atomic_json(destination / 'sbi_extension_audit.json', dict(families=calibration_audit, comparison_datasets=15,
                                                              prior_variance_seed=20263618,
                                                              parameter_posterior_reference_pending=True))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--received', type=Path, default=ROOT / 'cloud/received/runs')
    ap.add_argument('--destination', type=Path, default=ROOT / 'results')
    ap.add_argument('--classic-root', type=Path, default=ROOT / 'data/runs/revision_v2_main_20261110')
    args = ap.parse_args()
    main(args.received, args.destination, args.classic_root)
