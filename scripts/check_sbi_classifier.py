"""Compare neural and quadrature classifiers on the same saved BayesFlow draws."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from inference import estep, prepared
from legacy_model import ari
from sbi_model import atomic_json, constrain

FAMILIES = {'binary':'baseline', 'poisson':'poisson', 'nb':'nb'}


def chunk(task):
    family, first, last = task
    folder = ROOT/'data/sbi_extension/comparison'/family
    data_file = ROOT/'data/runs/revision_v2_main_20261110/comparison'/f'{FAMILIES[family]}_0_data.npz'
    meta = json.loads((folder/'metadata.json').read_text())
    assert hashlib.sha256(data_file.read_bytes()).hexdigest() == meta['comparison_inputs_sha256'][data_file.name]
    with np.load(data_file) as arrays:
        data = prepared(dict(arrays))
    with np.load(folder/'comparison_predictions.npz') as arrays:
        params = constrain(arrays['theta'][0,first:last])
    prob_sum = np.zeros((300,3))
    exceptions, accuracy = [], []
    for local, index in enumerate(range(first,last)):
        point = {name:value[local] for name,value in params.items()}
        try:
            ll, r, *_ = estep(data, point, order=61)
            if index % 64 == 0:
                ref, r_ref, *_ = estep(data, point, order=101)
                accuracy.append({'draw':index, 'loglik_diff':abs(ll-ref), 'probability_diff':float(abs(r-r_ref).max())})
            assert np.isfinite(r).all() and np.max(abs(r.sum(axis=-1)-1)) < 1e-8
        except (RuntimeError, FloatingPointError, AssertionError) as error:
            exceptions.append({'draw':index, 'message':str(error)})
        else:
            prob_sum += r
    return family, prob_sum, exceptions, accuracy, np.asarray(data['u'])


def main():
    tasks = [(family, start, start+128) for family in FAMILIES for start in range(0,1024,128)]
    groups = {family:[] for family in FAMILIES}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for result in pool.map(chunk, tasks):
            groups[result[0]].append(result)
    audit, probabilities = {}, {}
    for family, group in groups.items():
        failures = [error for item in group for error in item[2]]
        checks = [entry for item in group for entry in item[3]]
        n = 1024-len(failures)
        quadrature = sum(item[1] for item in group) / n
        with np.load(ROOT/'data/sbi_extension/comparison'/family/'comparison_predictions.npz') as arrays:
            neural = arrays['probabilities'][0]
        truth = group[0][4]
        audit[family] = dict(draws=n, calibration_checks=len(checks), failures=failures, rep=0,
                             max_log_likelihood_difference=max(d['loglik_diff'] for d in checks),
                             max_probability_difference=max(d['probability_diff'] for d in checks),
                             rms_probability_difference=float(np.sqrt(np.mean((quadrature-neural)**2))),
                             argmax_agreement=float(np.mean(quadrature.argmax(1)==neural.argmax(1))),
                             ari_neural=ari(truth, neural.argmax(1)), ari_quadrature=ari(truth, quadrature.argmax(1)))
        probabilities[f'{family}_gh'] = quadrature
        probabilities[f'{family}_neural'] = neural
    atomic_json(ROOT/'results/sbi_classifier_same_draws.json', audit)
    np.savez_compressed(ROOT/'data/sbi_reference/sbi_classifier_same_draws.npz', **probabilities)


if __name__ == '__main__':
    main()
