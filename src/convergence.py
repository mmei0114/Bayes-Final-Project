"""Supplementary trajectory runs using the original inference kernels.

Replay the originally selected initialization, log every parameter/likelihood,
and, for MCEM, allow at most 30 extra iterations after the recorded run length.
Original fits remain untouched. This is an empirical convergence diagnostic,
not a claim of global optimality or a change to the original results tables.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'

import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
import inference as core
from experiment import atomic

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'data/runs/revision_v2_main_20261110'
PROTOCOL = dict(likelihood_relative_tolerance=1e-6, consecutive_changes=3,
                parameter_scaled_tolerance=1e-3, extra_iterations=30,
                require_an_accepted_update_in_window=True, quadrature_order=61,
                final_quadrature_order=101)
FAMILIES = ('baseline', 'poisson', 'nb')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native(p):
    return np.r_[p['beta'].ravel(), p['var'], p['pi']]


def load_case(scenario, rep, method):
    folder = INPUT / ('comparison' if method == 'MCEM' else 'robustness')
    source = folder / f'{scenario}_{rep}_{method}.json'
    record = json.loads(source.read_text())
    data_path = folder / f'{scenario}_{rep}_data.npz'
    with np.load(data_path) as f:
        data = dict(f)
    data['family'] = str(data['family'])
    return record, data, dict(record_path=str(source.relative_to(ROOT)),
                              record_sha256=sha(source), data_sha256=sha(data_path))


def make_point(d, p, iteration, previous=None, **extra):
    ell = float(core.estep(d, p, PROTOCOL['quadrature_order'])[0])
    point = dict(iteration=iteration, loglik=ell,
                 params={k: v.tolist() for k, v in p.items()}, **extra)
    if previous is not None:
        old = {k: np.array(v) for k, v in previous['params'].items()}
        point['likelihood_relative_change'] = abs(ell-previous['loglik'])/(1+abs(previous['loglik']))
        point['parameter_scaled_change'] = float(np.max(abs(native(p)-native(old))/(1+abs(native(old)))))
    return point


def window_diagnostics(points):
    w = PROTOCOL['consecutive_changes']
    if len(points) <= w:
        return dict(likelihood_stable=False, parameter_stable=False, window_available=False)
    tail = points[-w:]
    accepts = sum(bool(v.get('accepted', True)) for v in tail)
    informative = accepts > 0
    max_l = max(v['likelihood_relative_change'] for v in tail)
    max_p = max(v['parameter_scaled_change'] for v in tail)
    return dict(likelihood_stable=bool(max_l <= PROTOCOL['likelihood_relative_tolerance'] and informative),
                parameter_stable=bool(max_p <= PROTOCOL['parameter_scaled_tolerance']),
                window_available=True, window_accepted_updates=accepts,
                window_max_likelihood_relative_change=float(max_l),
                window_max_parameter_scaled_change=float(max_p))


def replay_comparison(p, ell, original, d):
    target = {k: np.array(v) for k, v in original['fit']['params'].items()}
    delta = float(np.max(abs(native(p)-native(target))))
    delta_ell = float(abs(ell-original['fit']['loglik'])/len(d['y']))
    return dict(maximum_native_parameter_difference=delta,
                loglik_difference_per_subject=delta_ell,
                numerically_matches_saved_fit=bool(delta < 1e-4 and delta_ell < 1e-6))


def run_profile(record, data, save):
    d = core.prepared(data); method = record['method']; K = len(record['fit']['params']['pi'])
    seed = record['seed']+200000+1009*record['fit']['start']
    p = core.initialize(data, K, seed); order = 31 if method == 'GVA' else 61
    points = [make_point(d, p, 0, objective=float(-core.profile(d, core.pack(p), K, method, order)[0]*len(d['y'])))]

    def fun(x):
        return core.profile(d, x, K, method, order)

    def callback(x):
        value = float(-fun(x)[0]*len(d['y']))
        points.append(make_point(d, core.unpack(x, K), len(points), points[-1], objective=value, accepted=True))
        save(points, phase='profile_optimization')

    opt = minimize(fun, core.pack(p), jac=True, bounds=core.bounds(K), method='L-BFGS-B',
                   callback=callback, options=dict(maxiter=300, ftol=1e-12, gtol=2e-6, maxls=30, maxcor=15))
    p = core.unpack(opt.x, K)
    final_ell = float(core.estep(d, p, 101)[0])
    return dict(points=points, reason=str(opt.message), optimizer_success=bool(opt.success),
                original_iteration=record['fit']['iterations'],
                replay_comparison=replay_comparison(p, final_ell, record, d),
                final_loglik_101=final_ell,
                quadrature_difference=abs(final_ell-points[-1]['loglik']),
                **window_diagnostics(points))


def run_mcem(record, data, save):
    d = core.prepared(data); K = len(record['fit']['params']['pi']); n = len(d['y'])
    seed = record['seed']+200000+1009*record['fit']['start']
    p = core.initialize(data, K, seed); rng = np.random.default_rng(seed+73)
    M = 128; stable = 0; legacy_stop = None; at_saved_iteration = None
    original_iterations = record['fit']['iterations']
    points = [make_point(d, p, 0)]
    limit = original_iterations+PROTOCOL['extra_iterations']; reason = 'diagnostic_iteration_budget'
    for it in range(limit):
        _, r, bs, ws, ess = core.estep(d, p, M=M, rng=rng)
        candidate, ok, residual = core.mstep(d, p, r, bs, ws)
        _, rv, bv, wv, ev = core.estep(d, p, M=M, rng=rng)
        delta, se = core.increment(d, p, candidate, rv, bv, wv)
        lo, hi = delta-1.96*se, delta+1.96*se
        step = float(np.max(abs(core.pack(candidate)-core.pack(p))/(1+abs(core.pack(p)))))
        small = hi/n < 5e-5 and se/n < 2e-5 and step < .01 and residual < 1e-4
        accept = lo > 0
        if accept:
            p = candidate
        stable = stable+1 if small else 0
        points.append(make_point(d, p, it+1, points[-1], accepted=bool(accept), M=M,
                                 delta_Q=delta, mcse=se, mstep_success=bool(ok),
                                 mstep_gradient=residual, candidate_scaled_step=step))
        if it+1 == original_iterations:
            ell101 = float(core.estep(d, p, 101)[0])
            at_saved_iteration = dict(iteration=it+1, **replay_comparison(p, ell101, record, d),
                                     **window_diagnostics(points))
        newly_stopped = stable >= 3 and legacy_stop is None
        if newly_stopped:
            legacy_stop = it+1
        save(points, phase='replay' if it+1 <= original_iterations else 'additional_iterations',
             legacy_stop_iteration=legacy_stop, at_original_iteration=at_saved_iteration)
        if it+1 >= original_iterations and window_diagnostics(points)['likelihood_stable']:
            reason = 'likelihood_window_stable'
            break
        # At the original small-Q stop no further sample-size update occurred.
        if newly_stopped:
            continue
        if (not accept and not small) or min(ess, ev)/M < .4:
            if M >= 8192:
                reason = 'mc_budget'
                break
            M *= 2
    final_ell = float(core.estep(d, p, 101)[0])
    return dict(points=points, reason=reason, original_iteration=original_iterations,
                legacy_stop_iteration=legacy_stop, at_original_iteration=at_saved_iteration,
                final_loglik_101=final_ell,
                quadrature_difference=abs(final_ell-points[-1]['loglik']),
                **window_diagnostics(points))


def execute(task):
    scenario, rep, method, output = task
    output = Path(output); tag = f'{scenario}_{rep}_{method}'; started = time.perf_counter()
    record, data, provenance = load_case(scenario, rep, method)
    base = dict(scenario=scenario, rep=rep, method=method, n=len(data['y']), seed=record['seed'],
                selected_original_start=record['fit']['start'], protocol=PROTOCOL, provenance=provenance)

    def save(points, **state):
        atomic(output / f'{tag}.checkpoint.json', dict(**base, points=points, **state))

    try:
        result = (run_mcem if method == 'MCEM' else run_profile)(record, data, save)
        atomic(output / f'{tag}.json', dict(**base, **result, seconds=time.perf_counter()-started))
        return tag, dict(stable=result['likelihood_stable'], reason=result['reason'],
                         iterations=len(result['points'])-1, seconds=round(time.perf_counter()-started, 2))
    except Exception:
        error = traceback.format_exc()
        atomic(output / f'{tag}.error.json', dict(**base, error=error))
        return tag, dict(error=error)


def benchmark(output):
    timings = {}
    for scenario in FAMILIES:
        record, data, _ = load_case(scenario, 0, 'MCEM'); d = core.prepared(data)
        p = {k: np.array(v) for k, v in record['fit']['params'].items()}
        rng = np.random.default_rng(20261111); begin = time.perf_counter(); M = 1024
        _, r, bs, ws, _ = core.estep(d, p, M=M, rng=rng)
        candidate, _, _ = core.mstep(d, p, r, bs, ws)
        _, rv, bv, wv, _ = core.estep(d, p, M=M, rng=rng)
        core.increment(d, p, candidate, rv, bv, wv); core.estep(d, p, 61)
        seconds = time.perf_counter()-begin
        workload = 0
        for rep in range(5):
            rec, _, _ = load_case(scenario, rep, 'MCEM')
            trace = rec['fit']['trace']
            workload += sum(t['M'] for t in trace)+PROTOCOL['extra_iterations']*trace[-1]['M']
        timings[scenario] = dict(seconds_per_1024_draw_iteration=seconds,
                                 rough_serial_seconds=seconds*workload/M)
        print(scenario, timings[scenario], flush=True)
    atomic(output / 'benchmark.json', timings)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, default=ROOT/'data/convergence')
    ap.add_argument('--jobs', type=int, default=3); ap.add_argument('--benchmark', action='store_true')
    ap.add_argument('--scenarios', nargs='+', default=list(FAMILIES)); ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--methods', nargs='+', default=['MCEM', 'QML', 'GVA'])
    ap.add_argument('--phase', choices=('all', 'fast', 'slow'), default='all'); args = ap.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    if args.benchmark:
        benchmark(output); return
    spec = dict(protocol=PROTOCOL, sources={n: sha(ROOT/'src'/n) for n in ('convergence.py', 'inference.py', 'legacy_model.py')},
                scenarios=args.scenarios, reps=args.reps, methods=args.methods)
    metadata = output/'metadata.json'
    if metadata.exists():
        assert json.loads(metadata.read_text())['spec'] == spec, 'Use a new output directory after changing the protocol/source'
    else:
        atomic(metadata, dict(spec=spec, python=sys.version, numpy=np.__version__, platform=platform.platform(), jobs=args.jobs))
    tasks = [(s, r, m, str(output)) for s in args.scenarios for m in args.methods
             for r in (range(args.reps) if m == 'MCEM' else range(1))]
    eligible = [t for t in tasks if args.phase == 'all' or
                ((t[2] != 'MCEM' or t[0] == 'poisson') == (args.phase == 'fast'))]
    todo = [t for t in eligible if not (output/f'{t[0]}_{t[1]}_{t[2]}.json').exists()]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(execute, task) for task in todo]
        for future in as_completed(futures):
            tag, result = future.result(); print(tag, result, flush=True)
    count = sum((output/f'{s}_{r}_{m}.json').exists() for s, r, m, _ in tasks)
    atomic(output/'STATUS.json', dict(expected=len(tasks), complete=count,
                                     status='COMPLETE' if count == len(tasks) else 'INCOMPLETE'))
    if count != len(tasks) and args.phase == 'all':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
