"""Run the fixed-design, same-prior Stan comparison on saved simulated data."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from sbi_model import atomic_json, prior_sample

FAMILIES = {'binary': ('baseline', 1), 'poisson': ('poisson', 2), 'nb': ('nb', 3)}
RUN_DATA = ROOT/'data/runs/revision_v2_main_20261110'


def native_initialization(case, rep, chain, seed):
    rng = np.random.default_rng(seed)
    if chain <= 2:
        qml = json.loads((RUN_DATA/'robustness'/f'{case}_{rep}_QML.json').read_text())['fit']['params']
        beta = np.asarray(qml['beta']) + rng.normal(0, .15, (3, 2))
        variance = np.clip(np.asarray(qml['var']) * np.exp(rng.normal(0, .25, 3)), .01, 4.9)
        pi = (np.asarray(qml['pi']) + .04) / 1.12
        order = np.argsort(beta[:, 0])
        beta, variance, pi = beta[order], variance[order], pi[order]
    else:
        prior = prior_sample(rng, 1)
        beta, variance, pi = (prior[name][0] for name in ('beta', 'var', 'pi'))
    return {'intercept': beta[:, 0].tolist(), 'slope': beta[:, 1].tolist(),
            'log_variance': np.log(variance).tolist(), 'pi': pi.tolist(),
            'z': rng.normal(0, .05, 300).tolist()}


def stan_data(path, family):
    with np.load(path) as data:
        y, t, mask = (data[name] for name in ('y', 't', 'mask'))
        if y.shape != (300, 20) or not np.all(mask == 1) or not np.all(t == t[0]):
            raise ValueError(f'Expected the balanced fixed design in {path.name}')
        if not np.all(y == np.round(y)) or not np.all(y >= 0):
            raise ValueError(f'Expected discrete nonnegative responses in {path.name}')
        return dict(N=300, T=20, K=3, family=family, t=t[0].tolist(), y=y.astype(int).tolist())


def run_chain(args, family, chain):
    case, code = FAMILIES[family]
    out = args.output/family
    seed = args.seed + 100*code + chain
    init = out/f'init_{chain}.json'
    atomic_json(init, native_initialization(case, args.rep, chain, seed))
    output = out/f'chain_{chain}.csv'
    command = [str(args.model), 'sample', f'num_samples={args.draws}', f'num_warmup={args.warmup}',
               'adapt', f'delta={args.delta}', 'algorithm=hmc', 'engine=nuts', 'max_depth=12',
               f'id={chain}', f'init={init}', 'data', f'file={out / "data.json"}', 'random', f'seed={seed}',
               'output', f'file={output}', 'refresh=100', 'sig_figs=16']
    start = time.monotonic()
    with (out/f'chain_{chain}.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    duration = time.monotonic()-start
    atomic_json(out/f'status_{chain}.json', {'elapsed_seconds': duration, 'seed': seed,
                                         'initializer': 'perturbed QML' if chain <= 2 else 'prior draw'})
    print(f'{family} chain {chain} complete ({duration:.1f}s)', flush=True)


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = dict(rep=args.rep, warmup=args.warmup, draws_per_chain=args.draws, adapt_delta=args.delta,
                    max_depth=12, chains=4, jobs=args.jobs, base_seed=args.seed,
                    model_sha256=hashlib.sha256((ROOT/'src/sbi_reference.stan').read_bytes()).hexdigest(),
                    platform=platform.platform(), processor=platform.machine(), cmdstan_version='2.38.0', cases={})
    for family in args.families:
        case, code = FAMILIES[family]
        data = RUN_DATA/'comparison'/f'{case}_{args.rep}_data.npz'
        out = args.output/family
        out.mkdir()
        atomic_json(out/'data.json', stan_data(data, code))
        metadata['cases'][family] = {'name': data.name, 'sha256': hashlib.sha256(data.read_bytes()).hexdigest()}
    atomic_json(args.output/'configuration.json', metadata)
    tasks = [(f, c) for f in args.families for c in range(1, 5)]
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        for result in executor.map(lambda x: run_chain(args, *x), tasks):
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=lambda x: Path(x).resolve())
    parser.add_argument('--output', required=True, type=lambda x: Path(x).resolve())
    parser.add_argument('--families', nargs='+', choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument('--rep', type=int, default=0, choices=range(5))
    parser.add_argument('--seed', type=int, default=20262120)
    parser.add_argument('--warmup', type=int, default=1000)
    parser.add_argument('--draws', type=int, default=1000)
    parser.add_argument('--delta', type=float, default=.95)
    parser.add_argument('--jobs', type=int, default=8)
    main(parser.parse_args())
