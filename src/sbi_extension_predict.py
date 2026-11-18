"""Freeze SBI draws and probabilities for CPU-only comparison and diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import tensorflow as tf

from sbi_model import DESIGN, PRIOR, atomic_json
from sbi_networks import make_models
from sbi_pilot import configure, load_split, response_features


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restore(output, source, family):
    config = json.loads((output / 'configuration.json').read_text())
    assert config.get('family', 'binary') == family
    assert config['design'] == DESIGN and config['prior'] == PRIOR
    for name, digest in config['sources'].items():
        assert file_hash(source / name) == digest, name
    for name in ('sbi_model.py', 'sbi_networks.py'):
        assert file_hash(Path(__file__).with_name(name)) == config['sources'][name], name
    selection = json.loads((output / 'selection.json').read_text())
    posterior, classifier = make_models(config['network_seed'])
    states = {}
    for kind, model in (('npe', posterior), ('classifier', classifier)):
        checkpoint = output / f"checkpoint_{selection['best_' + kind + '_step']:06d}"
        trackable = 'posterior' if kind == 'npe' else kind
        status = tf.train.Checkpoint(**{trackable: model}).read(str(checkpoint / 'state'))
        status.assert_existing_objects_matched().expect_partial()
        states[kind] = {p.name: file_hash(p) for p in sorted(checkpoint.glob('state.*'))}
    standard = json.loads((output / 'standardization.json').read_text())
    return posterior, classifier, np.array(standard['mean'], np.float32), np.array(standard['scale'], np.float32), selection, states


def posterior_draws(models, y, family, draws, rng, dataset_batch):
    posterior, classifier, mean, scale = models
    samples, probabilities, durations = [], [], []
    for first in range(0, len(y), dataset_batch):
        begin = time.monotonic()
        x = response_features(y[first:first + dataset_batch], family)
        n = len(x)
        base = rng.normal(size=(n, draws, 11)).astype(np.float32)
        z = posterior.sample(x, base).numpy()
        samples.append(z * scale + mean)
        mean_prob = np.zeros((n, DESIGN['n'], DESIGN['K']), dtype=np.float64)
        for start in range(0, draws, 32):
            batch = z[:, start:start + 32]
            draw_count = batch.shape[1]
            logits = classifier((np.repeat(x, draw_count, axis=0), batch.reshape(-1, 11)), False)
            conditional = tf.nn.softmax(logits, axis=-1).numpy()
            mean_prob += conditional.reshape(n, draw_count, DESIGN['n'], DESIGN['K']).sum(1, dtype=np.float64)
        probabilities.append(mean_prob / draws)
        durations.append(time.monotonic() - begin)
    samples, probabilities = np.concatenate(samples), np.concatenate(probabilities)
    assert np.isfinite(samples).all() and np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(-1), 1., atol=1e-6)
    return samples, probabilities, durations


def comparison_inputs(root, family):
    scenario = 'baseline' if family == 'binary' else family
    designs, sources = [], {}
    for rep in range(5):
        path = root / f'{scenario}_{rep}_data.npz'
        with np.load(path) as data:
            assert str(data['family']) == family
            assert data['y'].shape == (DESIGN['n'], DESIGN['T'])
            assert np.all(data['mask'] == 1)
            assert np.allclose(data['t'], np.linspace(*DESIGN['times'], DESIGN['T']), rtol=0., atol=1e-12)
            designs.append(data['y'])
        sources[path.name] = file_hash(path)
    return np.stack(designs), sources


def save_npz(path, **arrays):
    tmp = path.with_suffix('.npz.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    tmp.replace(path)


def main(args):
    gpus = configure()
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / 'metadata.json').exists()
    start = time.monotonic()
    posterior, classifier, mean, scale, selection, states = restore(args.model, args.source, args.family)
    models = (posterior, classifier, mean, scale)
    restore_seconds = time.monotonic() - start
    calibration, cal_meta = load_split(args.calibration, 'calibration', ('y',))
    assert cal_meta['spec'].get('family', 'binary') == args.family
    formal, formal_sources = comparison_inputs(args.comparison, args.family)
    # Initialize device and forward paths on a calibration input before timing.
    posterior_draws(models, calibration['y'][:1], args.family, args.draws, np.random.default_rng(71), 1)
    z, p, comparison_seconds = posterior_draws(models, formal, args.family, args.draws, np.random.default_rng(20263218), 1)
    save_npz(args.output / 'comparison_predictions.npz', theta=z, probabilities=p, seconds=comparison_seconds)
    print(f'{args.family}: archived all five comparison predictions', flush=True)
    z, p, seconds = posterior_draws(models, calibration['y'], args.family, args.draws, np.random.default_rng(20263318), 8)
    save_npz(args.output / 'calibration_predictions.npz', theta=z, probabilities=p, batch_seconds=seconds)
    conditional, c_meta = load_split(args.data, 'classifier')
    assert c_meta['spec'].get('family', 'binary') == args.family
    x = response_features(conditional['y'], args.family)
    given_theta = tf.nn.softmax(classifier((x, (conditional['theta'] - mean) / scale), False), axis=-1).numpy()
    assert np.isfinite(given_theta).all() and np.allclose(given_theta.sum(-1), 1., atol=1e-6)
    save_npz(args.output / 'conditional_predictions.npz', probabilities=given_theta)
    atomic_json(args.output / 'metadata.json', dict(
        family=args.family, gpus=gpus, selection=selection, checkpoint_sha256=states, draws=args.draws,
        calibration_datasets=len(calibration['y']), calibration=cal_meta, conditional=c_meta,
        comparison_inputs_sha256=formal_sources, restore_seconds=restore_seconds,
        comparison_timing='Warm per-dataset wall time, data already loaded; feature encoding, draws and classification included',
        comparison_seconds=comparison_seconds,
        wall_seconds=time.monotonic() - start,
    ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('model', 'source', 'calibration', 'data', 'comparison', 'output'):
        parser.add_argument(f'--{arg}', required=True, type=Path)
    parser.add_argument('--family', choices=('binary', 'poisson', 'nb'), required=True)
    parser.add_argument('--draws', type=int, default=1024)
    main(parser.parse_args())
