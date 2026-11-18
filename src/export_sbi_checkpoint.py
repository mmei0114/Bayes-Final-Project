"""Export and verify an inference-only TensorFlow checkpoint on CPU."""
import argparse
import json
import os
from pathlib import Path
import shutil

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import tensorflow as tf

from sbi_extension_predict import posterior_draws, restore
from sbi_model import atomic_json
from sbi_networks import make_models


def main(family, model, source, output):
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    posterior, classifier, mean, scale, selection, state_hashes = restore(model, source, family)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = tf.train.Checkpoint(posterior=posterior, classifier=classifier)
    checkpoint.write(str(output / 'state'))
    config = json.loads((model / 'configuration.json').read_text())
    copy_posterior, copy_classifier = make_models(config['network_seed'])
    fresh_checkpoint = tf.train.Checkpoint(posterior=copy_posterior, classifier=copy_classifier)
    fresh_checkpoint.read(str(output / 'state')).assert_consumed()
    y = np.broadcast_to(np.arange(20, dtype=np.uint8) % 2, (2, 300, 20))
    before = posterior_draws((posterior, classifier, mean, scale), y, family, 32, np.random.default_rng(71), 2)
    after = posterior_draws((copy_posterior, copy_classifier, mean, scale), y, family, 32, np.random.default_rng(71), 2)
    theta_error = float(np.max(np.abs(before[0] - after[0])))
    prob_error = float(np.max(np.abs(before[1] - after[1])))
    assert theta_error < 1e-6 and prob_error < 1e-6
    for name in ('configuration.json', 'standardization.json', 'selection.json'):
        shutil.copy2(model / name, output / name)
    atomic_json(output / 'reload_check.json', dict(family=family, source_checkpoints=state_hashes,
                                                    selection=selection, device='CPU',
                                                    max_parameter_sample_difference=theta_error,
                                                    max_class_probability_difference=prob_error))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('family', choices=('binary', 'poisson', 'nb'))
    parser.add_argument('model', type=Path)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    main(args.family, args.model, args.source, args.output)
