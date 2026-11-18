"""Plot three-family marginal calibration from the saved summary CSV."""
import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'build/matplotlib'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    with (ROOT / 'results/sbi_extension_calibration.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    families = [('binary', 'Binary'), ('poisson', 'Poisson'), ('nb', 'Negative binomial')]
    labels = [rf'$\beta_{{{k},{j}}}$' for k in range(1, 4) for j in (0, 1)]
    labels += [rf'$D_{k}$' for k in range(1, 4)] + [rf'$\pi_{k}$' for k in range(1, 4)]
    position = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.3), sharey=True, layout='constrained')
    for axis, (name, title) in zip(axes, families):
        family_rows = [row for row in rows if row['family'] == name]
        assert len(family_rows) == len(labels)
        for target, color, marker in [(50, '#326b95', 'o'), (95, '#ba6f36', 's')]:
            coverage = [float(row[f'coverage_{target}']) for row in family_rows]
            axis.axhline(target / 100, color=color, ls=':', lw=1.)
            axis.plot(position, coverage, marker=marker, color=color, ms=4, label=f'{target}% intervals')
        axis.set(title=title, ylim=(.25, 1.01), xticks=position, xticklabels=labels)
        axis.tick_params(axis='x', labelrotation=45, labelsize=9)
        axis.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Empirical coverage')
    axes[-1].legend(loc='lower right', fontsize=8)
    fig.suptitle('SBI calibration across response families (500 datasets per family)')
    fig.savefig(ROOT / 'figures/sbi_extension_coverage.pdf', metadata={'CreationDate': None})
    plt.close(fig)


if __name__ == '__main__':
    main()
