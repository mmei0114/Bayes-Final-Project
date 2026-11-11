"""Import completed supplementary trajectories from the local cloud mirror.

Existing scientific records must match byte for byte. Only the progress status
and checksums are regenerated. The original data/runs and cloud backups remain.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, default=ROOT/'cloud/received/runs/convergence_20261111/convergence')
    args = ap.parse_args(); source = args.source.resolve(); target = ROOT/'data/convergence'
    assert source.is_dir(); target.mkdir(parents=True, exist_ok=True)
    assert not list(source.glob('*.error.json')), 'Remote task error files need attention'
    copied = 0
    for src in sorted(source.glob('*.json')):
        if '.checkpoint.' in src.name or src.name == 'STATUS.json':
            continue
        dst = target/src.name
        if dst.exists():
            assert digest(src) == digest(dst), f'Existing record differs: {src.name}'
        else:
            shutil.copy2(src, dst); copied += 1
    expected = [f'{s}_{r}_{m}.json' for s in ('baseline', 'poisson', 'nb')
                for m in ('QML', 'GVA', 'MCEM') for r in (range(5) if m == 'MCEM' else range(1))]
    completed = sum((target/name).exists() for name in expected)
    (target/'STATUS.json').write_text(json.dumps(dict(expected=21, complete=completed,
                                                    status='COMPLETE' if completed == 21 else 'INCOMPLETE'), indent=2)+'\n')
    files = [p for p in target.glob('*.json') if '.checkpoint.' not in p.name]
    (target/'SHA256SUMS').write_text(''.join(digest(p)+'  '+p.name+'\n' for p in sorted(files)))
    print(json.dumps(dict(imported=copied, complete=completed, expected=21)))


if __name__ == '__main__':
    main()
