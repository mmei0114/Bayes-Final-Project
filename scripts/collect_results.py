"""Copy the report's completed inputs out of local cloud backups, without overwriting.

No experiments or network operations are performed. Run from the repository root:
    conda run -n yellow python scripts/collect_results.py
The curated data/ tree is committed so a fresh clone can summarize without a server.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = 'revision_v2_main_20261110'
DIAGNOSTIC = 'revision_v2_diagnostic_20261110'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(source, destination):
    entries = []

    def copy_file(src, relative):
        assert src.is_file(), src
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        sha = digest(src)
        if target.exists():
            if digest(target) != sha:
                raise RuntimeError(f'Existing destination differs: {relative}')
        else:
            shutil.copy2(src, target)
        assert digest(target) == sha
        entries.append(dict(path=relative.as_posix(), sha256=sha, bytes=src.stat().st_size))

    for name in (MAIN, DIAGNOSTIC):
        run = source / name
        assert (run / 'STATE').read_text().strip() == 'COMPLETE', name
        assert (run / 'EXIT_CODE').read_text().strip() == '0', name
        for filename in ('STATE', 'EXIT_CODE', 'source_snapshot.txt', 'source_archive.sha256'):
            copy_file(run / filename, Path('runs') / name / filename)
        for filename in ('run.log', 'validation.log'):
            if (run / filename).is_file():
                copy_file(run / filename, Path('runs') / name / filename)
        # These are immutable execution copies, not the whole duplicated work tree.
        for src in sorted((run / 'work/revision_v2/src').glob('*.py')):
            copy_file(src, Path('runs') / name / src.relative_to(run))
        validation = run / 'work/revision_v2/qa/validation.json'
        if validation.exists():
            copy_file(validation, Path('runs') / name / validation.relative_to(run))

    run = source / MAIN
    for folder, expected in (('robustness', 140), ('comparison', 15)):
        status = json.loads((run / folder / 'STATUS.json').read_text())
        assert status['status'] == 'COMPLETE' and status['complete'] == expected
        assert not list((run / folder).glob('*.error.json'))
        fits = 0
        for src in sorted((run / folder).iterdir()):
            if not src.is_file() or '.checkpoint.' in src.name:
                continue
            if src.suffix not in ('.json', '.npz'):
                continue
            if src.suffix == '.json' and 'fit' in json.loads(src.read_text()):
                fits += 1
            copy_file(src, Path('runs') / MAIN / folder / src.name)
        assert fits == expected, (folder, fits)
    for filename in ('poisson.json', 'poisson.chains.npz'):
        copy_file(run / 'probe_mcmc' / filename, Path('runs') / MAIN / 'probe_mcmc' / filename)

    run = source / DIAGNOSTIC
    for scenario in ('baseline', 'poisson', 'nb'):
        for suffix in ('.json', '.chains.npz', '.log'):
            filename = scenario + suffix
            copy_file(run / 'mcmc' / filename, Path('runs') / DIAGNOSTIC / 'mcmc' / filename)
    copy_file(source / 'setup_20261110/environment.json', Path('environment.json'))
    entries.sort(key=lambda e: e['path'])
    (destination / 'inventory.json').write_text(json.dumps(entries, indent=2) + '\n')
    (destination / 'SHA256SUMS').write_text(''.join(f"{e['sha256']}  {e['path']}\n" for e in entries))
    print(json.dumps(dict(files=len(entries), bytes=sum(e['bytes'] for e in entries), fits=155,
                         destination=str(destination)), indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root', type=Path, default=ROOT / 'cloud/received/runs')
    ap.add_argument('--output', type=Path, default=ROOT / 'data')
    args = ap.parse_args()
    collect(args.source_root.resolve(), args.output.resolve())
