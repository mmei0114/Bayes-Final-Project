"""Verify and restore completed run archives downloaded from the private MatPool disk."""
import argparse
import hashlib
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
RUNS = ('sbi_extension_data_20261118', 'sbi_extension_train_20261118', 'sbi_extension_predict_20261118')


def restore(archives, received, manifest):
    verified = dict(line.split(maxsplit=1)[::-1] for line in manifest.read_text().splitlines() if line.strip())
    for run in RUNS:
        name = f'{run}.tar'
        archive = archives / name
        with archive.open('rb') as stream:
            observed_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        if observed_hash != verified[name]:
            raise ValueError(f'SHA256 mismatch for {name}')
        with tarfile.open(archive, 'r:') as container:
            for member in container:
                relative = PurePosixPath(member.name)
                if relative.parts[:2] != ('runs', run) or '..' in relative.parts:
                    raise ValueError(f'Unexpected member: {member.name}')
                path = received.joinpath(*relative.parts)
                if not path.resolve().is_relative_to(received.resolve()) or path.is_symlink():
                    raise ValueError(f'Restore path outside archive directory: {member.name}')
                if member.isdir():
                    path.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.restore.tmp', delete=False) as target:
                        temp = Path(target.name)
                        try:
                            with container.extractfile(member) as source:
                                shutil.copyfileobj(source, target, length=1024 * 1024)
                            target.close()
                            temp.replace(path)
                        finally:
                            temp.unlink(missing_ok=True)
                else:
                    raise ValueError(f'Non-file archive member: {member.name}')
        print(f'Restored verified {run}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archives', type=Path)
    parser.add_argument('--received', type=Path, default=ROOT / 'cloud/received')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'qa/sbi_extension/matpool_SHA256SUMS')
    opts = parser.parse_args()
    restore(opts.archives, opts.received, opts.manifest)
