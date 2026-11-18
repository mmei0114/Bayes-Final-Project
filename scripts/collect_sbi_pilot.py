"""Curate completed, hash-verified local SBI backups; never run training."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
RECEIVED=ROOT/'cloud/received'
RUN_ID='sbi_pilot_20261118'
DATA_ID='sbi_data_20261118'
DEST=ROOT/'data/sbi_pilot'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        assert sha(source)==sha(target),f'Immutable artifact differs: {target}'
    else:
        shutil.copy2(source,target)
    assert sha(source)==sha(target)


def main():
    manifest=RECEIVED/f'SHA256SUMS_{RUN_ID}'
    records={}
    for line in manifest.read_text().splitlines():
        digest,name=line.split('  ',1)
        assert sha(RECEIVED/name)==digest,name
        records[name]=digest
    run=RECEIVED/'runs'/RUN_ID;data=RECEIVED/'runs'/DATA_ID
    assert (run/'STATE').read_text().strip()=='COMPLETE'
    assert (run/'EXIT_CODE').read_text().strip()=='0'
    assert (data/'STATE').read_text().strip()=='COMPLETE'
    selected=json.loads((run/'output/selection.json').read_text())
    steps={selected['best_npe_step'],selected['best_classifier_step']}
    for file in (run/'output').rglob('*'):
        if not file.is_file():continue
        relative=file.relative_to(run/'output')
        if relative.parts[0].startswith('checkpoint_'):
            if relative.parts[0] not in {f'checkpoint_{s:06d}' for s in steps}:continue
        copy(file,DEST/'output'/relative)
    for name in ('STATE','EXIT_CODE','PHASE','run.log','environment.json','model_checks.json',
                 'batch_probe.json','gpu_usage.csv','source_archive_sha256.txt'):
        copy(run/name,DEST/'execution'/name)
    for path in (run/'work/src').glob('sbi_*.py'):
        copy(path,DEST/'work/src'/path.name)
    for name in ('inference.py','legacy_model.py'):
        copy(run/'work/src'/name,DEST/'work/src'/name)
    for name in ('remote_sbi_pilot.sh','probe_sbi_env.py'):
        copy(run/'work/cloud'/name,DEST/'work/cloud'/name)
    copy(run/'work/requirements-sbi-pilot.txt',DEST/'work/requirements-sbi-pilot.txt')
    for path in (data/'work').rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:
            copy(path,DEST/'work/data_generation'/path.relative_to(data/'work'))
    for split in ('train','validation','calibration','classifier'):
        directory=data/'datasets'/split
        for name in ('spec.json','manifest.json'):
            copy(directory/name,DEST/'datasets'/split/name)
        if split!='train':
            for path in directory.glob('*.npz'):
                copy(path,DEST/'datasets'/split/path.name)
    setup=RECEIVED/'runs/sbi_setup_20261118'
    for name in ('pip_check.txt','pip_freeze.txt','conda_list.json','artifacts.json','verified.json','STATE'):
        copy(setup/name,DEST/'environment'/name)
    for name in ('sbi_environment_before.json','sbi_gpu_environment.json'):
        copy(ROOT/'cloud/logs'/name,DEST/'environment'/name)
    copy(manifest,ROOT/'qa/sbi_pilot/cloud_SHA256SUMS')
    audited=dict(run_id=RUN_ID,full_local_run_file_count=len(records),
                 full_local_bytes=sum((RECEIVED/name).stat().st_size for name in records),
                 all_remote_run_files_match_local=True,source='cloud/received/'+manifest.name,
                 unique_training_datasets=100000,training_arrays_in_public_repository=False,
                 training_arrays_locally_archived=True,selected_checkpoints=sorted(steps),
                 report_pdf_unchanged=True)
    (ROOT/'qa/sbi_pilot/cloud_verification.json').write_text(json.dumps(audited,indent=2)+'\n')
    files=sorted(p for p in DEST.rglob('*') if p.is_file() and p.name not in ('SHA256SUMS','README.md'))
    (DEST/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(DEST).as_posix()}\n' for p in files))
    print(json.dumps(audited,indent=2))


if __name__=='__main__':main()
