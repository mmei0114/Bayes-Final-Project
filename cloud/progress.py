"""Read-only progress summary; execute via the server's selected base Python."""
import json,sys
from pathlib import Path
root=Path(sys.argv[1])/'runs'
for run in sorted(root.iterdir()):
    if not run.is_dir():continue
    state=run/'STATE'
    if state.exists():print(run.name,state.read_text().strip())
    for name in ('robustness','comparison','output','convergence'):
        folder=run/name
        if not folder.exists():continue
        complete=[]
        for p in folder.glob('*.json'):
            if any(s in p.name for s in ('.checkpoint.','.error.')):continue
            try:
                value=json.loads(p.read_text())
                if 'fit' in value or ('points' in value and 'method' in value):complete.append(p)
            except (OSError,ValueError):pass
        errors=list(folder.glob('*.error.json'))
        print(' ',name,'complete=',len(complete),'exceptions=',len(errors))
        for path in sorted(folder.glob('*.checkpoint.json')):
            tag=path.name.removesuffix('.checkpoint.json')
            if (folder/(tag+'.json')).exists():continue
            try:
                payload=json.loads(path.read_text());v=payload.get('state',payload);tr=v.get('trace',v.get('points',[]))
                print('  ',tag,'finished starts=',len(v.get('completed_starts',[])),
                      'active=',v.get('active_start'), 'last=',
                      {k:tr[-1].get(k) for k in ('iteration','M','delta_Q','mcse','accepted','loglik')} if tr else None)
            except (OSError,ValueError):print('  ',tag,'checkpoint changed during read')
