"""Read-only progress summary; execute via the server's selected base Python."""
import json,sys
from pathlib import Path
root=Path(sys.argv[1])/'runs'
for run in sorted(root.iterdir()):
    if not run.is_dir():continue
    state=run/'STATE'
    if state.exists():print(run.name,state.read_text().strip())
    for name in ('robustness','comparison','output'):
        folder=run/name
        if not folder.exists():continue
        complete=[p for p in folder.glob('*.json') if not any(s in p.name for s in ('.checkpoint.','.error.')) and p.name not in ('metadata.json','STATUS.json')]
        errors=list(folder.glob('*.error.json'))
        print(' ',name,'complete=',len(complete),'exceptions=',len(errors))
        for path in sorted(folder.glob('*.checkpoint.json')):
            tag=path.name.removesuffix('.checkpoint.json')
            if (folder/(tag+'.json')).exists():continue
            try:
                v=json.loads(path.read_text())['state'];tr=v.get('trace',[])
                print('  ',tag,'finished starts=',len(v.get('completed_starts',[])),
                      'active=',v.get('active_start'), 'last=',
                      {k:tr[-1].get(k) for k in ('iteration','M','delta_Q','mcse','accepted')} if tr else None)
            except (OSError,ValueError):print('  ',tag,'checkpoint changed during read')
