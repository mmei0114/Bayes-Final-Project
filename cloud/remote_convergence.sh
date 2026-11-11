#!/usr/bin/env bash
# Finish the supplementary diagnostic batch from a frozen project snapshot.
set -euo pipefail
RUN_DIR=${1:?independent run directory}
[[ "$RUN_DIR" == /root/bayes-project/runs/* ]] || exit 2
CONDA=/root/miniconda3/bin/conda
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
cd "$RUN_DIR/work"
finish() {
  code=$?
  printf '%s\n' "$code" > "$RUN_DIR/EXIT_CODE"
  if ((code==0)); then state=COMPLETE; else state=FAILED; fi
  printf '%s\n' "$state" > "$RUN_DIR/STATE.tmp"
  mv "$RUN_DIR/STATE.tmp" "$RUN_DIR/STATE"
}
trap finish EXIT
printf 'RUNNING\n' > "$RUN_DIR/STATE"
mkdir -p "$RUN_DIR/convergence"
cp -a data/convergence/. "$RUN_DIR/convergence/"
"$CONDA" run --no-capture-output -n base python - "$RUN_DIR" <<'PY'
import hashlib,json,os,platform,sys
from pathlib import Path
import numpy,scipy,matplotlib
run=Path(sys.argv[1]); out=run/'convergence'
present={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
         for p in out.glob('*.json') if 'method' in json.loads(p.read_text())}
env=dict(environment='base',python=sys.version,platform=platform.platform(),
         numpy=numpy.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,
         cpu_affinity=len(os.sched_getaffinity(0)),jobs=8,
         completed_before_remote_execution=present,
         scheduled_remote_tasks=[f'{s}_{r}_MCEM.json' for s in ('baseline','nb') for r in range(5)])
(out/'server_execution.json').write_text(json.dumps(env,indent=2)+'\n')
PY
"$CONDA" run --no-capture-output -n base python src/convergence.py --jobs 8 --output "$RUN_DIR/convergence"
