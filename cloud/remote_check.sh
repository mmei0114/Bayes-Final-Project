#!/usr/bin/env bash
# Execute only on the computation server. No simulation suite is started.
set -euo pipefail
RUN_DIR=${1:?Pass an existing independent run directory}
[[ "$RUN_DIR" == /root/bayes-project/runs/* ]] || exit 2
cd "$RUN_DIR/work"
CONDA=/root/miniconda3/bin/conda
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg
trap 'printf "FAILED\n" > "$RUN_DIR/CHECK_STATUS"' ERR
"$CONDA" run -n base python -c 'import sys,platform,os,json,numpy,scipy,matplotlib; print(json.dumps(dict(python=sys.version,platform=platform.platform(),cpu_affinity=len(os.sched_getaffinity(0)),numpy=numpy.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__),indent=2))' > "$RUN_DIR/environment.json"
"$CONDA" run -n base python -m pip freeze > "$RUN_DIR/packages_after.txt"
"$CONDA" run --no-capture-output -n base python revision_v2/src/validate.py > "$RUN_DIR/checks.log" 2>&1
printf 'PASSED\n' > "$RUN_DIR/CHECK_STATUS.tmp"
mv "$RUN_DIR/CHECK_STATUS.tmp" "$RUN_DIR/CHECK_STATUS"
cat "$RUN_DIR/environment.json" "$RUN_DIR/CHECK_STATUS"
