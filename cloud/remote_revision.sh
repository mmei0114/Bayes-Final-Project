#!/usr/bin/env bash
set -euo pipefail
RUN_DIR=${1:?run directory}; shift
[[ "$RUN_DIR" == /root/bayes-project/runs/* ]] || exit 2
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
CONDA=/root/miniconda3/bin/conda
cd "$RUN_DIR/work"
finish() { code=$?; printf '%s\n' "$code" > "$RUN_DIR/EXIT_CODE"; if ((code==0)); then state=COMPLETE; else state=FAILED; fi; printf '%s\n' "$state" > "$RUN_DIR/STATE.tmp"; mv "$RUN_DIR/STATE.tmp" "$RUN_DIR/STATE"; }
trap finish EXIT
printf 'RUNNING\n' > "$RUN_DIR/STATE"
printf '%q ' "$CONDA" run -n base python revision_v2/src/experiment.py --output "$RUN_DIR/output" "$@" > "$RUN_DIR/COMMAND.txt"
printf '\n' >> "$RUN_DIR/COMMAND.txt"
"$CONDA" run --no-capture-output -n base python revision_v2/src/validate.py > "$RUN_DIR/validation.log" 2>&1
"$CONDA" run --no-capture-output -n base python revision_v2/src/experiment.py --output "$RUN_DIR/output" "$@"
