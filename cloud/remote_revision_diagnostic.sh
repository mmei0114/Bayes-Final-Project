#!/usr/bin/env bash
set -euo pipefail
RUN_DIR=${1:?diagnostic directory}
TARGET=/root/bayes-project/runs/revision_v2_main_20261110/robustness
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
cd "$RUN_DIR/work"
finish() { code=$?; printf '%s\n' "$code" > "$RUN_DIR/EXIT_CODE"; if ((code==0)); then state=COMPLETE; else state=FAILED; fi; printf '%s\n' "$state" > "$RUN_DIR/STATE.tmp"; mv "$RUN_DIR/STATE.tmp" "$RUN_DIR/STATE"; }
trap finish EXIT
printf 'RUNNING\n' > "$RUN_DIR/STATE"
mkdir -p "$RUN_DIR/mcmc"
pids=()
for name in baseline poisson nb; do
  test -f "$TARGET/${name}_0_QML.json"
  /root/miniconda3/bin/conda run --no-capture-output -n base python src/conditional_mcmc.py --blocked --results "$TARGET" --scenario "$name" --output "$RUN_DIR/mcmc/$name.json" > "$RUN_DIR/mcmc/$name.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
