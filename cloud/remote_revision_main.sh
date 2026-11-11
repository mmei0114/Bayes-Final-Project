#!/usr/bin/env bash
set -euo pipefail
RUN_DIR=${1:?run directory}
[[ "$RUN_DIR" == /root/bayes-project/runs/* ]] || exit 2
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
CONDA=/root/miniconda3/bin/conda
cd "$RUN_DIR/work"
finish() { code=$?; printf '%s\n' "$code" > "$RUN_DIR/EXIT_CODE"; if ((code==0)); then state=COMPLETE; else state=FAILED; fi; printf '%s\n' "$state" > "$RUN_DIR/STATE.tmp"; mv "$RUN_DIR/STATE.tmp" "$RUN_DIR/STATE"; }
trap finish EXIT
printf 'RUNNING\n' > "$RUN_DIR/STATE"
"$CONDA" run --no-capture-output -n base python src/validate.py > "$RUN_DIR/validation.log" 2>&1
"$CONDA" run --no-capture-output -n base python src/experiment.py --output "$RUN_DIR/robustness" --reps 5 --starts 3 --jobs 8 --maxiter 300 --methods QML GVA
"$CONDA" run --no-capture-output -n base python src/experiment.py --output "$RUN_DIR/comparison" --reps 5 --starts 3 --jobs 8 --maxiter 250 --methods MCEM --scenarios baseline poisson nb
mkdir -p "$RUN_DIR/mcmc"
pids=()
for name in baseline poisson nb; do
  "$CONDA" run --no-capture-output -n base python src/conditional_mcmc.py --results "$RUN_DIR/robustness" --scenario "$name" --output "$RUN_DIR/mcmc/$name.json" > "$RUN_DIR/mcmc/$name.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
