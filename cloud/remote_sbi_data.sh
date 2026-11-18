#!/usr/bin/env bash
set -euo pipefail
RUN_DIR=${1:?run directory}
[[ "$RUN_DIR" == /root/bayes-project/runs/sbi_data_* ]] || exit 2
finish() {
  code=$?
  printf '%s\n' "$code" > "$RUN_DIR/EXIT_CODE"
  if ((code==0)); then state=COMPLETE; else state=FAILED; fi
  printf '%s\n' "$state" > "$RUN_DIR/STATE.tmp"
  mv "$RUN_DIR/STATE.tmp" "$RUN_DIR/STATE"
}
trap finish EXIT
printf 'RUNNING\n' > "$RUN_DIR/STATE"
cd "$RUN_DIR/work"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/root/miniconda3/bin/conda run --no-capture-output -n base python src/sbi_dataset.py \
  --output "$RUN_DIR/datasets" --size 100000 --jobs 8
