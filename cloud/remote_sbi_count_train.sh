#!/usr/bin/env bash
set -euo pipefail
RUN_DIR=${1:?run directory}
DATA_DIR=${2:?dataset directory}
TOY=${3:?successful toy check}
[[ "$RUN_DIR" == /root/bayes-project/runs/sbi_extension_train_* ]] || exit 2
CONDA=/root/miniconda3/bin/conda
py() { "$CONDA" run --no-capture-output -n base python "$@"; }
libs=$(py -c 'import glob; print(":".join(glob.glob("/root/miniconda3/lib/python3.11/site-packages/nvidia/*/lib")))')
export LD_LIBRARY_PATH="$libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=1 MPLCONFIGDIR="$RUN_DIR/cache/matplotlib"
monitor=''

finish() {
  local code=$?
  if [[ -n "$monitor" ]]; then kill "$monitor" 2>/dev/null || true; wait "$monitor" 2>/dev/null || true; fi
  printf '%s\n' "$code" > "$RUN_DIR/EXIT_CODE"
  if ((code == 0)); then state=COMPLETE; else state=FAILED; fi
  printf '%s\n' "$state" > "$RUN_DIR/STATE.tmp"
  mv "$RUN_DIR/STATE.tmp" "$RUN_DIR/STATE"
}
trap finish EXIT
cd "$RUN_DIR/work"
printf 'RUNNING\n' > "$RUN_DIR/STATE"
py cloud/probe_sbi_env.py > "$RUN_DIR/environment.json"
while true; do
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv,noheader >> "$RUN_DIR/gpu_usage.csv"
  sleep 15
done &
monitor=$!

failed=0
for family in poisson nb; do
  printf '%s\n' "$family" > "$RUN_DIR/PHASE"
  mkdir -p "$RUN_DIR/$family"
  code=0
  timeout --signal=TERM 45m "$CONDA" run --no-capture-output -n base python src/sbi_pilot.py \
    --family "$family" --data "$DATA_DIR/$family" --output "$RUN_DIR/$family/output" \
    --validated-toy "$TOY" --steps 10000 --minutes 39 --batch 1024 > "$RUN_DIR/$family/run.log" 2>&1 || code=$?
  printf '%s\n' "$code" > "$RUN_DIR/$family/EXIT_CODE"
  if ((code != 0)); then failed=1; fi
done
printf 'finished\n' > "$RUN_DIR/PHASE"
exit "$failed"
