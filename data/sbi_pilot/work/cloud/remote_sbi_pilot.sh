#!/usr/bin/env bash
set -euo pipefail
RUN_DIR=${1:?run directory}
DATA_DIR=${2:?frozen data directory}
[[ "$RUN_DIR" == /root/bayes-project/runs/sbi_pilot_* ]] || exit 2
[[ "$DATA_DIR" == /root/bayes-project/runs/sbi_data_*/datasets ]] || exit 2
CONDA=/root/miniconda3/bin/conda
py() { "$CONDA" run --no-capture-output -n base python "$@"; }
SBI_GPU_LIBS=$(py -c 'import glob; print(":".join(glob.glob("/root/miniconda3/lib/python3.11/site-packages/nvidia/*/lib")))')
export LD_LIBRARY_PATH="$SBI_GPU_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=1 MPLCONFIGDIR="$RUN_DIR/cache/matplotlib" XDG_CACHE_HOME="$RUN_DIR/cache"
MONITOR_PID=''
finish() {
  code=$?
  if [[ -n "$MONITOR_PID" ]]; then kill "$MONITOR_PID" 2>/dev/null || true; wait "$MONITOR_PID" 2>/dev/null || true; fi
  printf '%s\n' "$code" > "$RUN_DIR/EXIT_CODE"
  if ((code==0)); then state=COMPLETE; else state=FAILED; fi
  printf '%s\n' "$state" > "$RUN_DIR/STATE.tmp"
  mv "$RUN_DIR/STATE.tmp" "$RUN_DIR/STATE"
}
trap finish EXIT
cd "$RUN_DIR/work"
printf 'RUNNING\n' > "$RUN_DIR/STATE"
mkdir -p "$RUN_DIR/output"
py cloud/probe_sbi_env.py > "$RUN_DIR/environment.json"
while true; do
  nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,noheader >> "$RUN_DIR/gpu_usage.csv"
  sleep 15
done &
MONITOR_PID=$!
printf 'model_checks\n' > "$RUN_DIR/PHASE"
py src/sbi_checks.py --output "$RUN_DIR/model_checks.json"
printf 'gpu_batch_probe\n' > "$RUN_DIR/PHASE"
py src/sbi_gpu_probe.py --output "$RUN_DIR/batch_probe.json"
BATCH=$(py -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_batch"])' "$RUN_DIR/batch_probe.json")
printf 'toy_and_binary_training\n' > "$RUN_DIR/PHASE"
timeout --signal=TERM 115m "$CONDA" run --no-capture-output -n base python src/sbi_pilot.py \
  --data "$DATA_DIR" --output "$RUN_DIR/output" --steps 10000 --minutes 90 --batch "$BATCH"
printf 'heldout_evaluation\n' > "$RUN_DIR/PHASE"
timeout --signal=TERM 25m "$CONDA" run --no-capture-output -n base python src/sbi_evaluate.py \
  --data "$DATA_DIR" --output "$RUN_DIR/output" --draws 512
printf 'finished\n' > "$RUN_DIR/PHASE"
