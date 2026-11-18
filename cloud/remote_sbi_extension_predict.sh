#!/usr/bin/env bash
set -euo pipefail
RUN_DIR=${1:?run directory}
DATA=${2:?count data run}
TRAIN=${3:?count training run}
BINARY=${4:?binary pilot run}
BINARY_DATA=${5:?binary pilot data run}
[[ "$RUN_DIR" == /root/bayes-project/runs/sbi_extension_predict_* ]] || exit 2
CONDA=/root/miniconda3/bin/conda
py() { "$CONDA" run --no-capture-output -n base python "$@"; }
libs=$(py -c 'import glob; print(":".join(glob.glob("/root/miniconda3/lib/python3.11/site-packages/nvidia/*/lib")))')
export LD_LIBRARY_PATH="$libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=1 MPLCONFIGDIR="$RUN_DIR/cache/matplotlib"

finish() {
  local code=$?
  printf '%s\n' "$code" > "$RUN_DIR/EXIT_CODE"
  if ((code == 0)); then state=COMPLETE; else state=FAILED; fi
  printf '%s\n' "$state" > "$RUN_DIR/STATE.tmp"
  mv "$RUN_DIR/STATE.tmp" "$RUN_DIR/STATE"
}
trap finish EXIT
printf 'RUNNING\n' > "$RUN_DIR/STATE"
cd "$RUN_DIR/work"
printf 'binary_calibration_data\n' > "$RUN_DIR/PHASE"
PYTHONPATH=src py -c 'import sys; from sbi_dataset import create; create(sys.argv[1], "calibration", 500, 20263518, 1)' "$RUN_DIR/binary_data"
printf 'waiting_for_count_training\n' > "$RUN_DIR/PHASE"
for ((i=0; i<360; i++)); do
  if [[ -f "$TRAIN/EXIT_CODE" ]]; then break; fi
  sleep 15
done
test -f "$TRAIN/EXIT_CODE"

failed=0
for family in binary poisson nb; do
  printf '%s\n' "$family" > "$RUN_DIR/PHASE"
  mkdir -p "$RUN_DIR/$family"
  if [[ "$family" == binary ]]; then
    model="$BINARY/output"; source="$BINARY/work/src"; data="$BINARY_DATA/datasets"; cal="$RUN_DIR/binary_data"
  else
    if [[ ! -f "$TRAIN/$family/EXIT_CODE" || $(cat "$TRAIN/$family/EXIT_CODE") != 0 ]]; then
      printf 'SKIPPED_TRAINING_INCOMPLETE\n' > "$RUN_DIR/$family/STATE"
      failed=1
      continue
    fi
    model="$TRAIN/$family/output"; source="$TRAIN/work/src"; data="$DATA/datasets/$family"; cal="$data"
  fi
  code=0
  timeout --signal=TERM 10m "$CONDA" run --no-capture-output -n base python src/sbi_extension_predict.py \
    --family "$family" --model "$model" --source "$source" --data "$data" --calibration "$cal" \
    --comparison "$RUN_DIR/work/inputs" --output "$RUN_DIR/$family" --draws 1024 > "$RUN_DIR/$family/run.log" 2>&1 || code=$?
  printf '%s\n' "$code" > "$RUN_DIR/$family/EXIT_CODE"
  if ((code != 0)); then failed=1; fi
done
printf 'finished\n' > "$RUN_DIR/PHASE"
exit "$failed"
