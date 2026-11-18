#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/bayes-project
OUT=${1:?summary run directory}
[[ "$OUT" == "$ROOT/runs/sbi_extension_summary_"* ]] || exit 2
PRED="$ROOT/runs/sbi_extension_predict_20261118"
CLASSIC="$ROOT/runs/sbi_pilot_20261118/work"
STORAGE=/mnt/bayes-sbi-20261118

finish() {
  local code=$?
  printf '%s\n' "$code" > "$OUT/EXIT_CODE"
  if ((code == 0)); then state=COMPLETE; else state=FAILED; fi
  printf '%s\n' "$state" > "$OUT/STATE.tmp"
  mv "$OUT/STATE.tmp" "$OUT/STATE"
}
trap finish EXIT
printf 'RUNNING\n' > "$OUT/STATE"
for ((i=0; i<400; i++)); do
  if [[ -f "$PRED/EXIT_CODE" ]]; then break; fi
  sleep 15
done
[[ $(cat "$PRED/EXIT_CODE") == 0 ]]

export CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONPATH="$CLASSIC/src" /root/miniconda3/bin/conda run --no-capture-output -n base python \
  "$OUT/work/summarize_sbi_extension.py" --received "$ROOT/runs" --destination "$OUT/results" \
  --classic-root "$CLASSIC/data/runs/revision_v2_main_20261110"

mountpoint -q /mnt
archive="$OUT/results.tar"
tar -cf "$archive" -C "$OUT" results work
digest=$(sha256sum "$archive" | cut -d ' ' -f 1)
cp "$archive" "$STORAGE/sbi_extension_summary_20261118.tar.partial"
[[ $(sha256sum "$STORAGE/sbi_extension_summary_20261118.tar.partial" | cut -d ' ' -f 1) == "$digest" ]]
mv "$STORAGE/sbi_extension_summary_20261118.tar.partial" "$STORAGE/sbi_extension_summary_20261118.tar"
printf '%s  %s\n' "$digest" sbi_extension_summary_20261118.tar > "$STORAGE/sbi_extension_summary_20261118.tar.sha256"
