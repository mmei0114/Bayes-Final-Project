#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/bayes-project
STORAGE=${1:?user network disk directory}
[[ "$STORAGE" == /mnt/bayes-sbi-* ]] || exit 2
mountpoint -q /mnt
mkdir -p "$STORAGE" "$ROOT/extension_backup"

archive() {
  local run=$1 name="$1.tar" source digest
  # All runs write their final STATE after their result and exit code files.
  for ((i=0; i<420; i++)); do
    if [[ -f "$ROOT/runs/$run/STATE" ]] && [[ $(cat "$ROOT/runs/$run/STATE") =~ ^(COMPLETE|FAILED)$ ]]; then break; fi
    sleep 15
  done
  [[ $(cat "$ROOT/runs/$run/STATE") =~ ^(COMPLETE|FAILED)$ ]]
  source="$ROOT/extension_backup/$name"
  tar --exclude='*.tmp' -cf "$source" -C "$ROOT" "runs/$run"
  digest=$(sha256sum "$source" | cut -d ' ' -f 1)
  cp "$source" "$STORAGE/$name.partial"
  [[ $(sha256sum "$STORAGE/$name.partial" | cut -d ' ' -f 1) == "$digest" ]]
  mv "$STORAGE/$name.partial" "$STORAGE/$name"
  printf '%s  %s\n' "$digest" "$name" > "$STORAGE/$name.sha256"
  printf '%s verified %s\n' "$(date -Is)" "$name"
}

archive sbi_extension_data_20261118
archive sbi_extension_train_20261118
archive sbi_extension_predict_20261118
