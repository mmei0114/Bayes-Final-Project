#!/usr/bin/env bash
# Local source is authoritative. Never delete remote or local data during sync.
set -euo pipefail
umask 077
CLOUD_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$CLOUD_DIR/.." && pwd)
if [[ ! -f "$CLOUD_DIR/config.env" ]]; then
  echo 'Create cloud/config.env from cloud/config.env.example before using a server.' >&2
  exit 2
fi
source "$CLOUD_DIR/config.env"
mkdir -p "$CLOUD_DIR/private" "$CLOUD_DIR/snapshots" "$CLOUD_DIR/received" "$CLOUD_DIR/logs"
chmod 700 "$CLOUD_DIR/private"
cat > "$CLOUD_DIR/private/ssh_config.tmp.$$" <<EOF
Host bayes-cloud
    HostName $CLOUD_HOST
    Port $CLOUD_PORT
    User $CLOUD_USER
    ConnectTimeout 15
    ServerAliveInterval 15
    ServerAliveCountMax 2
    StrictHostKeyChecking accept-new
    UserKnownHostsFile "$CLOUD_DIR/private/known_hosts"
    ControlMaster auto
    ControlPersist 600
    ControlPath "$CLOUD_DIR/private/control"
EOF
mv "$CLOUD_DIR/private/ssh_config.tmp.$$" "$CLOUD_DIR/private/ssh_config"
SSH=(ssh -F "$CLOUD_DIR/private/ssh_config" bayes-cloud)
printf -v RSH 'ssh -F "%s"' "$CLOUD_DIR/private/ssh_config"
ACTION=${1:-help}
case "$ACTION" in
  push)
    ID=${2:-$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S')}
    [[ "$ID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid snapshot ID' >&2; exit 2; }
    DEST="$CLOUD_DIR/snapshots/$ID"
    [[ ! -e "$DEST" ]] || { echo 'Snapshot already exists; use a new ID' >&2; exit 2; }
    mkdir "$DEST"
    COPYFILE_DISABLE=1 tar --format=ustar --exclude='./.git' --exclude='./.repo-check' --exclude='./cloud/config.env' \
      --exclude='./cloud/private' --exclude='./cloud/snapshots' \
      --exclude='./cloud/received' --exclude='./cloud/logs' \
      --exclude='./build' --exclude='./qa/draft' --exclude='./qa/final' \
      --exclude='__pycache__' -czf "$DEST/project.tar.gz" -C "$PROJECT_DIR" .
    (cd "$DEST" && shasum -a 256 project.tar.gz > SHA256SUMS)
    "${SSH[@]}" "test ! -e '$CLOUD_ROOT/releases/$ID' && mkdir -p '$CLOUD_ROOT/releases/$ID/project' '$CLOUD_ROOT/runs'"
    rsync -a --partial --delay-updates -e "$RSH" "$DEST/" "bayes-cloud:$CLOUD_ROOT/releases/$ID/"
    "${SSH[@]}" "cd '$CLOUD_ROOT/releases/$ID' && sha256sum -c SHA256SUMS && tar -xzf project.tar.gz -C project"
    printf '%s\n' "$ID" > "$CLOUD_DIR/private/latest_snapshot"
    echo "Snapshot $ID is stored locally and uploaded with SHA256 verification."
    ;;
  pull)
    rsync -a --partial --delay-updates --exclude='*.tmp' -e "$RSH" \
      "bayes-cloud:$CLOUD_ROOT/runs/" "$CLOUD_DIR/received/runs/"
    echo "Local run copies: $CLOUD_DIR/received/runs"
    ;;
  verify)
    # Run only when writers have stopped; compares every regular run file.
    bash "$0" pull
    "${SSH[@]}" "cd '$CLOUD_ROOT' && find runs -type f ! -name '*.tmp' -print0 | sort -z | xargs -0 -r sha256sum" \
      > "$CLOUD_DIR/private/SHA256SUMS.incoming"
    mv "$CLOUD_DIR/private/SHA256SUMS.incoming" "$CLOUD_DIR/received/SHA256SUMS"
    (cd "$CLOUD_DIR/received" && shasum -a 256 -c SHA256SUMS) \
      > "$CLOUD_DIR/logs/verification_latest.log" 2>&1
    echo "All remote run files match local copies. See cloud/logs/verification_latest.log."
    ;;
  watch)
    INTERVAL=${2:-60}
    [[ "$INTERVAL" =~ ^[0-9]+$ ]] && (( INTERVAL >= 10 )) || exit 2
    trap 'exit 0' INT TERM
    while true; do
      if ! bash "$0" pull; then echo 'Sync failed; retrying next interval.' >&2; fi
      sleep "$INTERVAL" & wait $!
    done
    ;;
  ssh)
    shift
    exec "${SSH[@]}" "$@"
    ;;
  status)
    "${SSH[@]}" "$CLOUD_CONDA run --no-capture-output -n $CLOUD_ENV python - '$CLOUD_ROOT'" < "$CLOUD_DIR/progress.py"
    ;;
  *)
    echo 'Usage: bash project/cloud/manage.sh {push [ID]|pull|verify|watch [seconds]|status|ssh [command]}'
    ;;
esac
