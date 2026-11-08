#!/usr/bin/env bash
# Stage 2: build the SFT training data (standard + position-balanced, 20K + 500 dev).
#
#   bash scripts/02_build_data.sh --local    # Lambda H100: run here (start it inside tmux)
#   bash scripts/02_build_data.sh            # Slurm: submit one GPU job
#   bash scripts/02_build_data.sh --status   # status + data summary
set -euo pipefail

PROJ="${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJ
source "$PROJ/scripts/config.sh"
source "$PROJ/scripts/lib.sh"
mkdirs

MODE=submit
case "${1:-}" in
  --local)  MODE=local ;;
  --yes)    export ASSUME_YES=1 ;;
  --status) status_summary
            f="$TRAIN_DATA_DIR/final/build_stats.json"
            if [[ -f "$f" ]]; then
              "$PY" -c "import json; s=json.load(open('$f')); [print(f'   {k}: {v}') for k,v in s.items()]"
            fi
            exit 0 ;;
  "") ;;
  *) die "unknown option: $1" ;;
esac

start_log 02_build_data
is_ok setup_env || die "run setup first"
if is_ok build_data; then
  log "build_data already done: $(cat "$STATUS_DIR/build_data.ok")"
  log "to rebuild, delete $STATUS_DIR/build_data.ok (and $TRAIN_DATA_DIR/final to regenerate files)"
  exit 0
fi

if [[ $MODE == local ]]; then
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -dc '0-9')
  log "GPU memory in use right now: ${used} MiB"
  if (( used > 2000 )); then
    log "WARNING: something else is using the GPU (Michael's job?). Check nvidia-smi before continuing."
  fi
elif job_active build_data; then
  log "job $(cat "$STATUS_DIR/build_data.jobid") already queued or running"; exit 0
fi

cat <<EOF
About to run the training-data build ($MODE):
  1. leakage filter + random pool of $TRAIN_POOL InfoSeek train questions   (CPU, minutes)
  2. extract the pool's images from the 48 GB train image tar              (disk, one pass)
  3. PreFLMR retrieval until $TRAIN_N + $DEV_N questions qualify            (GPU, hours)
  4. write standard + balanced_s{${BALANCED_SEEDS// /,}} files, token lengths, hard checks
  output: $TRAIN_DATA_DIR/final/
  resumable: if it stops, run the same command again
EOF
confirm "Start?" || { log "not started"; exit 0; }

if [[ $MODE == local ]]; then
  bash "$PROJ/scripts/slurm/build_data.sub"
  status_summary
else
  submit_job build_data "$PROJ/scripts/slurm/build_data.sub" gpu "$BUILD_TIME"
fi
