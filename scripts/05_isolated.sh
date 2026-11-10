#!/usr/bin/env bash
# Stage 5: isolated-reading baseline (frozen reader, no training), then rebuild the tables.
#
#   bash scripts/05_isolated.sh --local    # Lambda (start in tmux); ~30-40 min
#   bash scripts/05_isolated.sh            # Slurm: one GPU job
set -euo pipefail

PROJ="${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJ
source "$PROJ/scripts/config.sh"
source "$PROJ/scripts/lib.sh"
mkdirs

MODE=submit
case "${1:-}" in
  --local) MODE=local ;;
  --yes)   export ASSUME_YES=1 ;;
  "") ;;
  *) die "unknown option: $1" ;;
esac

start_log 05_isolated
is_ok reproduce_p17 || die "the frozen reproduction must exist first"
if is_ok isolated; then
  log "isolated reading already done: $(cat "$STATUS_DIR/isolated.ok")"
  bash "$PROJ/scripts/04_evaluate.sh" --compare
  exit 0
fi
if [[ $MODE == local ]]; then
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -dc '0-9')
  log "free GPU memory: ${free} MiB (needs about 20,000)"
  (( free > 22000 )) || die "not enough free GPU memory right now"
fi
cat <<EOF
About to run isolated reading ($MODE): frozen Qwen2.5-VL-7B reads each of the same 10 passages
alone for each of the 496 questions (4,960 short generations), picks the most confident answer
(primary: mean log-prob per token; secondary: total log-prob), scores it with the official scorer.
  output: $RUNS_DIR/isolated/   then the comparison tables in $RESULTS_DIR are rebuilt
EOF
confirm "Start?" || { log "not started"; exit 0; }

if [[ $MODE == local ]]; then
  bash "$PROJ/scripts/slurm/isolated.sub"
  bash "$PROJ/scripts/04_evaluate.sh" --compare
else
  submit_job isolated "$PROJ/scripts/slurm/isolated.sub" gpu "$REPRO_TIME"
fi
