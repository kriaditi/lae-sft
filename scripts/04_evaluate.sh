#!/usr/bin/env bash
# Stage 4: evaluate every finished LoRA run with the authors' P17 protocol, then
# build the comparison tables and plots.
#
#   bash scripts/04_evaluate.sh --local      # Lambda: evaluate all finished, not-yet-evaluated runs
#   bash scripts/04_evaluate.sh --compare    # only rebuild tables/plots from finished evaluations
#   bash scripts/04_evaluate.sh --status
#
# The frozen baseline is the reproduction run (runs/p17_repro); it is not rerun.
# Evaluation can share the GPU with a training run (about 20 GB + 44 GB of 80 GB).
set -euo pipefail

PROJ="${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJ
source "$PROJ/scripts/config.sh"
source "$PROJ/scripts/lib.sh"
mkdirs

MODE=submit; ONLY_COMPARE=0
for a in "$@"; do
  case "$a" in
    --local)   MODE=local ;;
    --compare) ONLY_COMPARE=1 ;;
    --yes)     export ASSUME_YES=1 ;;
    --status)  status_summary; exit 0 ;;
    *) die "unknown option: $a" ;;
  esac
done

start_log 04_evaluate
is_ok reproduce_p17 || die "the frozen reproduction must exist first (scripts/01_reproduce.sh)"
python3 "$PROJ/lae/patch_authors.py" --authors-dir "$AUTHORS_DIR"   # adds the LoRA hook (idempotent)

compare() {
  log "building comparison tables and plots"
  "$PY" "$PROJ/lae/compare.py" \
    --paper "$AUTHORS_DIR/results/p17/per_cell_summary.json" \
    --frozen "$RUNS_DIR/p17_repro" \
    --eval-root "$RUNS_DIR/eval" --train-root "$RUNS_DIR/train" \
    --out "$RESULTS_DIR"
}
if [[ $ONLY_COMPARE == 1 ]]; then compare; exit 0; fi

# runs that finished training and are not yet evaluated
todo=()
for ok in "$STATUS_DIR"/train_*.ok; do
  [[ -e "$ok" ]] || continue
  name="$(basename "$ok" .ok)"; name="${name#train_}"
  [[ "$name" == smoke_* ]] && continue
  is_ok "eval_$name" || todo+=("$name")
done
if (( ${#todo[@]} == 0 )); then
  log "nothing new to evaluate"; compare; exit 0
fi

if [[ $MODE == local ]]; then
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -dc '0-9')
  log "free GPU memory: ${free} MiB (evaluation needs about 20,000)"
  (( free > 22000 )) || die "not enough free GPU memory right now"
fi
cat <<EOF
About to evaluate ($MODE), one after another, with the authors' P17 protocol (13 cells x 496 questions):
$(for n in "${todo[@]}"; do echo "   $n   adapter: $RUNS_DIR/train/$n/adapter"; done)
  frozen reader from our reproduction ran 51 min on this GPU; expect about the same per model
EOF
confirm "Start?" || { log "not started"; exit 0; }

failed=0
for n in "${todo[@]}"; do
  if [[ $MODE == local ]]; then
    EVAL_NAME="$n" ADAPTER="$RUNS_DIR/train/$n/adapter" bash "$PROJ/scripts/slurm/eval_p17.sub" \
      || { log "EVAL FAILED: $n"; failed=1; }
  else
    job_active "eval_$n" && { log "$n already queued"; continue; }
    EVAL_NAME="$n" ADAPTER="$RUNS_DIR/train/$n/adapter" submit_job "eval_$n" "$PROJ/scripts/slurm/eval_p17.sub" gpu "$REPRO_TIME"
  fi
done
if [[ $MODE == local ]]; then compare; fi
status_summary
exit $failed
