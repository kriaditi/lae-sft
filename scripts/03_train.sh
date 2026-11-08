#!/usr/bin/env bash
# Stage 3: LoRA SFT runs.
#
#   bash scripts/03_train.sh --local --smoke   # Lambda: 10-step test of both conditions (~minutes)
#   bash scripts/03_train.sh --local           # Lambda: all runs one after another (start in tmux)
#   bash scripts/03_train.sh                   # Slurm: one GPU job per run
#   bash scripts/03_train.sh --status          # status + per-run summary
#
# Run order puts one seed of each condition first, so a (labelled preliminary)
# single-seed comparison exists before the second seeds finish.
set -euo pipefail

PROJ="${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJ
source "$PROJ/scripts/config.sh"
source "$PROJ/scripts/lib.sh"
mkdirs

MODE=submit; SMOKE=0
for a in "$@"; do
  case "$a" in
    --local) MODE=local ;;
    --smoke) SMOKE=1 ;;
    --yes)   export ASSUME_YES=1 ;;
    --status)
      status_summary
      for m in "$RUNS_DIR"/train/*/train_metrics.json; do
        [[ -e "$m" ]] || continue
        "$PY" -c "import json,sys; m=json.load(open('$m')); n='$m'.split('/')[-2]; \
print(f'   {n:28s} steps {m[\"global_steps\"]:5d}  dev loss {m[\"dev_loss\"]:.4f}  s/step {m[\"s_per_step\"]}  peak {m[\"peak_gpu_mem_gb\"]} GB  wall {m[\"train_wall_s_this_session\"]/3600:.2f} h')"
      done
      exit 0 ;;
    *) die "unknown option: $a" ;;
  esac
done
export SMOKE

start_log 03_train
is_ok build_data || die "training data not built yet: bash scripts/02_build_data.sh --local"

# peft + accelerate into the authors' env (pinned; transformers stays at 4.49.0)
if ! "$PY" -c "import peft, accelerate" 2>/dev/null; then
  log "installing peft==$PEFT_VERSION accelerate==$ACCELERATE_VERSION into the venv"
  export PATH="$HOME/.local/bin:$PATH"
  uv pip install --python "$PY" "peft==$PEFT_VERSION" "accelerate==$ACCELERATE_VERSION"
fi
"$PY" -c "import transformers, peft, accelerate, torch; \
assert transformers.__version__ == '4.49.0', transformers.__version__; \
print(f'   torch {torch.__version__}  transformers {transformers.__version__}  peft {peft.__version__}  accelerate {accelerate.__version__}')"

RUNS=()
for s in $BALANCED_SEEDS; do RUNS+=("standard:$s" "balanced_s$s:$s"); done
if [[ $SMOKE == 1 ]]; then RUNS=("standard:0" "balanced_s0:0"); fi

if [[ $MODE == local ]]; then
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -dc '0-9')
  log "GPU memory in use right now: ${used} MiB"
  (( used < 2000 )) || log "WARNING: the GPU is already in use (Michael's job?). Check nvidia-smi first."
fi

cat <<EOF
About to run $([[ $SMOKE == 1 ]] && echo "the SMOKE TEST (10 steps, 160 examples)" || echo "FULL training") ($MODE):
$(for r in "${RUNS[@]}"; do echo "   ${r%%:*}  seed ${r##*:}"; done)
  LoRA r=$LORA_R alpha=$LORA_ALPHA dropout=$LORA_DROPOUT  lr=$LR  epochs=$EPOCHS
  batch $PER_DEVICE_BATCH x accum $GRAD_ACCUM = $((PER_DEVICE_BATCH * GRAD_ACCUM)) effective
  output: $RUNS_DIR/train/<condition>_seed<S>/   resumable: rerun the same command
EOF
confirm "Start?" || { log "not started"; exit 0; }

failed=0
for r in "${RUNS[@]}"; do
  cond="${r%%:*}"; seed="${r##*:}"
  name="${cond}_seed${seed}"; [[ $SMOKE == 1 ]] && name="smoke_${name}"
  if is_ok "train_${name}"; then log "skip $name (done)"; continue; fi
  if [[ $MODE == local ]]; then
    if ! RUN_COND="$cond" RUN_SEED="$seed" bash "$PROJ/scripts/slurm/train.sub"; then
      log "RUN FAILED: $name (continuing with the next run; it is marked failed)"; failed=1
    fi
  else
    job_active "train_${name}" && { log "$name already queued"; continue; }
    RUN_COND="$cond" RUN_SEED="$seed" submit_job "train_${name}" "$PROJ/scripts/slurm/train.sub" gpu "$TRAIN_TIME"
  fi
done
status_summary
exit $failed
