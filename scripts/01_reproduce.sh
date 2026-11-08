#!/usr/bin/env bash
# Stage 1: exact reproduction of the paper's P17 (Qwen2.5-VL-7B x InfoSeek).
# Its k=10 first/middle/last cells are also our frozen-Qwen baseline numbers.
#
#   bash scripts/01_reproduce.sh            # checks, asks, then submits one GPU job
#   bash scripts/01_reproduce.sh --yes      # submit without the prompt
#   bash scripts/01_reproduce.sh --local    # no Slurm (e.g. Lambda H100): run here, inside tmux
#   bash scripts/01_reproduce.sh --status   # status summary + result table if done
set -euo pipefail

PROJ="${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJ
source "$PROJ/scripts/config.sh"
source "$PROJ/scripts/lib.sh"
mkdirs

MODE=submit
case "${1:-}" in
  --yes)    export ASSUME_YES=1 ;;
  --local)  MODE=local ;;
  --status) status_summary
            if [[ -f "$RESULTS_DIR/paper_reproduction.csv" ]]; then
              if command -v column >/dev/null; then
                cut -d, -f1-10 "$RESULTS_DIR/paper_reproduction.csv" | column -s, -t | head -32
              else
                cut -d, -f1-10 "$RESULTS_DIR/paper_reproduction.csv" | head -32
              fi
            fi
            exit 0 ;;
  "") ;;
  *) die "unknown option: $1" ;;
esac

start_log 01_reproduce

# ---- preconditions ----------------------------------------------------------
is_ok setup_env || die "setup not finished. Run: bash scripts/00_setup.sh  (and wait for the job)"
if [[ $MODE == submit ]]; then need_cmd sbatch; else need_cmd nvidia-smi; fi
need_file "$PY" "(venv missing)"
need_file "$AUTHORS_DIR/results/p11_7b_subset/subset500_v2_qids.json"
grep -q '\[lae patch\]' "$AUTHORS_DIR/scripts/run_p17_reader_bottleneck.py" || die "authors' runner not patched; rerun 00_setup.sh"

if is_ok reproduce_p17; then
  log "reproduce_p17 already done: $(cat "$STATUS_DIR/reproduce_p17.ok")"
  log "to redo, delete $STATUS_DIR/reproduce_p17.ok (caches in $RUNS_DIR/p17_repro are reused)"
  exit 0
fi
if [[ $MODE == submit ]] && job_active reproduce_p17; then
  log "job $(cat "$STATUS_DIR/reproduce_p17.jobid") is already queued or running; not submitting again"
  exit 0
fi

# ---- ask before using a GPU --------------------------------------------------
if [[ $MODE == local ]]; then
  log "GPUs on this machine (check nothing else is using the one we take):"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader | sed 's/^/   /'
fi
cat <<EOF
About to run ONE GPU job ($MODE):
  what     : PreFLMR index over 98K InfoSeek passages, then the authors' full P17
             (13 cells: k=1, k in {3,5,10,20} x first/middle/last) on 496 questions
  resources: $( [[ $MODE == local ]] && echo "this machine, GPU ${CUDA_VISIBLE_DEVICES:-0}" || echo "$GPU_GRES, $GPU_JOB_CPUS CPUs, $GPU_JOB_MEM RAM, time limit $REPRO_TIME, --requeue")
  paper ref: P17 wall clock 43.2 min, peak 19.3 GB, on an H100 80GB
             (index build time is not reported in the paper)
  output   : $RUNS_DIR/p17_repro/ and $RESULTS_DIR/paper_reproduction.csv
EOF
confirm "Start?" || { log "not started"; exit 0; }

if [[ $MODE == local ]]; then
  bash "$PROJ/scripts/slurm/reproduce_p17.sub"   # output already goes to this script's log
  status_summary
else
  submit_job reproduce_p17 "$PROJ/scripts/slurm/reproduce_p17.sub" gpu "$REPRO_TIME"
  log "check progress: bash scripts/01_reproduce.sh --status   or   tail -f $LOG_DIR/reproduce_p17_<jobid>.log"
fi
