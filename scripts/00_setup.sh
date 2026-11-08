#!/usr/bin/env bash
# Stage 0: get the authors' code, patch it minimally, then submit one CPU Slurm job
# that builds the Python environment and downloads models + InfoSeek data.
#
#   bash scripts/00_setup.sh            # on the login node
#   bash scripts/00_setup.sh --local    # do the heavy part in this shell (e.g. inside srun)
#   bash scripts/00_setup.sh --status   # just print the status summary
set -euo pipefail

PROJ="${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJ
source "$PROJ/scripts/config.sh"
source "$PROJ/scripts/lib.sh"
mkdirs

MODE=submit
case "${1:-}" in
  --local)  MODE=local ;;
  --status) status_summary; exit 0 ;;
  "") ;;
  *) die "unknown option: $1" ;;
esac

start_log 00_setup

# ---- 1. checks ---------------------------------------------------------
need_cmd git; need_cmd python3; need_cmd df
if [[ $MODE == submit ]]; then need_cmd sbatch; need_cmd squeue; need_cmd sinfo; fi

free_gb=$(df -BG --output=avail "$HF_HOME" | tail -1 | tr -dc '0-9')
log "free disk at HF_HOME ($HF_HOME): ${free_gb} GB (want >= ${MIN_FREE_GB} GB)"
if (( free_gb < MIN_FREE_GB )); then
  die "not enough disk. Set HF_HOME in scripts/config.sh to larger storage and rerun."
fi

if command -v sinfo >/dev/null; then
  log "GPU partitions and nodes visible to you:"
  sinfo -h -o '%P %G %D %t' | grep -i gpu | sort -u | sed 's/^/   /' || true
fi

# ---- 2. authors' code at the pinned commit ---------------------------------
if [[ ! -d "$AUTHORS_DIR/.git" ]]; then
  log "cloning $AUTHORS_REPO_URL"
  mkdir -p "$(dirname "$AUTHORS_DIR")"
  git clone "$AUTHORS_REPO_URL" "$AUTHORS_DIR"
fi
head=$(git -C "$AUTHORS_DIR" rev-parse --short=7 HEAD)
if [[ "$head" != "$AUTHORS_COMMIT" ]]; then
  if [[ -n "$(git -C "$AUTHORS_DIR" status --porcelain --untracked-files=no)" ]]; then
    die "authors' repo has local edits and is at $head, not $AUTHORS_COMMIT. Resolve by hand."
  fi
  git -C "$AUTHORS_DIR" checkout -q "$AUTHORS_COMMIT"
fi
log "authors' repo at $(git -C "$AUTHORS_DIR" rev-parse --short=7 HEAD)"

# ---- 3. recovered subset + minimal patches (stdlib python) -----------------
python3 "$PROJ/lae/recover_subset.py" --authors-dir "$AUTHORS_DIR" --expected "$INFOSEEK_N_EXPECTED"
python3 "$PROJ/lae/patch_authors.py"  --authors-dir "$AUTHORS_DIR"
git -C "$AUTHORS_DIR" diff --stat | sed 's/^/   /'

# ---- 4. HF token (models and data are public; a token avoids rate limits) ----
if [[ -z "${HF_TOKEN:-}" && ! -f "$HF_HOME/token" ]]; then
  log "note: no Hugging Face token found. If downloads fail with 401/429, run:"
  log "      $VENV/bin/huggingface-cli login   (after this setup job builds the venv)"
fi

# ---- 5. environment + downloads ---------------------------------------------
if is_ok setup_env; then
  log "setup_env already done ($(cat "$STATUS_DIR/setup_env.ok")). Delete $STATUS_DIR/setup_env.ok to redo."
elif [[ $MODE == submit ]] && job_active setup_env; then
  log "setup_env job $(cat "$STATUS_DIR/setup_env.jobid") is already queued or running; not submitting again"
elif [[ $MODE == local ]]; then
  bash "$PROJ/scripts/slurm/setup_env.sub"
else
  submit_job setup_env "$PROJ/scripts/slurm/setup_env.sub" cpu "$SETUP_TIME"
  log "watch it with: squeue -u ${USER:-$(id -un)}    then rerun: bash scripts/00_setup.sh --status"
fi

status_summary
