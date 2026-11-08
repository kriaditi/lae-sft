# shellcheck shell=bash
# Shared helpers. Sourced by every script; requires config.sh to be sourced first.

ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }
die() { echo "[$(ts)] ERROR: $*" >&2; exit 1; }

need_cmd() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }
need_file() { [[ -e "$1" ]] || die "missing file: $1 ${2:-}"; }

mkdirs() { mkdir -p "$LOG_DIR" "$RUNS_DIR" "$RESULTS_DIR" "$STATUS_DIR" "$HF_HOME"; }

mark_ok()     { rm -f "$STATUS_DIR/$1.failed"; echo "$(ts) ${2:-}" > "$STATUS_DIR/$1.ok"; }
mark_failed() { rm -f "$STATUS_DIR/$1.ok";     echo "$(ts) ${2:-}" > "$STATUS_DIR/$1.failed"; }
is_ok()       { [[ -f "$STATUS_DIR/$1.ok" ]]; }

# Start a timestamped log for a script that runs in the current shell.
start_log() {
  local name="$1"
  LOGFILE="$LOG_DIR/${name}_$(date +%Y%m%d_%H%M%S).log"
  exec > >(tee -a "$LOGFILE") 2>&1
  log "log: $LOGFILE"
}

# submit_job <name> <sub_file> <cpu|gpu> <time> [extra sbatch args...]
# Prints the job id and records it in $LOG_DIR/jobs.tsv.
submit_job() {
  local name="$1" sub="$2" kind="$3" time="$4"; shift 4
  # shellcheck disable=SC2054  # commas in --mail-type=END,FAIL are intended
  local args=(--job-name="lae_${name}" --time="$time" --requeue
              --mail-type=END,FAIL --mail-user="$MAIL_USER"
              --output="$LOG_DIR/${name}_%j.log"
              --export=ALL,PROJ="$PROJ")
  if [[ -n "$SLURM_PARTITION" ]]; then args+=(--partition="$SLURM_PARTITION"); fi
  if [[ "$kind" == gpu ]]; then
    args+=(--gres="$GPU_GRES" --mem="$GPU_JOB_MEM" --cpus-per-task="$GPU_JOB_CPUS")
    if [[ -n "$GPU_CONSTRAINT" ]]; then args+=(--constraint="$GPU_CONSTRAINT"); fi
    if [[ -n "$GPU_NODELIST" ]];   then args+=(--nodelist="$GPU_NODELIST"); fi
  else
    args+=(--mem="$CPU_JOB_MEM" --cpus-per-task="$CPU_JOB_CPUS")
  fi
  local out jid rc=0 errf="$LOG_DIR/.sbatch_stderr"
  out=$(sbatch --parsable "${args[@]}" "$@" "$sub" 2>"$errf") || rc=$?
  [[ $rc -eq 0 ]] || die "sbatch failed (exit $rc): $(cat "$errf")"
  [[ -s "$errf" ]] && log "sbatch said: $(cat "$errf")"
  jid="${out%%;*}"
  [[ "$jid" =~ ^[0-9]+$ ]] || die "could not read a job id from sbatch output: '$out'"
  printf '%s\t%s\t%s\t%s\n' "$(ts)" "$name" "$jid" "$sub" >> "$LOG_DIR/jobs.tsv"
  log "submitted $name as job $jid  (log: $LOG_DIR/${name}_${jid}.log)"
  echo "$jid" > "$STATUS_DIR/$name.jobid"
}

# True if the last job submitted under <name> is still queued or running.
job_active() {
  local f="$STATUS_DIR/$1.jobid"
  [[ -f "$f" ]] || return 1
  squeue -h -j "$(cat "$f")" 2>/dev/null | grep -q .
}

# Ask before anything that costs GPU time. Skipped with --yes.
confirm() {
  [[ "${ASSUME_YES:-0}" == 1 ]] && return 0
  read -r -p "$1 [y/N] " ans
  [[ "$ans" == y || "$ans" == Y ]]
}

status_summary() {
  log "status summary:"
  local f any=0
  for f in "$STATUS_DIR"/*.ok "$STATUS_DIR"/*.failed; do
    [[ -e "$f" ]] || continue; any=1
    printf '   %-28s %-7s %s\n' "$(basename "${f%.*}")" "${f##*.}" "$(cat "$f")"
  done
  if [[ $any != 1 ]]; then echo "   (no stages finished yet)"; fi
  if command -v squeue >/dev/null 2>&1; then
    local q; q=$(squeue -u "${USER:-$(id -un)}" -h -o '%i %j %T %M %R' 2>/dev/null | grep ' lae_' || true)
    if [[ -n "$q" ]]; then echo "   jobs in queue:"; echo "$q" | sed 's/^/     /'; fi
  fi
  return 0
}
