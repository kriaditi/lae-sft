#!/usr/bin/env bash
# Stage 2, step 0: inspect the InfoSeek TRAIN files (text only, no images, no GPU).
#   bash scripts/02a_inspect_data.sh
set -euo pipefail
PROJ="${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJ
source "$PROJ/scripts/config.sh"
source "$PROJ/scripts/lib.sh"
mkdirs
start_log 02a_inspect_data

is_ok setup_env || die "run setup first: bash scripts/00_setup.sh --local"
"$PY" "$PROJ/lae/inspect_train_data.py" \
  --authors-dir "$AUTHORS_DIR" \
  --out "$RUNS_DIR/inspect_train/report.txt"
