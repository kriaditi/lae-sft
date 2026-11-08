# shellcheck shell=bash
# Project configuration. Every script sources this file.
# Change values here (or override with environment variables), not inside scripts.

# ---- paths -------------------------------------------------------------
export PROJ="${PROJ:-$HOME/lae-sft}"                  # this repo on the cluster
export AUTHORS_REPO_URL="https://github.com/WeMWish/lost_at_the_end_code"
export AUTHORS_COMMIT="02bd142"                         # pinned: code + numbers we reproduce
export AUTHORS_DIR="$PROJ/third_party/lost_at_the_end_code"
export VENV="$AUTHORS_DIR/.venv"
export PY="$VENV/bin/python"

# Big caches (models ~35 GB, data + images ~tens of GB). Point at large storage if home is small.
export HF_HOME="${HF_HOME:-$PROJ/hf_cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJ/.uv_cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$PROJ/.uv_python}"
export MIN_FREE_GB="${MIN_FREE_GB:-150}"

export LOG_DIR="$PROJ/logs"          # one log per job / script call, timestamped
export RUNS_DIR="$PROJ/runs"         # raw run outputs (caches, predictions); resumable
export RESULTS_DIR="$PROJ/results"   # final CSVs, never overwritten silently
export STATUS_DIR="$PROJ/status"     # <stage>.ok / <stage>.failed markers

# ---- Slurm -------------------------------------------------------------
export MAIL_USER="ak2877@cornell.edu"
export SLURM_PARTITION="${SLURM_PARTITION:-gupta}"
export GPU_GRES="${GPU_GRES:-gpu:nvidia_rtx_6000_ada_generation:1}"
export GPU_CONSTRAINT="${GPU_CONSTRAINT:-}"     # optional --constraint
export GPU_NODELIST="${GPU_NODELIST:-}"         # optional --nodelist
export CPU_JOB_MEM="${CPU_JOB_MEM:-32G}"
export CPU_JOB_CPUS="${CPU_JOB_CPUS:-8}"
export GPU_JOB_MEM="${GPU_JOB_MEM:-96G}"
export GPU_JOB_CPUS="${GPU_JOB_CPUS:-8}"
export SETUP_TIME="${SETUP_TIME:-12:00:00}"
export REPRO_TIME="${REPRO_TIME:-24:00:00}"

# ---- experiment constants (from the authors' P17 protocol) ---------------
export SEED=42
export INFOSEEK_N_EXPECTED=496   # authors' reported n (500 drawn, 4 excluded); see lae/recover_subset.py
export PYTHONUNBUFFERED=1   # show Python progress in logs immediately

# ---- stage 2: training data
export TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-$PROJ/data/train}"
export TRAIN_POOL="${TRAIN_POOL:-150000}"
export TRAIN_N="${TRAIN_N:-20000}"
export DEV_N="${DEV_N:-500}"
export BALANCED_SEEDS="${BALANCED_SEEDS:-0 1}"
export BUILD_TIME="${BUILD_TIME:-24:00:00}"
export BUILD_BATCH="${BUILD_BATCH:-32}"
