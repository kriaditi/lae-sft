# Position-balanced SFT against the "Lost at the End" effect

CS 5787 project: Aditi Kumari (ak2877), Anusha Ramachandrareddy (ar2664), Olamide Oke (ooo38).

Question: does position-balanced LoRA SFT of Qwen2.5-VL-7B-Instruct reduce the accuracy
drop when the gold passage moves from first to last among k = 10 retrieved passages,
compared with standard SFT, and at what cost?

## Layout

```
scripts/config.sh              all paths, Slurm and experiment settings (edit here only)
scripts/lib.sh                 logging, job submission, status markers
scripts/00_setup.sh            stage 0: authors' code + env + models + data
scripts/01_reproduce.sh        stage 1: exact reproduction of the paper's P17 (= frozen Qwen baseline)
scripts/slurm/*.sub            the Slurm jobs the stage scripts submit
lae/                           our Python modules (bash only orchestrates)
third_party/lost_at_the_end_code   authors' repo, pinned to commit 02bd142 (created by setup)
runs/  results/  logs/  status/    created by the scripts
```

## Run

From the repo root on the cluster login node:

```bash
bash scripts/00_setup.sh            # submits a CPU job: uv env + ~35 GB models + InfoSeek data
bash scripts/00_setup.sh --status   # until setup_env shows "ok"
bash scripts/01_reproduce.sh        # asks, then submits ONE GPU job (index + full P17)
bash scripts/01_reproduce.sh --status
```

Output of stage 1: `results/paper_reproduction.csv` (13 cells, ours vs paper, with Wilson CIs).

Every script is safe to rerun: finished stages are skipped (`status/<stage>.ok`), queued jobs
are not submitted twice, caches resume after preemption, and an existing results CSV with
different content is never overwritten (a timestamped copy is written instead).

## Deviations from the authors' release (all recorded)

1. `subset500_v2_qids.json` is not in the authors' repo. `lae/recover_subset.py` rebuilds it
   from their committed P22/P23 summaries: the exact 496 questions P17 reports on.
2. `lae/patch_authors.py` changes two lines of their P17 runner: the output directory
   (so their committed reference numbers are never overwritten) and the subset-size assert
   (496 instead of 500). Retrieval, gold selection, prompt, decoding and scoring are untouched.
   The exact diff is saved next to each run as `authors_patch.diff`.
