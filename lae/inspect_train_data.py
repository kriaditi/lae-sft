"""Stage 2, step 0: inspect the M2KR InfoSeek TRAIN files before building training data.

Downloads only the text parquets (no images) and prints facts we need to design
the builder: file names and sizes, columns, row counts, a sample row, how many
train questions have their gold passage in the train KB, and how many train
questions overlap the 496 evaluation questions (same image or same entity).

Output: printed, and saved to runs/inspect_train/report.txt. Read-only otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

M2KR = "BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR"
IMAGES = "BByrneLab/M2KR_Images"

lines: list[str] = []


def out(s: str = "") -> None:
    print(s, flush=True)
    lines.append(s)


def list_files(repo: str, prefix: str) -> list[tuple[str, int]]:
    info = HfApi().dataset_info(repo, files_metadata=True)
    return sorted((s.rfilename, s.size or 0) for s in info.siblings if s.rfilename.startswith(prefix))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--authors-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    authors = Path(args.authors_dir)

    out("== M2KR InfoSeek files ==")
    m2kr_files = list_files(M2KR, "Infoseek")
    for name, size in m2kr_files:
        out(f"  {name:70s} {size / 1e6:9.1f} MB")
    out("== M2KR_Images InfoSeek files ==")
    for name, size in list_files(IMAGES, "Infoseek"):
        out(f"  {name:70s} {size / 1e9:9.2f} GB")

    train_q = [n for n, _ in m2kr_files if n.startswith("Infoseek_data/train")]
    train_kb = [n for n, _ in m2kr_files if n.startswith("Infoseek_passages/train")]
    test_kb = [n for n, _ in m2kr_files if n.startswith("Infoseek_passages/test")]
    if not train_q or not train_kb:
        out(f"ERROR: train question files {train_q} or train KB files {train_kb} not found")
        Path(args.out).write_text("\n".join(lines) + "\n")
        return 1

    def load(files: list[str], cols=None):
        tables = [pq.read_table(hf_hub_download(M2KR, f, repo_type="dataset"), columns=cols) for f in files]
        return [r for t in tables for r in t.to_pylist()]

    out("\n== train questions ==")
    qs = load(train_q)
    out(f"  rows: {len(qs)}   columns: {sorted(qs[0].keys())}")
    sample = {k: (str(v)[:150]) for k, v in qs[0].items()}
    out(f"  sample row: {json.dumps(sample, indent=1)}")
    out(f"  question_type: {Counter(r.get('question_type') for r in qs).most_common(6)}")
    npos = Counter(min(len(r.get('pos_item_ids') or []), 5) for r in qs)
    out(f"  #positives per question (5 = 5+): {sorted(npos.items())}")
    nans = Counter(min(len(r.get('answers') or []), 5) for r in qs)
    out(f"  #answers per question (5 = 5+): {sorted(nans.items())}")
    out(f"  unique images: {len({r.get('img_path') for r in qs})}   "
        f"unique entities: {len({r.get('entity_id') for r in qs})}")

    out("\n== train KB ==")
    kb_ids = {r["passage_id"] for r in load(train_kb, ["passage_id"])}
    out(f"  passages: {len(kb_ids)}")
    test_ids = {r["passage_id"] for r in load(test_kb, ["passage_id"])}
    out(f"  test KB passages: {len(test_ids)}   shared ids with train KB: {len(kb_ids & test_ids)}")
    has_gold = sum(1 for r in qs if set(r.get("pos_item_ids") or []) & kb_ids)
    out(f"  train questions with >=1 positive in train KB: {has_gold} / {len(qs)}")

    out("\n== overlap with the 496 evaluation questions ==")
    eval_qids = set(json.loads((authors / "results/p11_7b_subset/subset500_v2_qids.json").read_text())["qids"])
    val_files = [n for n, _ in m2kr_files if n.startswith("Infoseek_data/test")]
    ev = [r for r in load(val_files) if r["question_id"] in eval_qids]
    out(f"  eval rows found: {len(ev)}")
    ev_img = {Path(r["img_path"]).name for r in ev}
    ev_ent = {r.get("entity_id") for r in ev}
    ev_gold = {p for r in ev for p in (r.get("pos_item_ids") or [])}
    same_img = sum(1 for r in qs if Path(r["img_path"]).name in ev_img)
    same_ent = sum(1 for r in qs if r.get("entity_id") in ev_ent)
    same_gold = sum(1 for r in qs if set(r.get("pos_item_ids") or []) & ev_gold)
    out(f"  train questions sharing an image file with eval: {same_img}")
    out(f"  train questions sharing an entity with eval:     {same_ent}")
    out(f"  train questions sharing a gold passage id with eval: {same_gold}")
    qid_overlap = len(eval_qids & {r['question_id'] for r in qs})
    out(f"  train question ids equal to eval ids: {qid_overlap}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    out(f"\nsaved: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
