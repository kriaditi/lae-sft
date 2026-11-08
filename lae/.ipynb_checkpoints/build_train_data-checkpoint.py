"""Stage 2: build the SFT training data (standard and position-balanced).

Design (agreed 2026-11-08):
  * Questions: M2KR InfoSeek TRAIN split. Removed before sampling: any question that
    shares a question id, image file, entity, or gold passage id with the 496
    evaluation questions (no leakage), or has no answer / no positive in the KB.
  * Candidate pool: the remaining questions in a seed-locked random order (PCG64, 42).
  * Retrieval: the authors' frozen PreFLMR ViT-G and their query (instruction + question
    + image), top-20 over the InfoSeek train passage KB.
  * Kept: exactly one positive passage in the top 10. The 10 passages are then
    "gold + 9 non-gold distractors", the same structure as the evaluation cells.
    Pool questions are processed in order until TRAIN_N + DEV_N are kept.
  * standard     : the 10 passages in retrieval order (gold at its natural rank).
  * balanced_s<S>: the same 10 passages per question; gold moved to a slot drawn so
                   that every slot gets exactly N/10 questions (seed S); the 9
                   distractors keep their retrieval order.
  * SFT target: M2KR gold_answer (canonical answer), else the first accepted answer.
  * Every condition has the same questions, in the same order, with the same passages.

Phases are resumable (cached outputs are reused). Hard checks at the end; any failure
exits non-zero. Output: <out>/final/<condition>/{train,dev}.jsonl + report + manifest.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

M2KR = "BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR"
IMAGES = "BByrneLab/M2KR_Images"
TRAIN_TAR = "Infoseek/infoseek_train_images.tar"
K = 10
TOP = 20
WANTED_COLS = ["question_id", "image_id", "question", "answers", "gold_answer", "entity_id",
               "entity_text", "img_path", "question_type", "pos_item_ids", "instruction"]


def log(msg: str) -> None:
    print(f"[build {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def m2kr_files(prefix: str) -> list[str]:
    info = HfApi().dataset_info(M2KR)
    return sorted(s.rfilename for s in info.siblings if s.rfilename.startswith(prefix))


def load_parquets(files: list[str], cols: list[str]) -> list[dict]:
    rows = []
    for f in files:
        p = hf_hub_download(M2KR, f, repo_type="dataset")
        present = [c for c in cols if c in pq.read_schema(p).names]
        rows.extend(pq.read_table(p, columns=present).to_pylist())
    return rows


# ---------------------------------------------------------------- phase 1
def phase_candidates(args, out: Path, stats: dict) -> list[dict]:
    path = out / "candidates.jsonl"
    if path.exists():
        cands = read_jsonl(path)
        stats.update(json.loads((out / "candidates_stats.json").read_text()))
        log(f"phase 1: reusing {len(cands)} candidates")
        return cands

    log("phase 1: load evaluation questions (for the leakage filter)")
    eval_qids = set(json.loads((args.authors_dir / "results/p11_7b_subset/subset500_v2_qids.json")
                               .read_text())["qids"])
    ev = [r for r in load_parquets(m2kr_files("Infoseek_data/test"), WANTED_COLS)
          if r["question_id"] in eval_qids]
    assert len(ev) == len(eval_qids), f"found {len(ev)} of {len(eval_qids)} eval rows"
    ev_img = {Path(r["img_path"]).name for r in ev}
    ev_ent = {r.get("entity_id") for r in ev} - {None, ""}
    ev_gold = {p for r in ev for p in (r.get("pos_item_ids") or [])}

    log("phase 1: load train questions")
    rows = load_parquets(m2kr_files("Infoseek_data/train"), WANTED_COLS)
    log(f"  train rows: {len(rows)}  columns: {sorted(rows[0].keys())}")
    kb_ids = {r["passage_id"] for r in load_parquets(m2kr_files("Infoseek_passages/train"), ["passage_id"])}

    drop = Counter()
    keep = []
    for r in rows:
        pos = list(r.get("pos_item_ids") or [])
        if r["question_id"] in eval_qids:
            drop["same_qid_as_eval"] += 1
        elif Path(r["img_path"]).name in ev_img:
            drop["same_image_as_eval"] += 1
        elif r.get("entity_id") in ev_ent:
            drop["same_entity_as_eval"] += 1
        elif set(pos) & ev_gold:
            drop["shares_gold_passage_with_eval"] += 1
        elif not (r.get("answers") or []):
            drop["no_answer"] += 1
        elif not (set(pos) & kb_ids):
            drop["no_positive_in_kb"] += 1
        else:
            keep.append(r)
    del rows
    gc.collect()

    rng = np.random.Generator(np.random.PCG64(args.seed))
    order = rng.permutation(len(keep))[: args.pool]
    cands = [keep[int(i)] for i in order]
    for r in cands:
        r["pos_item_ids"] = list(r.get("pos_item_ids") or [])
        r["answers"] = [str(a) for a in r["answers"]]
        # SFT target: M2KR's canonical gold_answer, falling back to the first accepted answer.
        r["target"] = str(r.get("gold_answer") or "").strip() or r["answers"][0]
    stats["train_rows"] = len(keep) + sum(drop.values())
    stats["dropped"] = dict(drop)
    stats["eligible_after_filter"] = len(keep)
    stats["pool"] = len(cands)
    write_jsonl(path, cands)
    (out / "candidates_stats.json").write_text(json.dumps(stats, indent=2))
    log(f"phase 1: dropped {dict(drop)}; eligible {len(keep)}; pool {len(cands)}")
    return cands


# ---------------------------------------------------------------- phase 2
def phase_images(args, cands: list[dict], image_dir: Path, stats: dict) -> None:
    wanted = {}
    for r in cands:
        rel = r["img_path"]
        if not (image_dir / rel).is_file():
            wanted.setdefault(Path(rel).name, rel)
    n_unique = len({r["img_path"] for r in cands})
    log(f"phase 2: {n_unique} unique images for the pool, {len(wanted)} to extract")
    if wanted:
        tar = hf_hub_download(IMAGES, TRAIN_TAR, repo_type="dataset")
        log(f"  tar: {tar} ({Path(tar).stat().st_size / 1e9:.1f} GB), single streaming pass")
        t0, seen = time.time(), 0
        with tarfile.open(tar, mode="r|*") as tf:
            for m in tf:
                seen += 1
                if seen % 100000 == 0:
                    log(f"  scanned {seen} tar entries, {len(wanted)} still wanted, {time.time() - t0:.0f}s")
                if not m.isfile():
                    continue
                rel = wanted.get(Path(m.name).name)
                if rel is None:
                    continue
                dst = image_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(tf.extractfile(m).read())
                del wanted[Path(m.name).name]
                if not wanted:
                    break
    stats["pool_images_missing_in_tar"] = len(wanted)
    if wanted:
        log(f"  {len(wanted)} images not in the tar; those questions are dropped")


# ---------------------------------------------------------------- phase 3
def qualify(top: list[str], pos: list[str]) -> int | None:
    """Natural gold slot if exactly one positive is in the top K, else None."""
    hits = [i for i, p in enumerate(top[:K]) if p in set(pos)]
    return hits[0] if len(hits) == 1 else None


def phase_retrieve(args, cands: list[dict], image_dir: Path, out: Path, stats: dict) -> dict[str, list[str]]:
    sys.path.insert(0, str(args.authors_dir / "src"))
    import torch
    from tts_kbvqa.data.infoseek import InfoSeekSample, load_passages
    from tts_kbvqa.retrieval.preflmr import PreFLMRRetriever

    torch.backends.cuda.enable_cudnn_sdp(False)  # as in the authors' runners
    target = args.train_n + args.dev_n
    path = out / f"retrieval_top{TOP}.jsonl"
    done: dict[str, list[str]] = {}
    if path.exists():
        for r in read_jsonl(path):
            done[r["qid"]] = r["top"]
    usable = [r for r in cands if (image_dir / r["img_path"]).is_file()]
    kept = sum(1 for r in usable if r["question_id"] in done
               and qualify(done[r["question_id"]], r["pos_item_ids"]) is not None)
    log(f"phase 3: {len(done)} retrieved before, {kept} qualify, target {target}")
    if kept >= target:
        return done

    # Index over the train KB. Reuse the evaluation index if the KB is the same passage set.
    kb_file = hf_hub_download(M2KR, m2kr_files("Infoseek_passages/train")[0], repo_type="dataset")
    test_kb = hf_hub_download(M2KR, m2kr_files("Infoseek_passages/test")[0], repo_type="dataset")
    ids_train = set(pq.read_table(kb_file, columns=["passage_id"]).column(0).to_pylist())
    ids_test = set(pq.read_table(test_kb, columns=["passage_id"]).column(0).to_pylist())
    if ids_train == ids_test:
        index_dir = args.authors_dir / "data/index/preflmr_infoseek"
        log("  train KB has exactly the evaluation KB's passage ids: reusing that index")
    else:
        index_dir = args.authors_dir / "data/index/preflmr_infoseek_train"
        log(f"  train KB differs from the evaluation KB: building/using {index_dir}")
    stats["kb_passages"] = len(ids_train)
    stats["kb_same_as_eval_kb"] = ids_train == ids_test

    retriever = PreFLMRRetriever(dtype=torch.bfloat16)
    man = index_dir / "manifest.json"
    if not (man.exists() and "build_wall_s" in json.loads(man.read_text())):
        from tts_kbvqa.data.infoseek import iter_passages
        retriever.build_index(passages=iter_passages(kb_file), out_dir=str(index_dir),
                              shard_size=5000, batch_size=16, verbose=True)
    index = retriever.load_index(str(index_dir))
    id_to_passage = {p.passage_id: p for p in load_passages(kb_file)}

    todo = [r for r in usable if r["question_id"] not in done]
    t0, n_new = time.time(), 0
    with path.open("a") as fh:
        for i in range(0, len(todo), args.batch):
            batch = todo[i:i + args.batch]
            samples = [InfoSeekSample(
                question_id=r["question_id"], image_id=r.get("image_id") or "",
                question=r["question"], answers=r["answers"], entity_id=r.get("entity_id") or "",
                entity_text=r.get("entity_text") or "", image_filename=r["img_path"],
                question_type=r.get("question_type") or "", pos_item_ids=r["pos_item_ids"],
                instruction=r.get("instruction") or "", image_dir=image_dir) for r in batch]
            tops = retriever.retrieve_batch(samples, index, id_to_passage, k=TOP)
            for r, rps in zip(batch, tops):
                top = [rp.passage_id for rp in rps]
                done[r["question_id"]] = top
                fh.write(json.dumps({"qid": r["question_id"], "top": top}) + "\n")
                if qualify(top, r["pos_item_ids"]) is not None:
                    kept += 1
            fh.flush()
            n_new += len(batch)
            if (i // args.batch) % 50 == 0:
                rate = (time.time() - t0) / n_new
                log(f"  retrieved {n_new} new ({len(done)} total), kept {kept}/{target}, "
                    f"{rate:.2f}s/q, keep rate {kept / len(done):.2f}")
            if kept >= target:
                break
    stats["retrieval_s_per_question"] = round((time.time() - t0) / max(n_new, 1), 3)
    log(f"phase 3: done, {len(done)} retrieved, {kept} qualify")
    retriever.unload()
    del index
    gc.collect()
    if kept < target:
        raise SystemExit(f"ERROR: only {kept} qualifying questions in the pool of {len(cands)}; "
                         f"rerun with a larger --pool")
    return done


# ---------------------------------------------------------------- phase 4
def place(top10: list[str], natural: int, slot: int) -> list[str]:
    gold = top10[natural]
    distractors = top10[:natural] + top10[natural + 1:]
    return distractors[:slot] + [gold] + distractors[slot:]


def balanced_slots(n: int, seed: int, stream: int) -> np.ndarray:
    assert n % K == 0, f"n={n} must be a multiple of {K} for an exactly uniform split"
    rng = np.random.Generator(np.random.PCG64([seed, stream]))
    return rng.permutation(np.arange(n) % K)


def phase_build(args, cands, done, image_dir: Path, out: Path, stats: dict) -> dict[str, dict]:
    items = []
    for r in cands:
        top = done.get(r["question_id"])
        if top is None or not (image_dir / r["img_path"]).is_file():
            continue
        nat = qualify(top, r["pos_item_ids"])
        if nat is None:
            continue
        items.append({"qid": r["question_id"], "question": r["question"], "answers": r["answers"],
                      "target": r["target"], "question_type": r.get("question_type"),
                      "entity_id": r.get("entity_id"), "img_path": r["img_path"],
                      "top10": top[:K], "natural_slot": nat, "gold_pid": top[nat],
                      "pos_item_ids": r["pos_item_ids"]})
        if len(items) == args.train_n + args.dev_n:
            break
    splits = {"train": items[:args.train_n], "dev": items[args.train_n:]}
    stats["kept_train"], stats["kept_dev"] = len(splits["train"]), len(splits["dev"])

    conds = {"standard": None, **{f"balanced_s{s}": s for s in args.balanced_seeds}}
    final = out / "final"
    hist: dict[str, dict] = {}
    for cname, seed in conds.items():
        for split, rows in splits.items():
            slots = ([r["natural_slot"] for r in rows] if seed is None
                     else balanced_slots(len(rows), seed, 0 if split == "train" else 1).tolist())
            recs = []
            for r, slot in zip(rows, slots):
                pids = place(r["top10"], r["natural_slot"], int(slot))
                recs.append({"qid": r["qid"], "question": r["question"], "answers": r["answers"],
                             "target": r["target"], "question_type": r["question_type"],
                             "img_path": r["img_path"], "passage_ids": pids,
                             "gold_pid": r["gold_pid"], "gold_slot": int(slot),
                             "natural_slot": r["natural_slot"], "pos_item_ids": r["pos_item_ids"],
                             "condition": cname})
            write_jsonl(final / cname / f"{split}.jsonl", recs)
            hist[f"{cname}/{split}"] = dict(sorted(Counter(slots).items()))
    stats["gold_slot_histograms"] = hist
    return splits


# ---------------------------------------------------------------- phase 5
def phase_lengths(args, splits, image_dir: Path, stats: dict) -> None:
    """Prompt length in tokens, built exactly as the authors' reader builds it."""
    sys.path.insert(0, str(args.authors_dir / "src"))
    from PIL import Image
    from transformers import AutoProcessor
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    from tts_kbvqa.data.infoseek import load_passages
    from tts_kbvqa.reader.qwen import SYSTEM_PROMPT, VISION_MAX_PIXELS, VISION_MIN_PIXELS, passage_block

    proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    kb_file = hf_hub_download(M2KR, m2kr_files("Infoseek_passages/train")[0], repo_type="dataset")
    id_to_passage = {p.passage_id: p for p in load_passages(kb_file)}
    totals = []
    for r in splits["train"]:
        # Same message layout as the authors' QwenReader._prepare_inputs (image first).
        user = f"{r['question']}\n\nContext:\n{passage_block([id_to_passage[p] for p in r['top10']])}"
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        text += r["target"] + "<|im_end|>\n"
        n_text = len(proc.tokenizer(text).input_ids) - 1  # minus the single <|image_pad|> placeholder
        w, h = Image.open(image_dir / r["img_path"]).size
        rh, rw = smart_resize(h, w, factor=28, min_pixels=VISION_MIN_PIXELS, max_pixels=VISION_MAX_PIXELS)
        totals.append(n_text + (rh // 28) * (rw // 28))
    a = np.array(totals)
    stats["train_seq_tokens"] = {"mean": int(a.mean()), "p50": int(np.percentile(a, 50)),
                                 "p95": int(np.percentile(a, 95)), "max": int(a.max()),
                                 "total_tokens": int(a.sum())}
    log(f"phase 5: training sequence tokens {stats['train_seq_tokens']}")


# ---------------------------------------------------------------- phase 6
def phase_check(args, out: Path, image_dir: Path, stats: dict) -> list[str]:
    errors = []
    final = out / "final"
    conds = ["standard"] + [f"balanced_s{s}" for s in args.balanced_seeds]
    eval_qids = set(json.loads((args.authors_dir / "results/p11_7b_subset/subset500_v2_qids.json")
                               .read_text())["qids"])
    for split, n in (("train", args.train_n), ("dev", args.dev_n)):
        data = {c: read_jsonl(final / c / f"{split}.jsonl") for c in conds}
        base = data["standard"]
        if len(base) != n:
            errors.append(f"{split}: {len(base)} rows, expected {n}")
        if len({r["qid"] for r in base}) != len(base):
            errors.append(f"{split}: duplicate qids")
        if {r["qid"] for r in base} & eval_qids:
            errors.append(f"{split}: contains evaluation qids")
        for c, rows in data.items():
            if [r["qid"] for r in rows] != [r["qid"] for r in base]:
                errors.append(f"{c}/{split}: question order differs from standard")
                continue
            for r, b in zip(rows, base):
                pids = r["passage_ids"]
                if len(pids) != K or sorted(pids) != sorted(b["passage_ids"]):
                    errors.append(f"{c}/{split} {r['qid']}: passage set differs"); break
                if pids[r["gold_slot"]] != r["gold_pid"]:
                    errors.append(f"{c}/{split} {r['qid']}: gold not at gold_slot"); break
                if sum(p in set(r["pos_item_ids"]) for p in pids) != 1:
                    errors.append(f"{c}/{split} {r['qid']}: not exactly one positive among the 10"); break
                if [p for p in pids if p != r["gold_pid"]] != [p for p in b["passage_ids"] if p != b["gold_pid"]]:
                    errors.append(f"{c}/{split} {r['qid']}: distractor order changed"); break
                if not (image_dir / r["img_path"]).is_file():
                    errors.append(f"{c}/{split} {r['qid']}: image missing"); break
            if c == "standard":
                if any(r["gold_slot"] != r["natural_slot"] for r in rows):
                    errors.append(f"standard/{split}: gold not at natural slot")
            else:
                h = Counter(r["gold_slot"] for r in rows)
                if set(h.values()) != {n // K} or len(h) != K:
                    errors.append(f"{c}/{split}: slot histogram not uniform {dict(h)}")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--authors-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pool", type=int, default=60000)
    ap.add_argument("--train-n", type=int, default=20000)
    ap.add_argument("--dev-n", type=int, default=500)
    ap.add_argument("--balanced-seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    image_dir = args.authors_dir / "data/infoseek/images/train"
    t_start = time.time()

    stats: dict = {"design": "see lae/build_train_data.py docstring", "seed": args.seed,
                   "train_n": args.train_n, "dev_n": args.dev_n, "k": K, "retrieve_top": TOP}
    cands = phase_candidates(args, args.out, stats)
    phase_images(args, cands, image_dir, stats)
    done = phase_retrieve(args, cands, image_dir, args.out, stats)
    splits = phase_build(args, cands, done, image_dir, args.out, stats)
    phase_lengths(args, splits, image_dir, stats)
    errors = phase_check(args, args.out, image_dir, stats)

    stats["wall_clock_min"] = round((time.time() - t_start) / 60, 1)
    stats["hard_check_errors"] = errors
    final = args.out / "final"
    (final / "build_stats.json").write_text(json.dumps(stats, indent=2))
    manifest = {str(p.relative_to(final)): sha256(p) for p in sorted(final.rglob("*.jsonl"))}
    (final / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(json.dumps({k: v for k, v in stats.items() if k != "gold_slot_histograms"}, indent=1))
    for name, h in stats["gold_slot_histograms"].items():
        log(f"  gold slot histogram {name:24s} {h}")
    if errors:
        for e in errors:
            log(f"HARD CHECK FAILED: {e}")
        return 1
    log("all hard checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
