"""E-VQA transfer: evaluate the frozen reader and every InfoSeek-trained adapter on
Encyclopedic-VQA with the authors' P20 protocol (Qwen2.5-VL-7B part only).

Reused unchanged from the authors' scripts/run_p20_evqa_replication.py (imported as a
module, not edited): PreFLMR index over the 51,472-passage M2KR E-VQA KB, the
committed subset500_evqa_v1 (500 qids), top-50 retrieval, gold pick + 9 distractors
(build_plan), the Qwen reader phase, and their one-way canonical matcher
(correct_canonical, the paper's headline E-VQA metric).

Our differences, all recorded:
  * only the ~500 subset images are extracted from the two image archives (the
    authors extract everything); same files, much less disk
  * InternVL3 is not run (not part of our study)
  * trained adapters reuse the frozen run's retrieval cache, so every model sees the
    same passage lists; the adapter is loaded by our [lae patch] in QwenReader

Modes:
  --check                       archive sizes, subset size, disk (no GPU)
  --run NAME [--adapter DIR]    one model -> <root>/<NAME>/
  --compare                     tables + paired bootstrap over all finished runs
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import shutil
import sys
import tarfile
import time
import zipfile
from pathlib import Path

B, SEED = 10_000, 42
POS = ("first", "middle", "last")


def log(msg: str) -> None:
    print(f"[evqa {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_p20(authors_dir: Path):
    sys.path.insert(0, str(authors_dir / "src"))
    spec = importlib.util.spec_from_file_location("p20", authors_dir / "scripts/run_p20_evqa_replication.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def subset_qids(authors_dir: Path) -> list[str]:
    rows = [json.loads(l) for l in (authors_dir / "results/p20/subset500_evqa_v1.jsonl").open() if l.strip()]
    return [r["qid"] for r in rows if "qid" in r]


def ensure_subset_images(p20, authors_dir: Path, qids: list[str]) -> None:
    from huggingface_hub import hf_hub_download
    ev = p20.evqa_data
    samples = {s.question_id: s for s in ev.load_test_samples(p20.IMAGE_DIR)}
    need = {Path(samples[q].image_filename).name: samples[q].image_filename for q in qids
            if not samples[q].image_path.is_file()}
    log(f"images: {len(qids)} questions, {len(need)} images to extract")
    for kind, fname in (("zip", ev.INAT_ZIP), ("tar", ev.GOOGLE_LANDMARK_TAR)):
        if not need:
            break
        path = Path(hf_hub_download(ev.IMAGES_REPO, fname, repo_type="dataset"))
        log(f"  scanning {fname} ({path.stat().st_size / 1e9:.1f} GB)")
        if kind == "zip":
            with zipfile.ZipFile(path) as zf:
                for info in zf.infolist():
                    rel = need.get(Path(info.filename).name)
                    if rel and not info.is_dir():
                        dst = p20.IMAGE_DIR / rel
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        dst.write_bytes(zf.read(info))
                        del need[Path(info.filename).name]
        else:
            with tarfile.open(path, "r|*") as tf:
                for m in tf:
                    rel = need.get(Path(m.name).name) if m.isfile() else None
                    if rel:
                        dst = p20.IMAGE_DIR / rel
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        dst.write_bytes(tf.extractfile(m).read())
                        del need[Path(m.name).name]
                        if not need:
                            break
    if need:
        raise SystemExit(f"ERROR: {len(need)} subset images not found in the archives, e.g. {list(need.values())[:3]}")
    missing = [q for q in qids if not samples[q].image_path.is_file()]
    if missing:
        raise SystemExit(f"ERROR: {len(missing)} questions still have no image, e.g. {missing[:3]}")
    log(f"images: verified, all {len(qids)} questions have their image file")


def cmd_check(authors_dir: Path, root: Path) -> int:
    from huggingface_hub import HfApi
    p20 = load_p20(authors_dir)
    info = HfApi().dataset_info(p20.evqa_data.IMAGES_REPO, files_metadata=True)
    for s in info.siblings:
        if s.rfilename in (p20.evqa_data.INAT_ZIP, p20.evqa_data.GOOGLE_LANDMARK_TAR):
            log(f"archive {s.rfilename}: {(s.size or 0) / 1e9:.1f} GB (downloaded once to the HF cache)")
    log(f"subset: {len(subset_qids(authors_dir))} qids (authors' committed subset500_evqa_v1)")
    root.mkdir(parents=True, exist_ok=True)
    log(f"free disk at {root}: {shutil.disk_usage(root).free / 1e9:.0f} GB")
    return 0


def cmd_run(authors_dir: Path, root: Path, name: str, adapter: str) -> int:
    import torch
    p20 = load_p20(authors_dir)
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    p20.OUT = out
    p20.SUB_PATH = root / "subset500_evqa_v1.jsonl"
    if not p20.SUB_PATH.exists():
        shutil.copy(authors_dir / "results/p20/subset500_evqa_v1.jsonl", p20.SUB_PATH)
    os.environ["LAE_ADAPTER"] = adapter or ""
    torch.manual_seed(SEED)
    t0 = time.time()

    qids = subset_qids(authors_dir)
    ensure_subset_images(p20, authors_dir, qids)
    full = p20.evqa_data.load_test_samples(p20.IMAGE_DIR)
    p20.ensure_index()
    sub = set(p20.draw_and_audit_subset(full))
    assert sub == set(qids), "subset differs from the authors' committed file"
    samples = [s for s in full if s.question_id in sub]
    by_qid = {s.question_id: s for s in samples}
    gold = {s.question_id: list(s.pos_item_ids) for s in samples}
    id_to_passage = {p.passage_id: p for p in p20.evqa_data.load_passages(p20.evqa_data.download_passages_parquet())}

    ref = root / "frozen" / "retrieval_p20.jsonl"
    if name != "frozen":
        if not ref.exists():
            raise SystemExit("run the frozen model first (it builds the retrieval cache)")
        mine = out / "retrieval_p20.jsonl"
        if mine.exists() and mine.read_bytes() != ref.read_bytes():
            raise SystemExit(f"{mine} differs from the frozen run's retrieval cache")
        shutil.copy(ref, mine)
    retrieval = p20.phase_retrieve(samples, id_to_passage)
    plan = p20.build_plan(samples, retrieval, gold)
    torch.cuda.reset_peak_memory_stats()
    rc = p20.run_qwen_phase(by_qid, plan, id_to_passage)
    preds = p20.build_predictions(rc, plan, by_qid, retrieval, id_to_passage)
    p20.analyze_errors(preds, plan, retrieval, id_to_passage)
    summary = {"name": name, "lae_adapter": adapter or "", "n": len(plan),
               "n_fallback": sum(p["gold_via_fallback"] for p in plan.values()),
               "n_excluded": len(samples) - len(plan), "cells": {},
               "peak_alloc_gb": torch.cuda.max_memory_allocated() / 1e9, "wall_s": time.time() - t0,
               "device": torch.cuda.get_device_name(0)}
    for pos in POS:
        rows = preds[pos]
        (out / f"predictions_qwen_{pos}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        hits = sum(r["correct_canonical"] for r in rows)
        acc, lo, hi = p20.wilson_ci(hits, len(rows))
        summary["cells"][pos] = {"n": len(rows), "hits": hits, "acc": acc, "ci_low": lo, "ci_high": hi,
                                 "acc_any_alternative": sum(r["correct_any_alternative"] for r in rows) / len(rows)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"{name}: n={summary['n']} fallback={summary['n_fallback']}  first/middle/last = "
        + " / ".join(f"{100 * summary['cells'][p]['acc']:.2f}" for p in POS))
    return 0


def cmd_compare(authors_dir: Path, root: Path, out_dir: Path, planned_seeds: int) -> int:
    import numpy as np
    paper = json.loads((authors_dir / "results/p20/per_cell_summary.json").read_text())["models"]["qwen"]["per_cell"]

    def load(d: Path) -> dict:
        return {p: {json.loads(l)["qid"]: float(json.loads(l)["correct_canonical"])
                    for l in (d / f"predictions_qwen_{p}.jsonl").open() if l.strip()} for p in POS}

    frozen = load(root / "frozen")
    qids = sorted(frozen["first"])
    n = len(qids)
    runs = {"standard": [], "balanced": []}
    per_seed = []
    for d in sorted(root.iterdir()):
        m = re.fullmatch(r"(standard|balanced_s\d+)_seed(\d+)", d.name)
        if m and (d / "summary.json").exists():
            r = load(d)
            assert all(set(r[p]) == set(qids) for p in POS), f"{d.name}: question set differs"
            runs["standard" if m.group(1) == "standard" else "balanced"].append((d.name, r))
    vec = lambda cells, p: np.array([cells[p][q] for q in qids])
    models = {"Frozen Qwen": {p: vec(frozen, p) for p in POS}}
    status = {"Frozen Qwen": "measured"}
    for cond, label in (("standard", "Standard LoRA SFT"), ("balanced", "Balanced LoRA SFT")):
        if runs[cond]:
            models[label] = {p: np.mean([vec(r, p) for _, r in runs[cond]], axis=0) for p in POS}
            status[label] = (f"mean of {len(runs[cond])} seeds" if len(runs[cond]) >= planned_seeds
                             else f"PRELIMINARY: {len(runs[cond])} of {planned_seeds} seeds")
        for name, r in runs[cond]:
            per_seed.append({"run": name, **{f"gold_{p}": round(100 * vec(r, p).mean(), 2) for p in POS}})
    per_seed.insert(0, {"run": "frozen", **{f"gold_{p}": round(100 * vec(frozen, p).mean(), 2) for p in POS}})
    for r in per_seed:
        r["gap"] = round(r["gold_first"] - r["gold_last"], 2)

    final = [{"model": "Paper P20 Qwen (authors' reported)", "status": "reported in paper",
              **{f"gold_{p}": round(100 * paper[p]["acc"], 2) for p in POS}}]
    for name, V in models.items():
        final.append({"model": name, "status": status[name], **{f"gold_{p}": round(100 * V[p].mean(), 2) for p in POS}})
    for r in final:
        r["first_minus_last_gap"] = round(r["gold_first"] - r["gold_last"], 2)

    rng = np.random.Generator(np.random.PCG64(SEED))
    idx = rng.integers(0, n, size=(B, n))
    contrasts = []

    def add(label, fn):
        pt = fn(np.arange(n))
        s = np.array([fn(i) for i in idx])
        lo, hi = np.percentile(s, [2.5, 97.5])
        contrasts.append({"contrast": label, "estimate_pp": round(100 * pt, 2), "ci95_low_pp": round(100 * lo, 2),
                          "ci95_high_pp": round(100 * hi, 2), "excludes_zero": bool(lo > 0 or hi < 0), "n_questions": n})

    gap = lambda V, i: V["first"][i].mean() - V["last"][i].mean()
    for m, V in models.items():
        add(f"{m}: gap (first - last)", lambda i, V=V: gap(V, i))
    F = models["Frozen Qwen"]
    for m, V in models.items():
        if m == "Frozen Qwen":
            continue
        add(f"{m} vs Frozen: change in gap", lambda i, V=V: gap(V, i) - gap(F, i))
        for p in POS:
            add(f"{m} vs Frozen: accuracy at gold {p}", lambda i, V=V, p=p: V[p][i].mean() - F[p][i].mean())
    if "Standard LoRA SFT" in models and "Balanced LoRA SFT" in models:
        S, Bm = models["Standard LoRA SFT"], models["Balanced LoRA SFT"]
        add("Balanced vs Standard: change in gap", lambda i: gap(Bm, i) - gap(S, i))
        for p in POS:
            add(f"Balanced vs Standard: accuracy at gold {p}", lambda i, p=p: Bm[p][i].mean() - S[p][i].mean())

    out_dir.mkdir(parents=True, exist_ok=True)
    for fname, rows in (("evqa_final_comparison.csv", final), ("evqa_per_seed.csv", per_seed),
                        ("evqa_bootstrap_contrasts.csv", contrasts)):
        with (out_dir / fname).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    md = [f"# E-VQA transfer (generated {time.strftime('%Y-%m-%d %H:%M')})", "",
          f"Encyclopedic-VQA, authors' P20 protocol (Qwen part), k = 10, n = {n}, canonical one-way matcher. "
          "Adapters were trained on InfoSeek only.", "",
          "| Model | Gold first | Gold middle | Gold last | First-last gap | Status |", "|---|---|---|---|---|---|"]
    md += [f"| {r['model']} | {r['gold_first']:.2f} | {r['gold_middle']:.2f} | {r['gold_last']:.2f} | "
           f"{r['first_minus_last_gap']:.2f} | {r['status']} |" for r in final]
    md += ["", "## Paired bootstrap (B = 10,000, seed 42), percentage points", "",
           "| Contrast | Estimate | 95% CI | CI excludes 0 |", "|---|---|---|---|"]
    md += [f"| {c['contrast']} | {c['estimate_pp']:+.2f} | [{c['ci95_low_pp']:+.2f}, {c['ci95_high_pp']:+.2f}] | "
           f"{'yes' if c['excludes_zero'] else 'no'} |" for c in contrasts]
    (out_dir / "COMPARISON_EVQA.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--authors-dir", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True, help="runs/evqa")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--run", metavar="NAME")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--out", type=Path, help="results dir for --compare")
    ap.add_argument("--planned-seeds", type=int, default=2)
    a = ap.parse_args()
    if a.check:
        return cmd_check(a.authors_dir, a.root)
    if a.run:
        return cmd_run(a.authors_dir, a.root, a.run, a.adapter)
    if a.compare:
        return cmd_compare(a.authors_dir, a.root, a.out, a.planned_seeds)
    ap.error("choose --check, --run or --compare")
    return 2


if __name__ == "__main__":
    sys.exit(main())
