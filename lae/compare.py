"""Stage 4: comparison tables, paired-bootstrap CIs, costs and plots.

Inputs
  --paper       authors' committed results/p17/per_cell_summary.json (reported numbers only)
  --frozen      our reproduction run dir (frozen Qwen2.5-VL-7B; predictions_<cell>.jsonl)
  --eval-root   runs/eval/<condition>_seed<S>/ (one per trained adapter)
  --train-root  runs/train/<condition>_seed<S>/ (train_config.json, train_metrics.json)

Conventions (same as the authors): primary analysis on all 496 questions; official
InfoSeek correctness (correct_official); paired bootstrap over questions, B=10,000,
PCG64 seed 42, percentile 95% CI. A condition with fewer seeds than planned is
labelled PRELIMINARY. Seed-mean models average each question's correctness over seeds
before bootstrapping. Nothing is estimated: a missing run is reported as missing.

Outputs (in --out): final_comparison.csv, per_seed.csv, bootstrap_contrasts.csv,
costs.csv, plots/position_k10.png, plots/acc_vs_k.png, COMPARISON.md
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

import numpy as np

B, SEED = 10_000, 42
POS = ("first", "middle", "last")
KS = (1, 3, 5, 10, 20)
COND_LABEL = {"standard": "Standard LoRA SFT", "balanced": "Balanced LoRA SFT"}


def load_run(run_dir: Path) -> dict:
    """{cell: {qid: correct(0/1)}} plus latency/token arrays and the summary."""
    cells, lat, toks = {}, [], []
    for p in sorted(run_dir.glob("predictions_*.jsonl")):
        cell = p.stem[len("predictions_"):]
        d = {}
        for line in p.open():
            if line.strip():
                r = json.loads(line)
                d[r["qid"]] = 1.0 if r["correct_official"] else 0.0
                if cell.startswith("k10_"):
                    lat.append(r["reader_latency_s"]); toks.append(r["input_tokens"])
        cells[cell] = d
    summ = json.loads((run_dir / "per_cell_summary.json").read_text())
    return {"cells": cells, "latency": lat, "tokens": toks, "summary": summ}


def vec(model_cells: dict, cell: str, qids: list[str]) -> np.ndarray:
    return np.array([model_cells[cell][q] for q in qids], dtype=np.float64)


def seed_mean(runs: list[dict], qids: list[str]) -> dict:
    out = {}
    for cell in runs[0]["cells"]:
        out[cell] = dict(zip(qids, np.mean([vec(r["cells"], cell, qids) for r in runs], axis=0)))
    return out


def boot(values_fn, n: int, idx: np.ndarray) -> tuple[float, float, float]:
    point = values_fn(np.arange(n))
    samples = np.array([values_fn(i) for i in idx])
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return point, float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", type=Path, required=True)
    ap.add_argument("--frozen", type=Path, required=True)
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--train-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--planned-seeds", type=int, default=2)
    ap.add_argument("--isolated", type=Path, default=None, help="runs/isolated (optional)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    frozen = load_run(args.frozen)
    qids = sorted(frozen["cells"]["k10_first"])
    n = len(qids)

    # trained runs: <condition>_seed<S>, condition "standard" or "balanced_s<S>"
    runs: dict[str, list[tuple[int, str, dict]]] = {"standard": [], "balanced": []}
    for d in sorted(args.eval_root.glob("*_seed*")):
        m = re.fullmatch(r"(standard|balanced_s\d+)_seed(\d+)", d.name)
        if not m or not (d / "per_cell_summary.json").exists():
            continue
        run = load_run(d)
        for cell, dd in run["cells"].items():
            if set(dd) != set(qids):
                raise SystemExit(f"{d.name}/{cell}: question set differs from the frozen run")
        cond = "standard" if m.group(1) == "standard" else "balanced"
        runs[cond].append((int(m.group(2)), d.name, run))

    # --- per-seed table
    per_seed = []

    def row(model, run_name, cells, status):
        r = {"model": model, "run": run_name, "status": status, "n": n}
        for k in KS:
            for pos in (("-",) if k == 1 else POS):
                cell = "k1" if k == 1 else f"k{k}_{pos}"
                r[f"{cell}"] = round(100 * np.mean([cells[cell][q] for q in qids]), 2)
            if k > 1:
                r[f"k{k}_gap"] = round(r[f"k{k}_first"] - r[f"k{k}_last"], 2)
        return r

    per_seed.append(row("Frozen Qwen2.5-VL-7B", "p17_repro", frozen["cells"], "measured"))
    for cond in ("standard", "balanced"):
        for seed, name, run in sorted(runs[cond]):
            per_seed.append(row(COND_LABEL[cond], name, run["cells"], "measured"))

    # --- final comparison (k=10), seed means
    paper = json.loads(args.paper.read_text())["primary"]
    final = [{"model": "Paper reproduction (authors' reported)", "seeds": "-", "status": "reported in paper",
              **{f"gold_{p}": round(100 * paper[f"k10_{p}"]["acc"], 2) for p in POS}}]
    final[0]["first_minus_last_gap"] = round(final[0]["gold_first"] - final[0]["gold_last"], 2)
    models = {"Frozen Qwen": frozen["cells"]}
    status = {"Frozen Qwen": "measured (1 run, greedy decoding)"}
    for cond in ("standard", "balanced"):
        rs = [r for _, _, r in sorted(runs[cond])]
        if rs:
            models[COND_LABEL[cond]] = seed_mean(rs, qids)
            status[COND_LABEL[cond]] = (f"mean of {len(rs)} seeds" if len(rs) >= args.planned_seeds
                                        else f"PRELIMINARY: {len(rs)} of {args.planned_seeds} seeds")
    iso = None
    if args.isolated and (args.isolated / "predictions.jsonl").exists():
        rows_iso = {json.loads(l)["qid"]: json.loads(l) for l in (args.isolated / "predictions.jsonl").open() if l.strip()}
        if set(rows_iso) != set(qids):
            raise SystemExit("isolated reading: question set differs from the frozen run")
        iso = json.loads((args.isolated / "summary.json").read_text())
        c = {q: (1.0 if rows_iso[q]["correct_official"] else 0.0) for q in qids}
        name_iso = "Isolated reading (frozen, mean log-prob)"
        models[name_iso] = {f"k10_{p}": c for p in POS}   # order-invariant: same answers at every position
        status[name_iso] = "measured (order-invariant by construction)"
    for name, cells in models.items():
        r = {"model": name, "seeds": status[name], "status": status[name]}
        for p in POS:
            r[f"gold_{p}"] = round(100 * np.mean([cells[f"k10_{p}"][q] for q in qids]), 2)
        r["first_minus_last_gap"] = round(r["gold_first"] - r["gold_last"], 2)
        final.append(r)

    # --- paired bootstrap contrasts at k=10
    rng = np.random.Generator(np.random.PCG64(SEED))
    idx = rng.integers(0, n, size=(B, n))
    V = {m: {p: vec(c, f"k10_{p}", qids) for p in POS} for m, c in models.items()}
    contrasts = []

    def add(label, fn):
        pt, lo, hi = boot(fn, n, idx)
        contrasts.append({"contrast": label, "estimate_pp": round(100 * pt, 2),
                          "ci95_low_pp": round(100 * lo, 2), "ci95_high_pp": round(100 * hi, 2),
                          "excludes_zero": bool(lo > 0 or hi < 0), "n_questions": n})

    for m in models:
        add(f"{m}: gap (first - last)", lambda i, m=m: V[m]["first"][i].mean() - V[m]["last"][i].mean())
    for m in [x for x in models if x != "Frozen Qwen"]:
        add(f"{m} vs Frozen: change in gap",
            lambda i, m=m: (V[m]["first"][i].mean() - V[m]["last"][i].mean())
            - (V["Frozen Qwen"]["first"][i].mean() - V["Frozen Qwen"]["last"][i].mean()))
        for p in POS:
            add(f"{m} vs Frozen: accuracy at gold {p}",
                lambda i, m=m, p=p: V[m][p][i].mean() - V["Frozen Qwen"][p][i].mean())
    s, b = COND_LABEL["standard"], COND_LABEL["balanced"]
    iso_name = "Isolated reading (frozen, mean log-prob)"
    if iso_name in models:
        for m in (s, b):
            if m in models:
                for p in POS:
                    add(f"{m} vs {iso_name}: accuracy at gold {p}",
                        lambda i, m=m, p=p: V[m][p][i].mean() - V[iso_name][p][i].mean())
    if s in models and b in models:
        add(f"{b} vs {s}: change in gap",
            lambda i: (V[b]["first"][i].mean() - V[b]["last"][i].mean())
            - (V[s]["first"][i].mean() - V[s]["last"][i].mean()))
        for p in POS:
            add(f"{b} vs {s}: accuracy at gold {p}", lambda i, p=p: V[b][p][i].mean() - V[s][p][i].mean())

    # --- costs
    costs = [{"run": "p17_repro (frozen)", "train_hours": 0, "train_s_per_step": "-", "train_peak_gpu_gb": "-",
              "infer_mean_latency_s_k10": round(float(np.mean(frozen["latency"])), 3) if frozen["latency"] else "-",
              "infer_mean_input_tokens_k10": round(float(np.mean(frozen["tokens"])), 1) if frozen["tokens"] else "-",
              "eval_peak_gpu_gb": frozen["summary"].get("peak_alloc_gb"),
              "eval_wall_min": round(frozen["summary"].get("wall_clock_s", 0) / 60, 1)}]
    for cond in ("standard", "balanced"):
        for seed, name, run in sorted(runs[cond]):
            tm_path = args.train_root / name / "train_metrics.json"
            tm = json.loads(tm_path.read_text()) if tm_path.exists() else {}
            costs.append({
                "run": name,
                "train_hours": round(tm.get("train_wall_s_this_session", 0) / 3600, 2)
                if tm and not tm.get("resumed") else ("resumed: see logs" if tm else "-"),
                "train_s_per_step": tm.get("s_per_step", "-"),
                "train_peak_gpu_gb": tm.get("peak_gpu_mem_gb", "-"),
                "infer_mean_latency_s_k10": round(float(np.mean(run["latency"])), 3) if run["latency"] else "-",
                "infer_mean_input_tokens_k10": round(float(np.mean(run["tokens"])), 1) if run["tokens"] else "-",
                "eval_peak_gpu_gb": run["summary"].get("peak_alloc_gb"),
                "eval_wall_min": round(run["summary"].get("wall_clock_s", 0) / 60, 1)})

    if iso:
        costs.append({"run": "isolated reading (frozen, 10 reads per question)", "train_hours": 0,
                      "train_s_per_step": "-", "train_peak_gpu_gb": "-",
                      "infer_mean_latency_s_k10": round(iso["mean_latency_s_per_question"], 3),
                      "infer_mean_input_tokens_k10": round(iso["mean_input_tokens_per_question"], 1),
                      "eval_peak_gpu_gb": iso.get("peak_alloc_gb"),
                      "eval_wall_min": round(iso.get("wall_s_this_session", 0) / 60, 1)})

    def write(name, rows):
        if not rows:
            return
        keys = list(dict.fromkeys(k for r in rows for k in r))
        p = args.out / name
        with p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    write("final_comparison.csv", final)
    write("per_seed.csv", per_seed)
    write("bootstrap_contrasts.csv", contrasts)
    write("costs.csv", costs)

    # --- plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (args.out / "plots").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for r in final:
        ax.plot(range(3), [r[f"gold_{p}"] for p in POS], marker="o", label=f"{r['model']} ({r['status']})")
    ax.set_xticks(range(3), ["gold first", "gold middle", "gold last"])
    ax.set_ylabel("InfoSeek accuracy (%)")
    ax.set_title(f"k = 10, n = {n} questions")
    ax.grid(alpha=.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out / "plots/position_k10.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, p in zip(axes, POS):
        for r in per_seed:
            ys = [r["k1"]] + [r[f"k{k}_{p}"] for k in KS[1:]]
            ax.plot(KS, ys, marker="o", label=f"{r['model']} ({r['run']})")
        ax.set_title(f"gold {p}")
        ax.set_xlabel("k (passages in context)")
        ax.set_xticks(KS)
        ax.grid(alpha=.3)
    axes[0].set_ylabel("InfoSeek accuracy (%)")
    axes[-1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(args.out / "plots/acc_vs_k.png", dpi=200)
    plt.close(fig)

    # --- markdown summary
    md = [f"# Comparison (generated {time.strftime('%Y-%m-%d %H:%M')})", "",
          f"InfoSeek, authors' P17 protocol, k = 10, n = {n}. Accuracy in %. Gap = gold first - gold last.", "",
          "| Model | Gold first | Gold middle | Gold last | First-last gap | Status |", "|---|---|---|---|---|---|"]
    for r in final:
        md.append(f"| {r['model']} | {r['gold_first']:.2f} | {r['gold_middle']:.2f} | {r['gold_last']:.2f} "
                  f"| {r['first_minus_last_gap']:.2f} | {r['status']} |")
    md += ["", "## Paired bootstrap (B = 10,000, seed 42), percentage points", "",
           "| Contrast | Estimate | 95% CI | CI excludes 0 |", "|---|---|---|---|"]
    for c in contrasts:
        md.append(f"| {c['contrast']} | {c['estimate_pp']:+.2f} | [{c['ci95_low_pp']:+.2f}, {c['ci95_high_pp']:+.2f}] "
                  f"| {'yes' if c['excludes_zero'] else 'no'} |")
    (args.out / "COMPARISON.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
