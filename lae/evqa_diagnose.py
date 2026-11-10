"""Diagnose the E-VQA accuracy drop of the InfoSeek-trained adapters (CPU only).

Reads the saved E-VQA predictions (runs/evqa/<model>/predictions_qwen_<pos>.jsonl)
and measures, per model and position:
  * canonical accuracy (authors' headline: normalized gold contained in prediction)
    and the any-alternative variant the authors also store
  * prediction length (words, generated tokens)
  * among wrong predictions: how often the normalized prediction is a non-empty
    substring of the normalized gold ("shorter but contained"), and exact matches
  * canonical accuracy by gold-answer length (1, 2-3, 4-7, 8+ words), gold first
  * 10 examples where the frozen model is right and standard_seed0 is wrong (gold first)
Nothing here changes any score; it only describes the saved predictions.
Output: printed and saved to <out>/evqa_diagnosis.md and evqa_diagnosis.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

POS = ("first", "middle", "last")
BUCKETS = (("1", 1, 1), ("2-3", 2, 3), ("4-7", 4, 7), ("8+", 8, 10**9))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--authors-dir", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True, help="runs/evqa")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).parent))
    from evqa_transfer import load_p20
    p20 = load_p20(a.authors_dir)
    norm = p20.normalize_answer
    questions = {s.question_id: s.question for s in p20.evqa_data.load_test_samples(p20.IMAGE_DIR)}

    models = [d.name for d in sorted(a.root.iterdir()) if (d / "predictions_qwen_first.jsonl").exists()]
    models = ["frozen"] + [m for m in models if m != "frozen"]
    data = {m: {p: [json.loads(l) for l in (a.root / m / f"predictions_qwen_{p}.jsonl").open() if l.strip()]
                for p in POS} for m in models}

    stats, md = {}, ["# E-VQA diagnosis", "",
                     "Saved predictions only; scores unchanged. Canonical = authors' one-way matcher.", "",
                     "| Model | Position | Canonical acc | Any-alternative acc | Mean pred words | Mean gen tokens "
                     "| Wrong but pred is substring of gold | Exact match |",
                     "|---|---|---|---|---|---|---|---|"]
    for m in models:
        for p in POS:
            rows = data[m][p]
            n = len(rows)
            wrong = [r for r in rows if not r["correct_canonical"]]
            sub = sum(1 for r in wrong if norm(r["pred"]) and norm(r["pred"]) in norm(r["gold_canonical"]))
            exact = sum(1 for r in rows if norm(r["pred"]) == norm(r["gold_canonical"]))
            s = {"n": n, "acc": sum(r["correct_canonical"] for r in rows) / n,
                 "acc_any_alt": sum(r["correct_any_alternative"] for r in rows) / n,
                 "pred_words": sum(len(r["pred"].split()) for r in rows) / n,
                 "gen_tokens": sum(r["gen_tokens"] for r in rows) / n,
                 "wrong_pred_substring_of_gold": sub / n, "exact_match": exact / n}
            stats[f"{m}/{p}"] = s
            md.append(f"| {m} | {p} | {100 * s['acc']:.1f} | {100 * s['acc_any_alt']:.1f} | {s['pred_words']:.1f} "
                      f"| {s['gen_tokens']:.1f} | {100 * s['wrong_pred_substring_of_gold']:.1f} | {100 * s['exact_match']:.1f} |")

    md += ["", "## Canonical accuracy by gold-answer length (gold first)", "",
           "| Gold words | n | " + " | ".join(models) + " |", "|---|---|" + "---|" * len(models)]
    for label, lo, hi in BUCKETS:
        qs = [r["qid"] for r in data["frozen"]["first"] if lo <= len(r["gold_canonical"].split()) <= hi]
        cells = []
        for m in models:
            by = {r["qid"]: r for r in data[m]["first"]}
            cells.append(f"{100 * sum(by[q]['correct_canonical'] for q in qs) / max(len(qs), 1):.1f}")
        stats[f"by_gold_len/{label}"] = {"n": len(qs), **dict(zip(models, cells))}
        md.append(f"| {label} | {len(qs)} | " + " | ".join(cells) + " |")

    if "standard_seed0" in data:
        fz = {r["qid"]: r for r in data["frozen"]["first"]}
        st = {r["qid"]: r for r in data["standard_seed0"]["first"]}
        ex = [q for q in fz if fz[q]["correct_canonical"] and not st[q]["correct_canonical"]][:10]
        md += ["", "## Examples: frozen right, standard_seed0 wrong (gold first)", "",
               "| Question | Gold | Frozen prediction | Standard s0 prediction |", "|---|---|---|---|"]
        for q in ex:
            cl = lambda t: t.replace("|", "/").replace("\n", " ")[:120]
            md.append(f"| {cl(questions.get(q, ''))} | {cl(fz[q]['gold_canonical'])} | {cl(fz[q]['pred'])} "
                      f"| {cl(st[q]['pred'])} |")

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "evqa_diagnosis.md").write_text("\n".join(md) + "\n")
    (a.out / "evqa_diagnosis.json").write_text(json.dumps(stats, indent=2))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
