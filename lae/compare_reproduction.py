"""Compare our P17 run with the authors' committed P17 numbers.

Writes one CSV row per cell (13 cells) plus first-minus-last gap rows per k.
Only measured numbers are written; nothing is estimated. Paired tests are not
possible because the authors did not release per-question predictions, so the
check is: same n, same fallback count, and our accuracy inside the paper's
Wilson 95% CI (and vice versa).

Never overwrites an existing different CSV: writes <name>_<timestamp>.csv instead.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from pathlib import Path

LABEL = ("exact_reproduction: authors' code @ pinned commit, unmodified protocol; "
         "eval set = recovered 496-qid subset500_v2; patches = output dir + n assert only")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paper = json.loads(Path(args.paper).read_text())
    ours = json.loads(Path(args.ours).read_text())

    checks = {
        "n_kept_primary": (paper["n_kept_primary"], ours["n_kept_primary"]),
        "n_fallback": (paper["n_fallback"], ours["n_fallback"]),
    }
    print(f"[compare] paper: {paper['device']} torch {paper['torch_version']}")
    print(f"[compare] ours : {ours['device']} torch {ours['torch_version']}")
    for k, (p, o) in checks.items():
        print(f"[compare] {k:15s} paper={p}  ours={o}  {'OK' if p == o else 'MISMATCH'}")

    rows = []
    for split in ("primary", "sensitivity"):
        pc, oc = paper[split], ours[split]
        if set(pc) != set(oc):
            print(f"ERROR: cell sets differ in {split}", file=sys.stderr)
            return 1
        for name in pc:
            p, o = pc[name], oc[name]
            rows.append({
                "split": split, "cell": name, "k": p["k"], "position": p["position"] or "-",
                "gold_idx": p["gold_idx"],
                "n_paper": p["n"], "n_ours": o["n"],
                "acc_paper": round(100 * p["acc"], 2), "acc_ours": round(100 * o["acc"], 2),
                "diff_pp": round(100 * (o["acc"] - p["acc"]), 2),
                "ci_paper": f"[{100*p['ci_low']:.2f}, {100*p['ci_high']:.2f}]",
                "ci_ours": f"[{100*o['ci_low']:.2f}, {100*o['ci_high']:.2f}]",
                "ours_in_paper_ci": p["ci_low"] <= o["acc"] <= p["ci_high"],
                "paper_in_ours_ci": o["ci_low"] <= p["acc"] <= o["ci_high"],
            })
        for k in sorted({c["k"] for c in pc.values() if c["k"] > 1}):
            f, l = f"k{k}_first", f"k{k}_last"
            gp = 100 * (pc[f]["acc"] - pc[l]["acc"])
            go = 100 * (oc[f]["acc"] - oc[l]["acc"])
            rows.append({
                "split": split, "cell": f"k{k}_gap_first_minus_last", "k": k, "position": "gap",
                "gold_idx": "-", "n_paper": pc[f]["n"], "n_ours": oc[f]["n"],
                "acc_paper": round(gp, 2), "acc_ours": round(go, 2), "diff_pp": round(go - gp, 2),
                "ci_paper": "-", "ci_ours": "-", "ours_in_paper_ci": "-", "paper_in_ours_ci": "-",
            })

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]) + ["label"])
    w.writeheader()
    for r in rows:
        w.writerow({**r, "label": LABEL})
    text = buf.getvalue()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.read_text() != text:
        out = out.with_name(f"{out.stem}_{time.strftime('%Y%m%d_%H%M%S')}{out.suffix}")
        print(f"[compare] {args.out} exists with different content; writing {out}")
    out.write_text(text)

    print(f"\n[compare] k=10, primary (n={ours['n_kept_primary']}):")
    for r in rows:
        if r["split"] == "primary" and r["k"] == 10:
            print(f"   {r['cell']:26s} paper {r['acc_paper']:6.2f}  ours {r['acc_ours']:6.2f}  "
                  f"diff {r['diff_pp']:+6.2f} pp  in paper CI: {r['ours_in_paper_ci']}")
    prim = [r for r in rows if r["split"] == "primary" and r["position"] != "gap"]
    n_in = sum(1 for r in prim if r["ours_in_paper_ci"])
    print(f"[compare] {n_in}/{len(prim)} primary cells inside the paper's 95% CI")
    print(f"[compare] wrote {out}")
    return 0 if all(p == o for p, o in checks.values()) else 2


if __name__ == "__main__":
    sys.exit(main())
