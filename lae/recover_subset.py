"""Recover the authors' InfoSeek evaluation subset (subset500_v2).

The authors' runners read results/p11_7b_subset/subset500_v2_qids.json, which is
not in their public repo. Their P22 and P23 summaries (same reader, same subset,
same exclusion rule as P17) store the 496 question ids they actually evaluated.
P17 reports on exactly these 496 (500 drawn, 4 excluded for having fewer than
19 distractors). We rebuild the file from those ids and record where it came from.

Stdlib only, so it runs on the login node before the environment exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--authors-dir", required=True)
    ap.add_argument("--expected", type=int, default=496)
    args = ap.parse_args()

    root = Path(args.authors_dir)
    src = [root / "results/p22/per_cell_summary.json", root / "results/p23/per_cell_summary.json"]
    lists = []
    for p in src:
        if not p.exists():
            print(f"ERROR: {p} not found", file=sys.stderr)
            return 1
        lists.append(json.loads(p.read_text())["qids"])

    if lists[0] != lists[1]:
        print("ERROR: P22 and P23 qid lists differ; cannot recover a single subset", file=sys.stderr)
        return 1
    qids = lists[0]
    if len(qids) != args.expected or len(set(qids)) != len(qids):
        print(f"ERROR: expected {args.expected} unique qids, got {len(qids)} ({len(set(qids))} unique)",
              file=sys.stderr)
        return 1

    # P17's own summary must report the same n.
    p17 = json.loads((root / "results/p17/per_cell_summary.json").read_text())
    if p17["n_kept_primary"] != len(qids):
        print(f"ERROR: P17 reports n={p17['n_kept_primary']}, recovered {len(qids)}", file=sys.stderr)
        return 1

    out = root / "results/p11_7b_subset/subset500_v2_qids.json"
    payload = {
        "_rule": "RECOVERED: the 496 kept qids of subset500_v2, read from the authors' committed "
                 "results/p22 and results/p23 per_cell_summary.json (identical lists). The 4 qids "
                 "P17 excluded for short distractors are not recoverable and not needed.",
        "_source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in src},
        "qids": qids,
    }
    text = json.dumps(payload, indent=2) + "\n"
    if out.exists():
        if out.read_text() == text:
            print(f"[recover_subset] already present and identical: {out}")
            return 0
        print(f"ERROR: {out} exists with different content; not overwriting", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"[recover_subset] wrote {len(qids)} qids -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
