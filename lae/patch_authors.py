"""Minimal, recorded patches to the authors' P17 runner. Nothing that changes
retrieval, gold selection, prompts, decoding or scoring is touched.

  1. Output directory comes from $P17_OUT, so our run never overwrites the
     authors' committed results/p17 (the reference numbers we compare against).
  2. The subset-size assert accepts $P17_N_EXPECTED (496 for the recovered
     subset; see recover_subset.py) instead of a hard-coded 500.

Idempotent: each patch is applied only if not already present, and the script
fails if the original text is not found exactly once. Stdlib only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PATCHES = [
    ("import math\n",
     "import math\nimport os  # [lae patch]\n"),
    ('OUT = REPO_ROOT / "results" / "p17"\n',
     'OUT = Path(os.environ.get("P17_OUT", str(REPO_ROOT / "results" / "p17")))  # [lae patch]\n'),
    ('    assert len(qids) == 500, f"expected 500 qids, got {len(qids)}"\n',
     '    _n_exp = int(os.environ.get("P17_N_EXPECTED", "500"))  # [lae patch]\n'
     '    assert len(qids) == _n_exp, f"expected {_n_exp} qids, got {len(qids)}"\n'),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--authors-dir", required=True)
    args = ap.parse_args()
    f = Path(args.authors_dir) / "scripts/run_p17_reader_bottleneck.py"
    text = f.read_text()
    changed = False
    for old, new in PATCHES:
        if new in text:
            continue
        n = text.count(old)
        if n != 1:
            print(f"ERROR: expected 1 occurrence in {f.name}, found {n}: {old.strip()[:70]}", file=sys.stderr)
            return 1
        text = text.replace(old, new)
        changed = True
    if changed:
        f.write_text(text)
        print(f"[patch] patched {f}")
    else:
        print(f"[patch] already patched: {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
