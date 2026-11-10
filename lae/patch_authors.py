"""Minimal, recorded patches to the authors' code. Nothing that changes retrieval,
gold selection, prompts, decoding or scoring is touched.

scripts/run_p17_reader_bottleneck.py
  1. Output directory comes from $P17_OUT, so our runs never overwrite the
     authors' committed results/p17 (the reference numbers we compare against).
  2. The subset-size assert accepts $P17_N_EXPECTED (496 for the recovered
     subset; see recover_subset.py) instead of a hard-coded 500.
  3. The run summary records which LoRA adapter (if any) was evaluated.

src/tts_kbvqa/reader/qwen.py
  4. If $LAE_ADAPTER is set, the reader loads that LoRA adapter on top of the
     frozen model and merges it (PEFT merge_and_unload). Unset = the authors'
     frozen reader, unchanged. Same processor, prompt and greedy decoding either way.

Idempotent: each patch is applied only if not already present, and the script
fails if the original text is not found exactly once. Stdlib only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PATCHES = {
    "scripts/run_p17_reader_bottleneck.py": [
        ("import math\n",
         "import math\nimport os  # [lae patch]\n"),
        ('OUT = REPO_ROOT / "results" / "p17"\n',
         'OUT = Path(os.environ.get("P17_OUT", str(REPO_ROOT / "results" / "p17")))  # [lae patch]\n'),
        ('    assert len(qids) == 500, f"expected 500 qids, got {len(qids)}"\n',
         '    _n_exp = int(os.environ.get("P17_N_EXPECTED", "500"))  # [lae patch]\n'
         '    assert len(qids) == _n_exp, f"expected {_n_exp} qids, got {len(qids)}"\n'),
        ('        "model": MODEL_7B, "device": device_name,',
         '        "model": MODEL_7B, "lae_adapter": os.environ.get("LAE_ADAPTER", ""),  # [lae patch]\n'
         '        "device": device_name,'),
    ],
    "src/tts_kbvqa/reader/qwen.py": [
        ("import gc\n",
         "import gc\nimport os  # [lae patch]\n"),
        ("            .to(device)\n            .eval()\n        )\n",
         "            .to(device)\n            .eval()\n        )\n"
         "        _adapter = os.environ.get(\"LAE_ADAPTER\")  # [lae patch] evaluate a LoRA SFT adapter\n"
         "        if _adapter:\n"
         "            from peft import PeftModel\n"
         "            self.model = PeftModel.from_pretrained(self.model, _adapter).merge_and_unload().eval()\n"
         "            print(f\"[lae] loaded and merged LoRA adapter {_adapter}\", flush=True)\n"),
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--authors-dir", required=True)
    args = ap.parse_args()
    for rel, patches in PATCHES.items():
        f = Path(args.authors_dir) / rel
        text = f.read_text()
        changed = False
        for old, new in patches:
            if new in text:
                continue
            n = text.count(old)
            if n != 1:
                print(f"ERROR: expected 1 occurrence in {rel}, found {n}: {old.strip()[:70]}", file=sys.stderr)
                return 1
            text = text.replace(old, new)
            changed = True
        if changed:
            f.write_text(text)
            print(f"[patch] patched {rel}")
        else:
            print(f"[patch] already patched: {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
