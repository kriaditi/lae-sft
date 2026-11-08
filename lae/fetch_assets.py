"""Download everything the InfoSeek reproduction needs into HF_HOME, and extract
only the query images of the evaluation subset.

Uses the authors' own download helpers, so files land exactly where their
runners look for them. Safe to rerun: Hugging Face skips finished downloads and
images already on disk are not extracted again.
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
import time
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

MODELS = {
    "Qwen/Qwen2.5-VL-7B-Instruct": None,                           # full weights
    "LinWeizheDragon/PreFLMR_ViT-G": None,                          # full weights
    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k": ["*.json", "*.txt"],  # image processor config only
}


def log(msg: str) -> None:
    print(f"[fetch {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch_models() -> None:
    for repo, patterns in MODELS.items():
        log(f"model {repo}")
        path = snapshot_download(repo, allow_patterns=patterns)
        log(f"  -> {path}")


def fetch_infoseek(authors_dir: Path) -> None:
    sys.path.insert(0, str(authors_dir / "src"))
    from tts_kbvqa.data import infoseek as ds  # authors' module

    val_pq = ds.download_val_parquet()
    log(f"InfoSeek val (M2KR test) parquet -> {val_pq}")
    kb_pq = ds.download_passages_parquet()
    n_kb = pq.ParquetFile(kb_pq).metadata.num_rows
    log(f"InfoSeek KB parquet -> {kb_pq}  ({n_kb} passages)")

    subset = json.loads((authors_dir / "results/p11_7b_subset/subset500_v2_qids.json").read_text())["qids"]
    rows = pq.read_table(val_pq, columns=["question_id", "img_path"]).to_pylist()
    by_qid = {r["question_id"]: r["img_path"] for r in rows}
    missing_q = [q for q in subset if q not in by_qid]
    if missing_q:
        raise SystemExit(f"ERROR: {len(missing_q)} subset qids not in val parquet, e.g. {missing_q[:3]}")

    image_dir = authors_dir / "data/infoseek/images/val"
    unique = sorted({by_qid[q] for q in subset})  # several questions can share one image
    wanted = {}  # basename -> relative img_path
    for rel in unique:
        if not (image_dir / rel).exists():
            wanted[Path(rel).name] = rel
    log(f"images: {len(subset)} questions use {len(unique)} unique images; "
        f"{len(unique) - len(wanted)} already on disk, {len(wanted)} to extract")
    if not wanted:
        verify_images(subset, by_qid, image_dir)
        return

    tar_path = ds.download_val_images_tar()
    log(f"images tar -> {tar_path} ({Path(tar_path).stat().st_size / 1e9:.1f} GB); extracting {len(wanted)}")
    with tarfile.open(tar_path, mode="r|*") as tf:
        for m in tf:
            if not m.isfile():
                continue
            rel = wanted.get(Path(m.name).name)
            if rel is None:
                continue
            dst = image_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(m)
            dst.write_bytes(src.read())
            del wanted[Path(m.name).name]
            if not wanted:
                break
    if wanted:
        raise SystemExit(f"ERROR: {len(wanted)} images not found in tar, e.g. {list(wanted.values())[:3]}")
    log("images: all subset images extracted")
    verify_images(subset, by_qid, image_dir)


def verify_images(subset: list[str], by_qid: dict[str, str], image_dir: Path) -> None:
    """Every evaluation question must have a non-empty image file."""
    bad = [q for q in subset if not (image_dir / by_qid[q]).is_file()
           or (image_dir / by_qid[q]).stat().st_size == 0]
    if bad:
        raise SystemExit(f"ERROR: {len(bad)} questions have no image file, e.g. {bad[:3]}")
    log(f"images: verified, all {len(subset)} questions have their image file")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--authors-dir", required=True)
    ap.add_argument("--what", default="models,infoseek")
    args = ap.parse_args()
    what = set(args.what.split(","))
    if "models" in what:
        fetch_models()
    if "infoseek" in what:
        fetch_infoseek(Path(args.authors_dir))
    log("done")


if __name__ == "__main__":
    main()
