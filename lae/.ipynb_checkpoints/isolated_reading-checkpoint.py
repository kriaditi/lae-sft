"""Isolated-reading baseline (training-free, frozen Qwen2.5-VL-7B).

For each of the 496 evaluation questions, the reader sees each of the SAME 10
passages used in the k=10 cells (gold + the 9 P17 distractors) on its own, with
the authors' prompt and greedy decoding (exactly their k=1 prompt format), and
answers once per passage. The final answer is the one with the highest
confidence. Order cannot matter by construction, so gold first / middle / last
all receive the same score.

Confidence (declared before seeing results):
  primary   = mean log-probability per generated token (incl. the end token)
  secondary = total log-probability of the generated tokens
Ties: the earlier passage in the k10_first order wins.

Hard check: the answer for the gold passage alone must match our reproduction's
k=1 cell (same prompt, same passage, greedy) for at least 99% of questions.

Resumable: per (question, passage) results are cached in <out>/cache.jsonl.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch


def log(msg: str) -> None:
    print(f"[isolated {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wilson(h: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p, z2 = h / n, z * z
    c = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n) / (1 + z2 / n)
    return c - half, c + half


@torch.no_grad()
def generate_scored(model, tokenizer, inputs: dict, input_len: int, max_new: int) -> dict:
    """Greedy answer (same call as the authors' reader) plus the model's raw log-probability
    of each generated token. Raw logits are used, not the processed scores, because the
    model's default generation config applies a repetition penalty to the scores."""
    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, temperature=1.0,
                         output_logits=True, return_dict_in_generate=True)
    gen = out.sequences[0, input_len:]
    logits = torch.stack(out.logits, dim=1)[0].float()          # [n_gen, vocab]
    lp = torch.log_softmax(logits, dim=-1).gather(1, gen[:, None])[:, 0]
    eos = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list)
              else [model.generation_config.eos_token_id])
    pad = tokenizer.pad_token_id
    keep = []
    for t, l in zip(gen.tolist(), lp.tolist()):
        if t == pad and t not in eos:
            break
        keep.append(l)
        if t in eos:
            break
    return {"pred": tokenizer.decode(gen, skip_special_tokens=True).strip(),
            "sum_logprob": float(sum(keep)), "mean_logprob": float(sum(keep) / max(len(keep), 1)),
            "n_gen_tokens": len(keep)}


def select(rows: list[dict], key: str) -> dict:
    best = rows[0]
    for r in rows[1:]:
        if r[key] > best[key]:
            best = r
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--authors-dir", type=Path, required=True)
    ap.add_argument("--frozen", type=Path, required=True, help="our reproduction run dir (runs/p17_repro)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-new", type=int, default=32)
    args = ap.parse_args()
    sys.path.insert(0, str(args.authors_dir / "src"))
    import pyarrow.parquet as pq
    from tts_kbvqa.data.infoseek import download_passages_parquet, download_val_parquet, load_passages
    from tts_kbvqa.eval import infoseek_official
    from tts_kbvqa.reader.qwen import QwenReader

    torch.backends.cuda.enable_cudnn_sdp(False)   # as in the authors' runners
    torch.manual_seed(42)
    args.out.mkdir(parents=True, exist_ok=True)

    # the same 10 passages per question as the k=10 cells (gold first + 9 distractors)
    k10 = {}
    for line in (args.frozen / "predictions_k10_first.jsonl").open():
        if line.strip():
            r = json.loads(line)
            k10[r["qid"]] = r
    k1 = {}
    for line in (args.frozen / "predictions_k1.jsonl").open():
        if line.strip():
            r = json.loads(line)
            k1[r["qid"]] = r["pred"]
    qids = sorted(k10)
    log(f"{len(qids)} questions x 10 passages")

    cols = ["question_id", "question", "answers", "answer_eval", "img_path", "question_type"]
    rows = {r["question_id"]: r for r in pq.read_table(download_val_parquet(), columns=cols).to_pylist()}
    id_to_passage = {p.passage_id: p for p in load_passages(download_passages_parquet())}
    image_dir = args.authors_dir / "data/infoseek/images/val"

    cache_path = args.out / "cache.jsonl"
    cache = {}
    if cache_path.exists():
        for line in cache_path.open():
            if line.strip():
                r = json.loads(line)
                cache[(r["qid"], r["pid"])] = r
    todo = [(q, p) for q in qids for p in k10[q]["passages_ordered"] if (q, p) not in cache]
    log(f"cached {len(cache)}, to run {len(todo)}")

    t0 = time.time()
    if todo:
        if __import__("os").environ.get("LAE_ADAPTER"):
            raise SystemExit("LAE_ADAPTER is set: isolated reading is defined on the frozen reader")
        reader = QwenReader(dtype=torch.bfloat16, model_id="Qwen/Qwen2.5-VL-7B-Instruct")
        torch.cuda.reset_peak_memory_stats()
        with cache_path.open("a") as fh:
            for i, (q, pid) in enumerate(todo, 1):
                r = rows[q]
                ts = time.time()
                inputs, _, input_len = reader._prepare_inputs(image_dir / r["img_path"], r["question"],
                                                              [id_to_passage[pid]])
                res = generate_scored(reader.model, reader.processor.tokenizer, inputs, input_len, args.max_new)
                rec = {"qid": q, "pid": pid, **res, "input_tokens": input_len, "latency_s": time.time() - ts}
                cache[(q, pid)] = rec
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                if i % 200 == 0 or i == len(todo):
                    rate = (time.time() - t0) / i
                    log(f"  {i}/{len(todo)}  {rate:.2f}s/passage  eta {rate * (len(todo) - i) / 60:.1f} min  "
                        f"peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
        peak = torch.cuda.max_memory_allocated() / 1e9
    else:
        peak = None

    preds, agree = [], 0
    for q in qids:
        r = rows[q]
        cand = [cache[(q, p)] for p in k10[q]["passages_ordered"]]   # gold is index 0 here
        ok = lambda pred: bool(infoseek_official.is_correct(pred, r["question_type"], list(r["answers"] or []),
                                                            gold_answer_eval=list(r["answer_eval"] or [])))
        b_mean, b_sum = select(cand, "mean_logprob"), select(cand, "sum_logprob")
        agree += cand[0]["pred"] == k1[q]
        preds.append({"qid": q, "pred": b_mean["pred"], "chosen_pid": b_mean["pid"],
                      "chosen_is_gold": b_mean["pid"] == k10[q]["gold_id"],
                      "correct_official": ok(b_mean["pred"]),
                      "pred_sum": b_sum["pred"], "correct_official_sum": ok(b_sum["pred"]),
                      "any_passage_correct": any(ok(c["pred"]) for c in cand),
                      "gold_alone_correct": ok(cand[0]["pred"]),
                      "latency_s_10_calls": sum(c["latency_s"] for c in cand),
                      "input_tokens_10_calls": sum(c["input_tokens"] for c in cand),
                      "gold_via_fallback": k10[q]["gold_via_fallback"]})
    with (args.out / "predictions.jsonl").open("w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")

    n = len(preds)
    def acc(key):
        h = sum(p[key] for p in preds)
        lo, hi = wilson(h, n)
        return {"hits": h, "acc": h / n, "ci_low": lo, "ci_high": hi}
    summary = {"n": n, "scoring_primary": "mean_logprob", "scoring_secondary": "sum_logprob",
               "primary": acc("correct_official"), "secondary_sum": acc("correct_official_sum"),
               "any_of_10_correct": acc("any_passage_correct"), "gold_alone": acc("gold_alone_correct"),
               "chose_gold_passage": sum(p["chosen_is_gold"] for p in preds) / n,
               "gold_alone_matches_k1_cell": agree / n,
               "mean_latency_s_per_question": sum(p["latency_s_10_calls"] for p in preds) / n,
               "mean_input_tokens_per_question": sum(p["input_tokens_10_calls"] for p in preds) / n,
               "peak_alloc_gb": peak, "wall_s_this_session": time.time() - t0,
               "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    log(json.dumps({k: v for k, v in summary.items()}, indent=1))
    if summary["gold_alone_matches_k1_cell"] < 0.99:
        log(f"HARD CHECK FAILED: gold-alone answers match the k=1 cell for only "
            f"{100 * summary['gold_alone_matches_k1_cell']:.1f}% of questions (need >= 99%)")
        return 1
    log("hard check passed: gold-alone answers match the reproduction's k=1 cell")
    return 0


if __name__ == "__main__":
    sys.exit(main())
