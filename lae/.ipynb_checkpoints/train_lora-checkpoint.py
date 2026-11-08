"""Stage 3: LoRA SFT of Qwen2.5-VL-7B-Instruct on one training condition.

Same for every condition (only the data file differs):
  * model + processor loaded exactly as the authors' reader loads them
    (bf16, min/max pixels 3,136 / 802,816), cuDNN SDP disabled as in their runners
  * input = the authors' prompt: system prompt, then user [image, "question\\n\\nContext:\\n[1] ...\\n---\\n[2] ..."]
  * target = answer + <|im_end|>; loss on target tokens only
  * LoRA r=64, alpha=128, dropout=0.05 on the language model's q,k,v,o,gate,up,down
    (vision encoder and merger frozen; matched by full module path, because the
    vision blocks also contain gate/up/down_proj layers)
  * 1 epoch, effective batch 16, lr 1e-4, cosine, 3% warmup, bf16, gradient checkpointing
  * the run seed sets LoRA init and example order, so standard seed S and balanced_sS
    see the same examples in the same order; only passage order differs

Resumable: rerunning with the same --out continues from the last checkpoint.
--smoke: a short run that also prints one example with its label mask.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

LM_LORA_TARGETS = r"model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
SUFFIX = "<|im_end|>\n"   # closes the assistant turn in Qwen's chat template


def log(msg: str) -> None:
    print(f"[train {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_jsonl(path: Path, limit: int = 0) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict], passages: dict, image_dir: Path):
        self.rows, self.passages, self.image_dir = rows, passages, image_dir

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        return {"qid": r["qid"], "question": r["question"], "target": r["target"],
                "passages": [self.passages[p] for p in r["passage_ids"]],
                "image_path": str(self.image_dir / r["img_path"])}


class Collator:
    """Builds the authors' reader prompt, appends the target, masks everything but the target."""

    def __init__(self, processor, system_prompt: str, passage_block, max_len: int, load_image=None):
        self.p, self.system_prompt, self.passage_block = processor, system_prompt, passage_block
        self.max_len = max_len
        self.load_image = load_image or (lambda path: __import__("PIL.Image", fromlist=["Image"])
                                         .open(path).convert("RGB"))

    def prompt_text(self, ex: dict) -> str:
        # Identical to tts_kbvqa.reader.qwen.QwenReader._prepare_inputs with image_position="start".
        user_text = f"{ex['question']}\n\nContext:\n{self.passage_block(ex['passages'])}"
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]}]
        return self.p.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def __call__(self, batch: list[dict]) -> dict:
        texts = [self.prompt_text(ex) + ex["target"] + SUFFIX for ex in batch]
        images = [self.load_image(ex["image_path"]) for ex in batch]
        enc = self.p(text=texts, images=images, padding=True, return_tensors="pt")
        ids, att = enc["input_ids"], enc["attention_mask"]
        labels = torch.full_like(ids, -100)
        for i, ex in enumerate(batch):
            tgt = self.p.tokenizer(ex["target"] + SUFFIX, add_special_tokens=False)["input_ids"]
            n = int(att[i].sum())
            if n > self.max_len:
                raise ValueError(f"{ex['qid']}: {n} tokens > max_len {self.max_len}")
            if ids[i, n - len(tgt):n].tolist() != tgt:
                raise ValueError(f"{ex['qid']}: target tokens not found at the end of the sequence")
            labels[i, n - len(tgt):n] = ids[i, n - len(tgt):n]
        enc["labels"] = labels
        return dict(enc)


def load_passages(authors_dir: Path) -> dict:
    sys.path.insert(0, str(authors_dir / "src"))
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    from tts_kbvqa.data.infoseek import M2KR_REPO, Passage
    f = sorted(s.rfilename for s in HfApi().dataset_info(M2KR_REPO).siblings
               if s.rfilename.startswith("Infoseek_passages/train"))[0]
    t = pq.read_table(hf_hub_download(M2KR_REPO, f, repo_type="dataset"),
                      columns=["passage_id", "passage_content", "title"]).to_pylist()
    return {r["passage_id"]: Passage(r["passage_id"], r["title"], r["passage_content"]) for r in t}


def build_model(args):
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    return wrap_lora(model, args)


def wrap_lora(model, args):
    from peft import LoraConfig, get_peft_model
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=LM_LORA_TARGETS, bias="none", task_type="CAUSAL_LM"))
    lora = [n for n, _ in model.named_modules() if n.endswith(".lora_A")]
    bad = [n for n in lora if "visual" in n]
    n_layers = model.base_model.model.config.num_hidden_layers
    assert not bad, f"LoRA landed in the vision tower: {bad[:3]}"
    assert len(lora) == 7 * n_layers, f"expected {7 * n_layers} LoRA modules, got {len(lora)}"
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"LoRA modules: {len(lora)} (7 x {n_layers} layers, none in vision); "
        f"trainable params {trainable / 1e6:.1f}M of {total / 1e9:.2f}B ({100 * trainable / total:.2f}%)")
    return model


def git_rev(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main(argv=None, model_override=None, processor_override=None, passages_override=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--authors-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True, help=".../final")
    ap.add_argument("--condition", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--per-device-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--save-steps", type=int, default=250)
    ap.add_argument("--max-len", type=int, default=12000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="short run: 160 train / 32 dev examples, 10 steps")
    args = ap.parse_args(argv)

    from transformers import Trainer, TrainingArguments, set_seed
    sys.path.insert(0, str(args.authors_dir / "src"))
    from tts_kbvqa.reader.qwen import SYSTEM_PROMPT, VISION_MAX_PIXELS, VISION_MIN_PIXELS, passage_block

    torch.backends.cuda.enable_cudnn_sdp(False)
    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "adapter" / "adapter_model.safetensors").exists() and (args.out / "train_metrics.json").exists():
        log(f"already complete: {args.out / 'adapter'} exists; delete the run directory to retrain")
        return 0
    cond_dir = args.data_dir / args.condition
    train_rows = read_jsonl(cond_dir / "train.jsonl", 160 if args.smoke else 0)
    dev_rows = read_jsonl(cond_dir / "dev.jsonl", 32 if args.smoke else 0)
    log(f"condition {args.condition}: {len(train_rows)} train, {len(dev_rows)} dev, seed {args.seed}")

    passages = passages_override if passages_override is not None else load_passages(args.authors_dir)
    image_dir = args.authors_dir / "data/infoseek/images/train"
    if processor_override is not None:
        processor = processor_override
    else:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=VISION_MIN_PIXELS,
                                                  max_pixels=VISION_MAX_PIXELS)
    processor.tokenizer.padding_side = "right"
    collator = Collator(processor, SYSTEM_PROMPT, passage_block, args.max_len)
    train_ds = SFTDataset(train_rows, passages, image_dir)
    dev_ds = SFTDataset(dev_rows, passages, image_dir)

    if args.smoke:  # show exactly what the model is trained on for one example
        b = collator([train_ds[0]])
        ids, lab = b["input_ids"][0], b["labels"][0]
        tail = processor.tokenizer.decode(ids[-120:])
        trained = processor.tokenizer.decode(ids[lab != -100])
        log(f"example {train_rows[0]['qid']}: {int(b['attention_mask'][0].sum())} tokens, "
            f"{int((lab != -100).sum())} trained")
        log(f"  ...end of input: {tail!r}")
        log(f"  trained-on tokens (label mask): {trained!r}   target was {train_rows[0]['target']!r}")

    model = model_override if model_override is not None else build_model(args)

    eff = args.per_device_batch * args.grad_accum
    steps = 10 if args.smoke else math.ceil(len(train_ds) * args.epochs / eff)
    targs = TrainingArguments(
        output_dir=str(args.out), seed=args.seed, data_seed=args.seed,
        per_device_train_batch_size=args.per_device_batch, per_device_eval_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr, num_train_epochs=args.epochs,
        max_steps=10 if args.smoke else -1, lr_scheduler_type="cosine", warmup_ratio=args.warmup_ratio,
        weight_decay=0.0, max_grad_norm=1.0, bf16=torch.cuda.is_available(),
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1 if args.smoke else 10, eval_strategy="steps",
        eval_steps=5 if args.smoke else args.save_steps, save_strategy="steps",
        save_steps=5 if args.smoke else args.save_steps, save_total_limit=2,
        dataloader_num_workers=0 if args.smoke else args.workers, remove_unused_columns=False,
        report_to="none", optim="adamw_torch", disable_tqdm=True, label_names=["labels"])
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=dev_ds,
                      data_collator=collator)

    config = {"condition": args.condition, "seed": args.seed, "model": MODEL_ID,
              "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout,
                       "target_modules_regex": LM_LORA_TARGETS},
              "lr": args.lr, "epochs": args.epochs, "per_device_batch": args.per_device_batch,
              "grad_accum": args.grad_accum, "effective_batch": eff, "planned_steps": steps,
              "warmup_ratio": args.warmup_ratio, "max_len": args.max_len,
              "n_train": len(train_ds), "n_dev": len(dev_ds), "smoke": args.smoke,
              "vision_min_max_pixels": [VISION_MIN_PIXELS, VISION_MAX_PIXELS],
              "data_manifest": json.loads((args.data_dir / "manifest.json").read_text())
              if (args.data_dir / "manifest.json").exists() else None,
              "authors_commit": git_rev(args.authors_dir), "lae_commit": git_rev(Path(__file__).parent),
              "torch": torch.__version__, "transformers": __import__("transformers").__version__,
              "peft": __import__("peft").__version__,
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    (args.out / "train_config.json").write_text(json.dumps(config, indent=2))
    log(f"{steps} optimizer steps, effective batch {eff}; config -> {args.out / 'train_config.json'}")

    resume = any(p.name.startswith("checkpoint-") for p in args.out.iterdir())
    if resume:
        log("resuming from the last checkpoint in the output directory")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=True if resume else None)
    wall = time.time() - t0
    dev = trainer.evaluate()
    model.save_pretrained(str(args.out / "adapter"))

    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    metrics = {"train_wall_s_this_session": round(wall, 1),
               "train_loss_mean_this_session": result.training_loss,
               "train_loss_last_logged": losses[-1] if losses else None,
               "dev_loss": dev.get("eval_loss"), "global_steps": result.global_step,
               "s_per_step": None if resume else round(wall / max(result.global_step, 1), 2),
               "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
               if torch.cuda.is_available() else None,
               "resumed": resume, "log_history": trainer.state.log_history}
    (args.out / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    log(f"done: {result.global_step} steps in {wall / 60:.1f} min ({metrics['s_per_step']} s/step), "
        f"last logged train loss {metrics['train_loss_last_logged']}, dev loss {metrics['dev_loss']}, "
        f"peak GPU {metrics['peak_gpu_mem_gb']} GB; adapter -> {args.out / 'adapter'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
