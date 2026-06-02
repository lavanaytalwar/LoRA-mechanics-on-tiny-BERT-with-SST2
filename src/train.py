from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from .config import load_yaml
from .dataset import DatasetConfig, get_label_info, load_sst2
from .lora import LoRAConfig, count_parameters, inject_lora, mark_only_lora_as_trainable
from .metrics import compute_classification_metrics
from .utils import ensure_dir, now_ts, set_seed, write_json


def _tokenize(ds, tokenizer, text_key: str, max_length: int):
    def f(batch):
        return tokenizer(batch[text_key], truncation=True, max_length=max_length)

    # Keep labels, drop raw text to avoid collator tensorization issues
    remove_cols = [c for c in ds.column_names if c != "label"]
    return ds.map(f, batched=True, remove_columns=remove_cols)


@torch.no_grad()
def evaluate(model, dl, device) -> Dict[str, float]:
    model.eval()
    preds = []
    labels = []
    loss_sum = 0.0
    n = 0
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss
        logits = out.logits
        pred = torch.argmax(logits, dim=-1)
        preds.append(pred.cpu().numpy())
        labels.append(batch["labels"].cpu().numpy())
        loss_sum += float(loss.item()) * batch["labels"].shape[0]
        n += batch["labels"].shape[0]

    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    m = compute_classification_metrics(preds, labels)
    m["eval_loss"] = loss_sum / max(1, n)
    return m


def _save_checkpoint(run_dir: Path, model, tokenizer) -> int:
    ckpt = run_dir / "checkpoint"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tokenizer.save_pretrained(ckpt)
    # bytes
    total = 0
    for p in ckpt.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--mode", type=str, choices=["full", "lora", "peft_lora"], required=True)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--run_name", type=str, default=None)
    ap.add_argument("--no_train_classifier", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))

    device = torch.device("cpu")

    # data
    dcfg = DatasetConfig(**cfg["dataset"])
    ds = load_sst2(dcfg)
    num_labels, label_names = get_label_info(ds)

    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    except ValueError:
        print("Fast tokenizer unavailable; using BertTokenizer.")
        tokenizer = BertTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(cfg["model_name"], num_labels=num_labels)
    model.to(device)

    # tokenize + dataloaders
    max_length = int(cfg["max_length"])
    train_ds = _tokenize(ds["train"], tokenizer, dcfg.text_key, max_length)
    eval_ds = _tokenize(ds["validation"], tokenizer, dcfg.text_key, max_length)
    train_ds = train_ds.rename_column("label", "labels")
    eval_ds = eval_ds.rename_column("label", "labels")
    train_ds.set_format(type="torch")
    eval_ds.set_format(type="torch")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dl = DataLoader(train_ds, batch_size=int(cfg["train"]["train_batch_size"]), shuffle=True, collate_fn=collator)
    eval_dl = DataLoader(eval_ds, batch_size=int(cfg["train"]["eval_batch_size"]), shuffle=False, collate_fn=collator)

    # LoRA / PEFT
    replaced = 0
    lora_cfg_dict = None

    if args.mode == "lora":
        lcfg = cfg["lora"]
        lora_cfg = LoRAConfig(
            r=int(args.lora_r),
            alpha=int(lcfg["alpha"]),
            dropout=float(lcfg["dropout"]),
            target_modules=tuple(lcfg["target_modules"]),
        )
        replaced = inject_lora(model, lora_cfg)
        mark_only_lora_as_trainable(model, train_classifier=not args.no_train_classifier)
        lora_cfg_dict = asdict(lora_cfg)

    elif args.mode == "peft_lora":
        from peft import LoraConfig as PeftLoraConfig
        from peft import get_peft_model

        lcfg = cfg["lora"]
        peft_cfg = PeftLoraConfig(
            r=int(args.lora_r),
            lora_alpha=int(lcfg["alpha"]),
            lora_dropout=float(lcfg["dropout"]),
            bias="none",
            target_modules=list(lcfg["target_modules"]),
            task_type="SEQ_CLS",
        )
        model = get_peft_model(model, peft_cfg)
        model.to(device)
        lora_cfg_dict = {
            "peft": True,
            "r": int(args.lora_r),
            "alpha": int(lcfg["alpha"]),
            "dropout": float(lcfg["dropout"]),
            "target_modules": list(lcfg["target_modules"]),
        }

    # params + optimizer
    total_params, trainable_params = count_parameters(model)

    lr = float(cfg["train"]["lr_full"] if args.mode == "full" else cfg["train"]["lr_lora"])
    wd = float(cfg["train"]["weight_decay"])
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)

    epochs = int(cfg["train"]["epochs"])
    grad_accum = int(cfg["train"]["grad_accum_steps"])
    steps_per_epoch = math.ceil(len(train_dl) / grad_accum)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * float(cfg["train"]["warmup_ratio"]))
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # run folder
    base_name = args.run_name or cfg["run_name"]
    suffix = args.mode if args.mode != "lora" else f"lora_r{args.lora_r}"
    run_dir = ensure_dir(Path(cfg["outputs_dir"]) / f"{base_name}_{suffix}_{now_ts()}")

    meta = {
        "run_name": str(run_dir.name),
        "mode": args.mode,
        "model_name": cfg["model_name"],
        "dataset": asdict(dcfg),
        "train": cfg["train"],
        "lora": lora_cfg_dict,
        "replaced_modules": int(replaced),
        "seed": int(cfg["seed"]),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "trainable_frac": float(trainable_params) / float(max(1, total_params)),
        "label_names": label_names,
        "device": str(device),
    }
    write_json(run_dir / "meta.json", meta)

    # training
    log_every = int(cfg["train"]["log_every"])
    best_acc = -1.0
    best_path = run_dir / "best"
    best_path.mkdir(parents=True, exist_ok=True)

    global_step = 0
    t0 = time.time()
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(train_dl, desc=f"epoch {epoch+1}/{epochs}", leave=True)
        optim.zero_grad(set_to_none=True)
        running = 0.0
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / grad_accum
            loss.backward()
            running += float(loss.item()) * grad_accum

            if (step + 1) % grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % log_every == 0:
                    pbar.set_postfix({"loss": f"{running/log_every:.4f}", "lr": f"{sched.get_last_lr()[0]:.2e}"})
                    running = 0.0

        # eval each epoch
        metrics = evaluate(model, eval_dl, device)
        write_json(run_dir / f"eval_epoch_{epoch+1}.json", {"epoch": epoch + 1, **metrics})

        if metrics["accuracy"] > best_acc:
            best_acc = float(metrics["accuracy"])
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)

    train_seconds = time.time() - t0
    ckpt_bytes = _save_checkpoint(run_dir, model, tokenizer)
    best_bytes = 0
    for p in best_path.rglob("*"):
        if p.is_file():
            best_bytes += p.stat().st_size

    final_metrics = evaluate(model, eval_dl, device)
    metrics_out = {
        **final_metrics,
        "best_accuracy": best_acc,
        "train_seconds": float(train_seconds),
        "checkpoint_bytes": int(ckpt_bytes),
        "best_bytes": int(best_bytes),
    }
    write_json(run_dir / "metrics.json", metrics_out)

    # also dump a compact line for quick grepping
    (run_dir / "summary.jsonl").write_text(json.dumps({"meta": meta, "metrics": metrics_out}) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

