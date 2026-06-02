from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--text", type=str, required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / "best"
    if not ckpt_dir.exists():
        ckpt_dir = run_dir / "checkpoint"

    try:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)
    except ValueError:
        print("Fast tokenizer unavailable; using BertTokenizer.")
        tokenizer = BertTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
    model.eval()

    batch = tokenizer(args.text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        out = model(**batch)
        probs = torch.softmax(out.logits, dim=-1)[0].cpu().tolist()
        pred = int(torch.argmax(out.logits, dim=-1)[0].item())

    print({"pred": pred, "probs": probs})


if __name__ == "__main__":
    main()

