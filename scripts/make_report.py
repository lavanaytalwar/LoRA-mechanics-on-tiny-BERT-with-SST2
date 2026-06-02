from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def bytes_to_mb(b: float) -> float:
    return float(b) / (1024.0 * 1024.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, required=True)
    args = ap.parse_args()

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = out_dir / "metrics.csv"
    if not metrics_csv.exists():
        raise FileNotFoundError("Run `python -m src.evaluate ...` first to create outputs/metrics.csv")

    df = pd.read_csv(metrics_csv)

    # Plots: accuracy vs trainable params, and accuracy vs rank (LoRA)
    df2 = df.copy()
    df2["trainable_m"] = df2["trainable_params"] / 1e6
    df2["ckpt_mb"] = df2["checkpoint_bytes"].apply(bytes_to_mb)
    df2["time_s"] = df2["train_seconds"]

    # accuracy vs trainable params
    plt.figure(figsize=(7, 4))
    for mode, g in df2.groupby("mode"):
        plt.scatter(g["trainable_m"], g["accuracy"], label=mode)
    plt.xlabel("Trainable parameters (M)")
    plt.ylabel("Validation accuracy")
    plt.title("Accuracy vs trainable parameters")
    plt.legend()
    p1 = plots_dir / "acc_vs_trainable_params.png"
    plt.tight_layout()
    plt.savefig(p1, dpi=160)
    plt.close()

    # LoRA accuracy vs rank
    lora = df2[df2["mode"].isin(["lora", "peft_lora"])].dropna(subset=["lora_r"])
    if len(lora) > 0:
        plt.figure(figsize=(7, 4))
        for mode, g in lora.groupby("mode"):
            g = g.sort_values("lora_r")
            plt.plot(g["lora_r"], g["accuracy"], marker="o", label=mode)
        plt.xlabel("LoRA rank r")
        plt.ylabel("Validation accuracy")
        plt.title("Accuracy vs LoRA rank")
        plt.legend()
        p2 = plots_dir / "acc_vs_rank.png"
        plt.tight_layout()
        plt.savefig(p2, dpi=160)
        plt.close()

    # Markdown report
    def _md_table(df_in: pd.DataFrame) -> str:
        cols = [
            "mode",
            "lora_r",
            "accuracy",
            "f1",
            "trainable_params",
            "total_params",
            "train_seconds",
            "checkpoint_bytes",
        ]
        d = df_in[cols].copy()
        d["trainable_params"] = d["trainable_params"].map(lambda x: f"{int(x):,}" if pd.notna(x) else "")
        d["total_params"] = d["total_params"].map(lambda x: f"{int(x):,}" if pd.notna(x) else "")
        d["train_seconds"] = d["train_seconds"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "")
        d["checkpoint_bytes"] = d["checkpoint_bytes"].map(lambda x: f"{bytes_to_mb(x):.2f} MB" if pd.notna(x) else "")
        d["accuracy"] = d["accuracy"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
        d["f1"] = d["f1"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
        d["lora_r"] = d["lora_r"].map(lambda x: "" if pd.isna(x) else int(x))
        return d.to_markdown(index=False)

    # pick best per setting
    best_full = df[df["mode"] == "full"].sort_values("accuracy", ascending=False).head(1)
    best_lora = df[df["mode"] == "lora"].sort_values(["lora_r", "accuracy"], ascending=[True, False])
    best_peft = df[df["mode"] == "peft_lora"].sort_values(["lora_r", "accuracy"], ascending=[True, False])

    report = []
    report.append("# LoRA reproduction report (lightweight)\n")
    report.append("## What we reproduced\n")
    report.append(
        "- Custom LoRA layer injection with frozen base weights (`src/lora.py`)\n"
        "- Trainable low-rank adapters (A/B) + scaling alpha/r\n"
        "- Full fine-tuning vs LoRA fine-tuning comparison\n"
        "- Rank sweep: r = 2, 4, 8, 16\n"
        "- Parameter-efficiency + checkpoint size + runtime measurements\n"
    )
    report.append("## Environment constraints\n")
    report.append("- CPU-only, small model (`prajjwal1/bert-tiny`), SST-2 subset.\n")

    report.append("## Key results (best runs)\n")
    if len(best_full) > 0:
        report.append("### Best full fine-tune\n")
        report.append(_md_table(best_full) + "\n")
    if len(best_lora) > 0:
        report.append("### LoRA (custom) rank sweep\n")
        report.append(_md_table(best_lora) + "\n")
    if len(best_peft) > 0:
        report.append("### LoRA (PEFT) rank sweep\n")
        report.append(_md_table(best_peft) + "\n")

    report.append("## Observations vs paper expectations\n")
    report.append(
        "- **Trainable parameter reduction**: LoRA trains a small fraction of parameters while keeping base weights frozen.\n"
        "- **Quality trend vs rank**: accuracy typically improves with rank up to a point (diminishing returns).\n"
        "- **Checkpoint size**: LoRA runs save smaller effective updates (in this repo we save full HF checkpoints for simplicity; see `Future improvements`).\n"
    )

    report.append("## What differs (scaled-down reproduction)\n")
    report.append(
        "- We reproduce mechanics + trends on a tiny encoder model and SST-2 subset, not large-scale generative LMs.\n"
        "- We use short CPU-friendly runs and small data slices.\n"
    )

    report.append("## Artifacts\n")
    report.append(f"- Metrics CSV: `{metrics_csv.as_posix()}`\n")
    report.append(f"- Plots: `{plots_dir.as_posix()}`\n")

    report.append("## Future improvements\n")
    report.append(
        "- Save **adapter-only** weights for custom LoRA (A/B matrices only) for true checkpoint-size comparisons.\n"
        "- Add a tiny causal LM task (perplexity) to mirror the original paper setting.\n"
    )

    Path("results.md").write_text("\n".join(report), encoding="utf-8")
    print(str(Path("results.md")))


if __name__ == "__main__":
    main()

