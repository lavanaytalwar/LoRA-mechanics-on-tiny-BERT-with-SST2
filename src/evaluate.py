from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .utils import ensure_dir, read_json


def collect_runs(runs_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in sorted(runs_dir.glob("*")):
        if not run.is_dir():
            continue
        meta_p = run / "meta.json"
        metrics_p = run / "metrics.json"
        if not meta_p.exists() or not metrics_p.exists():
            continue
        meta = read_json(meta_p)
        metrics = read_json(metrics_p)
        row = {
            "run": run.name,
            "mode": meta.get("mode"),
            "model_name": meta.get("model_name"),
            "seed": meta.get("seed"),
            "total_params": meta.get("total_params"),
            "trainable_params": meta.get("trainable_params"),
            "trainable_frac": meta.get("trainable_frac"),
            "replaced_modules": meta.get("replaced_modules"),
            "lora_r": (meta.get("lora") or {}).get("r"),
            "lora_alpha": (meta.get("lora") or {}).get("alpha"),
            "accuracy": metrics.get("accuracy"),
            "f1": metrics.get("f1"),
            "eval_loss": metrics.get("eval_loss"),
            "best_accuracy": metrics.get("best_accuracy"),
            "train_seconds": metrics.get("train_seconds"),
            "checkpoint_bytes": metrics.get("checkpoint_bytes"),
            "best_bytes": metrics.get("best_bytes"),
        }
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--runs_dir", type=str, required=True)
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = ensure_dir(Path("outputs"))

    rows = collect_runs(runs_dir)
    df = pd.DataFrame(rows)
    df = df.sort_values(by=["mode", "lora_r", "accuracy"], ascending=[True, True, False], na_position="last")
    out_csv = out_dir / "metrics.csv"
    df.to_csv(out_csv, index=False)

    print(str(out_csv))


if __name__ == "__main__":
    main()

