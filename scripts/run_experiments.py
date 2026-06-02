from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/sst2_bert_tiny.yaml")
    ap.add_argument("--ranks", type=str, default="2,4,8,16")
    ap.add_argument("--with_peft", action="store_true")
    args = ap.parse_args()

    cwd = Path(__file__).resolve().parents[1]
    py = sys.executable  # assumes you're running inside the venv

    run([py, "-m", "src.train", "--config", args.config, "--mode", "full"], cwd=cwd)

    ranks = [int(x.strip()) for x in args.ranks.split(",") if x.strip()]
    for r in ranks:
        run([py, "-m", "src.train", "--config", args.config, "--mode", "lora", "--lora_r", str(r)], cwd=cwd)

    if args.with_peft and 8 in ranks:
        run([py, "-m", "src.train", "--config", args.config, "--mode", "peft_lora", "--lora_r", "8"], cwd=cwd)

    run([py, "-m", "src.evaluate", "--config", args.config, "--runs_dir", "outputs/runs"], cwd=cwd)
    run([py, "-m", "scripts.make_report", "--runs_dir", "outputs/runs"], cwd=cwd)


if __name__ == "__main__":
    main()

