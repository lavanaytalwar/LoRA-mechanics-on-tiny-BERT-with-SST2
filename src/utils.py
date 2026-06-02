from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)  # CPU kernels may be slow with True


def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_text(path: str | Path, text: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def get_cpu_memory_mb() -> Optional[float]:
    # Best-effort, no extra deps. Uses Windows-only env when available.
    try:
        import resource  # type: ignore

        rusage = resource.getrusage(resource.RUSAGE_SELF)
        # On Windows, resource may be missing; on Linux ru_maxrss is KB.
        return float(rusage.ru_maxrss) / 1024.0
    except Exception:
        return None


def atomic_write(path: str | Path, data: bytes) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(data)
    os.replace(tmp, path)

