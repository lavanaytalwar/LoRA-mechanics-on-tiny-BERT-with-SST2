from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import datasets
from datasets import DatasetDict


@dataclass
class DatasetConfig:
    name: str = "glue"
    subset: str = "sst2"
    text_key: str = "sentence"
    label_key: str = "label"
    train_max_samples: int = 2000
    eval_max_samples: int = 500
    seed: int = 42


def load_sst2(cfg: DatasetConfig) -> DatasetDict:
    ds = datasets.load_dataset(cfg.name, cfg.subset)
    # glue/sst2: train/validation/test (test has no labels). use validation for eval.
    train = ds["train"].shuffle(seed=cfg.seed)
    val = ds["validation"].shuffle(seed=cfg.seed)

    if cfg.train_max_samples > 0:
        train = train.select(range(min(cfg.train_max_samples, len(train))))
    if cfg.eval_max_samples > 0:
        val = val.select(range(min(cfg.eval_max_samples, len(val))))

    return DatasetDict(train=train, validation=val)


def get_label_info(ds: DatasetDict) -> Tuple[int, Dict[int, str]]:
    features = ds["train"].features
    if "label" in features and hasattr(features["label"], "names") and features["label"].names:
        names = {i: n for i, n in enumerate(features["label"].names)}
        return len(names), names
    # fallback
    return 2, {0: "neg", 1: "pos"}

