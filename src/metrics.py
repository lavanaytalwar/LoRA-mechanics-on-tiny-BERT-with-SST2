from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


@dataclass
class Metrics:
    accuracy: float
    f1: float


def compute_classification_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    preds = preds.astype(int)
    labels = labels.astype(int)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average="binary")),
    }

