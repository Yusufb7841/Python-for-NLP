"""Test-set metrics, flattened into one CSV row per run."""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from agnews.data import LABEL_NAMES

CLASS_KEYS = ["world", "sports", "business", "scitech"]


def classification_metrics(y_true, y_pred):
    labels = list(range(len(LABEL_NAMES)))
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "test_accuracy": accuracy_score(y_true, y_pred),
        "test_macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        **{f"f1_{key}": value for key, value in zip(CLASS_KEYS, per_class_f1, strict=True)},
        "confusion": " ".join(str(count) for count in matrix.ravel()),
    }


def parse_confusion(value):
    return np.array(value.split(), dtype=int).reshape(len(LABEL_NAMES), len(LABEL_NAMES))
