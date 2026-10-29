"""Classification metrics with NumPy only (no scikit-learn needed)."""
from __future__ import annotations

import numpy as np


def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    """Rows = true class, columns = predicted class."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def _div(a, b):
    return float(a) / float(b) if b else 0.0


def classification_report(cm: np.ndarray, class_names: list[str]) -> dict:
    n = cm.sum()
    per = {}
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p, r = _div(tp, tp + fp), _div(tp, tp + fn)
        per[name] = {"precision": p, "recall": r, "f1": _div(2 * p * r, p + r),
                     "support": int(cm[i, :].sum())}
    sup = np.array([per[c]["support"] for c in class_names], dtype=float)
    avg = lambda key, w=None: float(np.average([per[c][key] for c in class_names], weights=w))
    return {
        "accuracy": _div(np.trace(cm), n),
        "n_images": int(n),
        "per_class": per,
        "macro_avg": {k: avg(k) for k in ("precision", "recall", "f1")},
        "weighted_avg": {k: avg(k, sup) for k in ("precision", "recall", "f1")},
    }


def threshold_table(correct, confidence, margin, thresholds, margin_min: float) -> list[dict]:
    """For each confidence threshold: how many predictions would be accepted
    and how accurate those are (optionally also requiring margin >= margin_min)."""
    correct = np.asarray(correct, dtype=bool)
    confidence = np.asarray(confidence, dtype=float)
    margin = np.asarray(margin, dtype=float)
    n = len(correct)
    rows = []
    for t in thresholds:
        for use_margin in (False, True):
            keep = confidence >= t
            if use_margin:
                keep &= margin >= margin_min
            k = int(keep.sum())
            rows.append({
                "confidence_min": t, "margin_min": margin_min if use_margin else None,
                "accepted": k, "coverage": _div(k, n),
                "accuracy_of_accepted": _div(correct[keep].sum(), k),
                "errors_accepted": int((~correct[keep]).sum()),
                "sent_to_review_or_reject": n - k,
            })
    return rows
