"""Validation module: turns prediction + preprocessing result into a status.

Statuses (cahier des charges): Accepté / Révision requise / Rejeté.
Pure function, JSON-serialisable output, no model or file access except an
existence check of the preprocessed artifact.
"""
from __future__ import annotations

from pathlib import Path

import config

ACCEPTED = "ACCEPTED"
REVIEW = "REVIEW_REQUIRED"
REJECTED = "REJECTED"
LABELS_FR = {ACCEPTED: "Accepté", REVIEW: "Révision requise", REJECTED: "Rejeté"}


def validate(preprocessing: dict, prediction: dict | None) -> dict:
    reject: list[str] = []
    review: list[str] = []
    checks: dict[str, bool] = {}

    # 1. Preprocessing succeeded?
    checks["preprocessing_valid"] = bool(preprocessing.get("valid"))
    if not checks["preprocessing_valid"]:
        reject.append("preprocessing_failed:" + ",".join(preprocessing.get("errors", [])))

    # 2. Previous artifact present? (full hash check comes with the hashing module)
    path = preprocessing.get("preprocessed_path")
    checks["preprocessed_file_exists"] = bool(path) and Path(path).is_file()
    if checks["preprocessing_valid"] and not checks["preprocessed_file_exists"]:
        reject.append("preprocessed_artifact_missing")

    if prediction is not None and checks["preprocessing_valid"]:
        probs = prediction.get("probabilities", {})
        conf = float(prediction.get("confidence", 0.0))
        margin = float(prediction.get("margin_top2", 0.0))

        # 3. Internal consistency of the prediction
        checks["probabilities_sum_to_1"] = abs(sum(probs.values()) - 1.0) < 1e-3
        checks["class_in_probabilities"] = prediction.get("predicted_class") in probs
        checks["model_version_present"] = bool(prediction.get("model_version"))
        if not all(checks[k] for k in ("probabilities_sum_to_1",
                                       "class_in_probabilities",
                                       "model_version_present")):
            reject.append("inconsistent_prediction")

        # 4. Confidence rules
        if conf < config.REJECT_CONFIDENCE:
            reject.append(f"confidence_too_low:{conf:.3f}")
        elif conf < config.ACCEPT_CONFIDENCE:
            review.append(f"confidence_below_accept_threshold:{conf:.3f}")

        # 5. Ambiguity between the two best classes
        if margin < config.MIN_MARGIN_TOP2:
            review.append(f"small_margin_top2:{margin:.3f}")

        # 6. Image quality flags from preprocessing
        for flag in preprocessing.get("quality_flags", []):
            review.append(f"quality:{flag}")

    status = REJECTED if reject else REVIEW if review else ACCEPTED
    return {
        "status": status,
        "status_fr": LABELS_FR[status],
        "reasons": reject + review,
        "checks": checks,
        "thresholds": {
            "reject_confidence": config.REJECT_CONFIDENCE,
            "accept_confidence": config.ACCEPT_CONFIDENCE,
            "min_margin_top2": config.MIN_MARGIN_TOP2,
        },
        "rules_version": config.RULES_VERSION,
    }