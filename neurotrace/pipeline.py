"""End-to-end pipeline (sequential). The agents will call these same steps."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import config
from ai.explain import explain
from ai.predict import predict
from hashing.artifacts import create_artifact
from hashing.sha256 import sha256_file
from ledger.factory import get_ledger
from preprocessing.preprocess import preprocess_image
from storage.offchain import next_case_id, store_file, store_json
from validation.validate import validate, REJECTED


def run_case(image_path, case_id=None, root=None, ledger=None) -> dict:
    ledger = ledger or get_ledger()
    case_id = case_id or next_case_id(root)
    parent = None
    arts = {}

    def register(artifact_type, ref, model_version=None):
        nonlocal parent
        art = ledger.record(create_artifact(
            case_id, artifact_type, ref, parent, model_version, root))
        parent = art["artifact_id"]
        arts[artifact_type] = art
        return art

    # 1. Original image (hashed BEFORE any processing)
    ref = store_file(case_id, image_path, "ORIGINAL_IMAGE", root)
    register("ORIGINAL_IMAGE", ref)
    from storage.offchain import resolve
    stored_original = resolve(ref, root)

    # 2. Preprocessing (on the stored, hashed copy)
    work = Path(tempfile.mkdtemp())
    pre, model_img = preprocess_image(stored_original, case_id, out_dir=work)
    if pre["valid"]:
        ref = store_file(case_id, pre["preprocessed_path"], "PREPROCESSED_IMAGE", root)
        register("PREPROCESSED_IMAGE", ref)

    # 3. Prediction + 4. Validation
    pred, model_version = None, None
    if pre["valid"]:
        pred = predict(model_img)
        model_version = pred["model_version"]
        register("PREDICTION", store_json(case_id, pred, "PREDICTION", root), model_version)
    val = validate(pre, pred)
    val["preprocessing"] = {k: pre.get(k) for k in
                            ("source_file", "original", "operations", "quality", "quality_flags", "errors")}
    register("VALIDATION", store_json(case_id, val, "VALIDATION", root), model_version)

    # 5. Explanation (only if not rejected)
    if pred and val["status"] != REJECTED:
        exp = explain(model_img, pred, case_id, work)
        ref = store_file(case_id, exp["overlay_path"], "EXPLANATION", root)
        register("EXPLANATION", ref, model_version)

    # 6. Final result
    final = {
        "case_id": case_id,
        "predicted_class": pred["predicted_class"] if pred else None,
        "confidence": pred["confidence"] if pred else None,
        "status": val["status"], "status_fr": val["status_fr"],
        "reasons": val["reasons"], "model_version": model_version,
        "disclaimer": "Prototype pédagogique, pas un diagnostic médical.",
    }
    register("FINAL_RESULT", store_json(case_id, final, "FINAL_RESULT", root), model_version)
    # 7. Case summary on the ledger (only when a prediction exists)
    register_case = getattr(ledger, "register_case", None)
    if register_case and pred:
        register_case(
            case_id=case_id,
            mri_hash=arts["ORIGINAL_IMAGE"]["sha256"],
            prediction_hash=arts["PREDICTION"]["sha256"],
            model_hash=sha256_file(config.CHECKPOINT_PATH),
            predicted_class=pred["predicted_class"],
            confidence=pred["confidence"],
            model_version=model_version,
            status=val["status"],
        )
    shutil.rmtree(work, ignore_errors=True)
    return {"case_id": case_id, "result": final, "artifacts": ledger.history(case_id)}
