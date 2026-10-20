
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

PREDICTION_PATH = OUTPUT_DIR / "CASE_001_prediction.json"
GRADCAM_PATH = OUTPUT_DIR / "gradcam" / "Te-gl_1_gradcam.png"

PROVENANCE_DIR = OUTPUT_DIR / "provenance"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hash of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    if not PREDICTION_PATH.is_file():
        raise FileNotFoundError(
            f"Prediction record not found: {PREDICTION_PATH}"
        )

    with PREDICTION_PATH.open("r", encoding="utf-8") as file:
        prediction = json.load(file)

    image_path = PROJECT_DIR / "Images" / prediction["image_name"]
    model_path = PROJECT_DIR / prediction["model_checkpoint"]

    for path in (image_path, model_path, GRADCAM_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    # Recompute hashes from the current files.
    current_image_hash = sha256_file(image_path)
    current_model_hash = sha256_file(model_path)
    gradcam_hash = sha256_file(GRADCAM_PATH)

    # Compare the image and model against the prediction record.
    image_matches = (
        current_image_hash == prediction["image_sha256"]
    )
    model_matches = (
        current_model_hash == prediction["model_sha256"]
    )

    integrity_status = (
        "VERIFIED" if image_matches and model_matches
        else "INTEGRITY_MISMATCH"
    )

    record = {
        "record_type": "NeuroTraceProvenance",
        "schema_version": "1.0",
        "case_id": prediction["case_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),

        "prediction": {
            "predicted_class": prediction["predicted_class"],
            "top_score": prediction["top_score"],
            "scores_by_class": prediction["scores_by_class"],
            "model_name": prediction["model_name"],
            "model_stage": prediction.get("model_stage"),
            "best_validation_macro_f1": prediction.get(
                "best_validation_macro_f1"
            )
        },

        "artifacts": {
            "image": {
                "name": image_path.name,
                "sha256": current_image_hash
            },
            "model": {
                "name": model_path.name,
                "sha256": current_model_hash
            },
            "gradcam_visualization": {
                "name": GRADCAM_PATH.name,
                "sha256": gradcam_hash
            }
        },

        "integrity_check": {
            "image_matches_prediction_record": image_matches,
            "model_matches_prediction_record": model_matches,
            "status": integrity_status
        },

        "storage_policy": {
            "image_storage": "off-chain",
            "gradcam_storage": "off-chain",
            "provenance_record_target": "Hyperledger Fabric"
        },

        "disclaimer": (
            "Research prototype only. Not a medical diagnosis. "
            "SHA-256 checks detect changes relative to recorded hashes; "
            "they do not independently prove who created a file."
        )
    }

    # Serialize consistently before calculating the record hash.
    canonical_record = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False
    )

    record_hash = hashlib.sha256(
        canonical_record.encode("utf-8")
    ).hexdigest()

    record["record_sha256"] = record_hash

    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)
    destination = PROVENANCE_DIR / f"{prediction['case_id']}_provenance.json"

    with destination.open("w", encoding="utf-8") as file:
        json.dump(record, file, indent=4, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("NEUROTRACE - PROVENANCE RECORD")
    print("=" * 50)
    print(f"Case ID:          {record['case_id']}")
    print(f"Predicted class:  {record['prediction']['predicted_class']}")
    print(f"Integrity status: {integrity_status}")
    print(f"Image matches:    {image_matches}")
    print(f"Model matches:    {model_matches}")
    print(f"Record SHA-256:   {record_hash}")
    print(f"\nSaved to:\n{destination}")
    print("=" * 50)


if __name__ == "__main__":
    main()
