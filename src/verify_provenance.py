
import hashlib
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

PROVENANCE_PATH = (
    PROJECT_DIR / "outputs" / "provenance" / "CASE_001_provenance.json"
)
IMAGE_DIR = PROJECT_DIR / "Images"
MODEL_DIR = PROJECT_DIR / "models" / "efficientnet_b0"
GRADCAM_DIR = PROJECT_DIR / "outputs" / "gradcam"


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 hash of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def calculate_record_hash(record: dict) -> str:
    """Recalculate the record hash without its record_sha256 field."""
    content = dict(record)
    content.pop("record_sha256", None)

    canonical_json = json.dumps(
        content,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(canonical_json).hexdigest()


def verify_file(label: str, path: Path, expected_hash: str) -> bool:
    """Compare a file's current hash against the recorded hash."""
    if not path.is_file():
        print(f"[MISSING] {label}: {path}")
        return False

    actual_hash = sha256_file(path)
    matches = actual_hash == expected_hash

    status = "OK" if matches else "MISMATCH"
    print(f"[{status}] {label}")

    if not matches:
        print(f"  Expected: {expected_hash}")
        print(f"  Actual:   {actual_hash}")

    return matches


def main():
    print("=" * 55)
    print("NEUROTRACE - PROVENANCE VERIFICATION")
    print("=" * 55)

    if not PROVENANCE_PATH.is_file():
        print(f"[ERROR] Provenance record not found: {PROVENANCE_PATH}")
        return

    try:
        with PROVENANCE_PATH.open("r", encoding="utf-8") as file:
            record = json.load(file)
    except (json.JSONDecodeError, OSError) as error:
        print(f"[ERROR] Cannot read provenance record: {error}")
        return

    print(f"Case ID: {record.get('case_id', 'UNKNOWN')}")
    print()

    artifacts = record.get("artifacts", {})
    image = artifacts.get("image", {})
    model = artifacts.get("model", {})
    gradcam = artifacts.get("gradcam_visualization", {})

    image_path = IMAGE_DIR / image.get("name", "")
    model_path = MODEL_DIR / model.get("name", "")
    gradcam_path = GRADCAM_DIR / gradcam.get("name", "")

    results = []

    results.append(
        verify_file("MRI image", image_path, image.get("sha256", ""))
    )
    results.append(
        verify_file("Model checkpoint", model_path, model.get("sha256", ""))
    )
    results.append(
        verify_file(
            "Grad-CAM visualization",
            gradcam_path,
            gradcam.get("sha256", ""),
        )
    )

    print()

    stored_record_hash = record.get("record_sha256", "")
    actual_record_hash = calculate_record_hash(record)
    record_matches = (
        bool(stored_record_hash)
        and actual_record_hash == stored_record_hash
    )

    print(
        f"[{'OK' if record_matches else 'MISMATCH'}] "
        "Provenance record hash"
    )

    if not record_matches:
        print(f"  Expected: {stored_record_hash or '(missing)'}")
        print(f"  Actual:   {actual_record_hash}")

    results.append(record_matches)

    print()
    all_verified = all(results)

    if all_verified:
        print("FINAL STATUS: VERIFIED")
        print("All recorded hashes match the current files.")
    else:
        print("FINAL STATUS: INTEGRITY MISMATCH")
        print("One or more files or the provenance record differ.")

    print("=" * 55)


if __name__ == "__main__":
    main()
