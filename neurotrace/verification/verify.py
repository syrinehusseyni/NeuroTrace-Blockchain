"""Integrity verification: recomputed SHA-256 vs hash recorded on the ledger."""
from __future__ import annotations

from hashing.sha256 import sha256_file
from storage.offchain import resolve

OK = "INTEGRITY_VERIFIED"
TAMPERED = "MODIFICATION_DETECTED"
MISSING = "FILE_MISSING"
LABELS_FR = {OK: "Intégrité vérifiée", TAMPERED: "Modification détectée",
             MISSING: "Fichier manquant"}


def verify_artifact(record: dict, root=None) -> dict:
    try:
        path = resolve(record["storage_ref"], root)
    except ValueError:
        path = None
    if path is None or not path.is_file():
        status, actual = MISSING, None
    else:
        actual = sha256_file(path)
        status = OK if actual == record["sha256"] else TAMPERED
    return {"artifact_id": record["artifact_id"],
            "artifact_type": record["artifact_type"],
            "status": status, "status_fr": LABELS_FR[status],
            "expected_sha256": record["sha256"], "actual_sha256": actual}


def verify_case(case_id: str, ledger, root=None) -> dict:
    records = ledger.history(case_id)
    results = [verify_artifact(r, root) for r in records]

    # provenance chain: every parent must exist earlier in the same case
    seen, chain_ok = set(), True
    for r in records:
        parent = r["parent_artifact_id"]
        if parent is not None and parent not in seen:
            chain_ok = False
        seen.add(r["artifact_id"])

    all_ok = bool(results) and chain_ok and all(x["status"] == OK for x in results)
    overall = OK if all_ok else TAMPERED if results else MISSING
    return {"case_id": case_id, "overall": overall, "overall_fr": LABELS_FR[overall],
            "chain_ok": chain_ok, "artifacts": results}
