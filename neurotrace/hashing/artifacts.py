"""Artifact records = exactly what will be written to Hyperledger Fabric.

Fields (cahier des charges): artifact_id, case_id, artifact_type, sha256,
timestamp, model_version, parent_artifact_id, storage_ref.
The image itself is NEVER part of the record (off-chain only).
"""
from __future__ import annotations

from datetime import datetime, timezone

from hashing.sha256 import sha256_file
from storage.offchain import resolve

# Provenance chain order (original -> ... -> final result)
ARTIFACT_TYPES = [
    "ORIGINAL_IMAGE", "PREPROCESSED_IMAGE", "PREDICTION",
    "VALIDATION", "EXPLANATION", "FINAL_RESULT",
]


def create_artifact(case_id: str, artifact_type: str, storage_ref: str,
                    parent_artifact_id: str | None = None,
                    model_version: str | None = None, root=None) -> dict:
    if artifact_type not in ARTIFACT_TYPES:
        raise ValueError(f"Unknown artifact type: {artifact_type}")
    return {
        "artifact_id": f"{case_id}_{artifact_type}",
        "case_id": case_id,
        "artifact_type": artifact_type,
        "sha256": sha256_file(resolve(storage_ref, root)),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_version": model_version,
        "parent_artifact_id": parent_artifact_id,
        "storage_ref": storage_ref,
    }
