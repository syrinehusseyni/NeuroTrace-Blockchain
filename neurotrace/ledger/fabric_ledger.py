"""Fabric-backed ledger: same interface as LocalLedger (record, get, history).

Bridge: Python (Windows) -> wsl -> peer CLI on the test-network (Org1).
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone

SCRIPT = os.environ.get("NEUROTRACE_FABRIC_SCRIPT", "")
DISTRO = os.environ.get("NEUROTRACE_WSL_DISTRO", "Ubuntu")

_FROM_CC = {
    "artifactId": "artifact_id", "caseId": "case_id", "artifactType": "artifact_type",
    "sha256": "sha256", "timestamp": "timestamp", "modelVersion": "model_version",
    "parentArtifactId": "parent_artifact_id", "storageRef": "storage_ref",
    "txId": "tx_id", "recordedAt": "recorded_at",
}


_STATUS_CC = {"ACCEPTED": "Accepted", "REVIEW_REQUIRED": "Review Required",
              "REJECTED": "Rejected"}


def _run(mode: str, function: str, args: list[str]) -> str:
    if not SCRIPT:
        raise RuntimeError("Set NEUROTRACE_FABRIC_SCRIPT to the WSL peer-wrapper path")
    ctor = json.dumps({"function": function, "Args": args})
    p = subprocess.run(
        ["wsl", "-d", DISTRO, "--", SCRIPT, mode],
        input=ctor, capture_output=True, text=True, encoding="utf-8", timeout=180,
    )
    if p.returncode != 0:
        msg = (p.stderr or p.stdout).strip()
        raise RuntimeError(msg or f"wsl call failed (exit code {p.returncode}, no output)")
    return p.stdout.strip()


def _query(function, args):
    """Queries are safe to repeat: retry once if wsl returned nothing."""
    try:
        return _run("query", function, args)
    except RuntimeError as e:
        if "no output" not in str(e):
            raise
        time.sleep(3)
        return _run("query", function, args)


def _iso_z(ts: str) -> str:
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_local(rec: dict) -> dict:
    out = {_FROM_CC[k]: v for k, v in rec.items() if k in _FROM_CC}
    for k in ("model_version", "parent_artifact_id"):
        if out.get(k) == "":
            out[k] = None
    return out


class FabricLedger:
    def record(self, artifact: dict) -> dict:
        args = [
            artifact["artifact_id"], artifact["case_id"], artifact["artifact_type"],
            artifact["sha256"], _iso_z(artifact["timestamp"]),
            artifact.get("model_version") or "",
            artifact.get("parent_artifact_id") or "",
            artifact["storage_ref"].replace("\\", "/"),
        ]
        try:
            _run("invoke", "RegisterArtifact", args)
        except RuntimeError as e:
            msg = str(e)
            if "already exists" in msg:
                raise ValueError(f"Artifact already recorded: {artifact['artifact_id']}") from e
            if "no output" not in msg:
                raise
            # wsl returned nothing: the transaction may or may not have been committed
            time.sleep(3)
            done = self.get(artifact["artifact_id"])
            if done and done["sha256"] == artifact["sha256"]:
                return artifact
            _run("invoke", "RegisterArtifact", args)
        return artifact

    def get(self, artifact_id: str) -> dict | None:
        try:
            out = _query("GetArtifact", [artifact_id])
        except RuntimeError as e:
            if "does not exist" in str(e):
                return None
            raise
        return _to_local(json.loads(out))

    def history(self, case_id: str) -> list[dict]:
        try:
            out = _query("GetCaseArtifacts", [case_id])
        except RuntimeError as e:
            if "does not exist" in str(e):
                return []
            raise
        return [_to_local(r) for r in json.loads(out or "[]")]

    def register_case(self, case_id, mri_hash, prediction_hash, model_hash,
                      predicted_class, confidence, model_version, status):
        _run("invoke", "RegisterCase", [
            case_id, mri_hash, prediction_hash, model_hash,
            str(predicted_class).lower(), str(float(confidence)),
            model_version, _STATUS_CC[status],
        ])
