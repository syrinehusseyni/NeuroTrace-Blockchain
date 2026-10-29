"""Local JSON ledger: STAND-IN for Hyperledger Fabric.

Same interface the Fabric client will expose later:
  record(artifact), get(artifact_id), history(case_id)
Append-only: an artifact_id can be written once (like a registered asset).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import config


class LocalLedger:
    def __init__(self, path=None):
        self.path = Path(path) if path else config.LEDGER_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> list[dict]:
        if not self.path.is_file():
            return []
        return json.loads(self.path.read_text(encoding="utf-8"))

    def record(self, artifact: dict) -> dict:
        records = self._load()
        if any(r["artifact_id"] == artifact["artifact_id"] for r in records):
            raise ValueError(f"Artifact already recorded: {artifact['artifact_id']}")
        records.append(artifact)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)  # atomic
        return artifact

    def get(self, artifact_id: str) -> dict | None:
        return next((r for r in self._load() if r["artifact_id"] == artifact_id), None)

    def history(self, case_id: str) -> list[dict]:
        return [r for r in self._load() if r["case_id"] == case_id]
