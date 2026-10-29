"""Off-chain storage on the local file system.

Layout: offchain_store/<CASE_ID>/<ARTIFACT_TYPE>.<ext>
storage_ref is always RELATIVE and POSIX-style (e.g. "CASE_001/PREDICTION.json")
so it can be written to the blockchain without leaking local paths.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import config

_CASE_RE = re.compile(r"^CASE_(\d{3,})$")


def _root(root) -> Path:
    r = Path(root) if root else config.OFFCHAIN_DIR
    r.mkdir(parents=True, exist_ok=True)
    return r


def next_case_id(root=None) -> str:
    nums = [int(m.group(1)) for p in _root(root).iterdir()
            if (m := _CASE_RE.match(p.name))]
    return f"CASE_{(max(nums) + 1 if nums else 1):03d}"


def store_file(case_id: str, src, artifact_type: str, root=None) -> str:
    src = Path(src)
    dest_dir = _root(root) / case_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{artifact_type}{src.suffix.lower()}"
    shutil.copyfile(src, dest)
    return f"{case_id}/{dest.name}"


def store_json(case_id: str, obj: dict, artifact_type: str, root=None) -> str:
    dest_dir = _root(root) / case_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{artifact_type}.json"
    dest.write_text(
        json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    return f"{case_id}/{dest.name}"


def resolve(storage_ref: str, root=None) -> Path:
    base = _root(root).resolve()
    p = (base / storage_ref).resolve()
    if base not in p.parents:  # block "../" tricks
        raise ValueError(f"Unsafe storage_ref: {storage_ref}")
    return p
