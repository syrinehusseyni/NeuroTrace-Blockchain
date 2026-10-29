"""Shared types for the agents and the orchestrator."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ArtifactSpec:
    """A file an agent produced; the ProvenanceAgent hashes and registers it."""
    artifact_type: str
    storage_ref: str
    model_version: str | None = None


@dataclass
class CaseContext:
    case_id: str
    image_path: Path
    root: Any
    ledger: Any
    work: Path
    on_event: Callable[[dict], None] | None = None
    stored_original: Path | None = None
    pre: dict | None = None
    model_img: Any = None
    pred: dict | None = None
    model_version: str | None = None
    val: dict | None = None
    final: dict | None = None
    parent: str | None = None            # last registered artifact id
    arts: dict = field(default_factory=dict)
    log: list = field(default_factory=list)
    failure: dict | None = None          # first failure, if any


class Skip(Exception):
    """Raised by an agent that has nothing to do for this case."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class Agent:
    name = "Agent"
    action = ""

    def run(self, ctx: CaseContext) -> tuple[list[ArtifactSpec], dict]:
        """Return (artifacts to register, detail for the action log)."""
        raise NotImplementedError
