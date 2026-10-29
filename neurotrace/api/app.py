"""NeuroTrace REST API (FastAPI).

Run (from the neurotrace folder):
    uvicorn api.app:app --host 127.0.0.1 --port 8000
Docs: http://127.0.0.1:8000/docs

Environment variables:
    NEUROTRACE_LEDGER=fabric      use Hyperledger Fabric (default: local JSON ledger)
    NEUROTRACE_API_KEY=<secret>   if set, every request needs header X-API-Key
    NEUROTRACE_DEMO_MODE=1        enables the tamper / restore demo endpoints
"""
from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import (APIRouter, Depends, FastAPI, File, Header, HTTPException,
                     UploadFile)
from fastapi import Path as PathParam
from fastapi.responses import FileResponse

import config
from agents.base import utc_now
from agents.orchestrator import Orchestrator
from ledger.factory import get_ledger
from storage.offchain import next_case_id
from verification.verify import verify_case

CASE_RE = r"^CASE_\d{3,}$"
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}
FILE_TYPES = {"ORIGINAL_IMAGE", "PREPROCESSED_IMAGE", "PREDICTION", "VALIDATION",
              "EXPLANATION", "FINAL_RESULT", "AGENT_LOG"}

CaseId = PathParam(..., pattern=CASE_RE)


# ------------------------------------------------------------------ security
def require_key(x_api_key: str | None = Header(default=None)):
    expected = os.environ.get("NEUROTRACE_API_KEY")
    if expected and not secrets.compare_digest(x_api_key or "", expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def demo_enabled() -> bool:
    return os.environ.get("NEUROTRACE_DEMO_MODE") == "1"


# ------------------------------------------------------------------ jobs
class JobStore:
    def __init__(self):
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def create(self, case_id: str):
        with self._lock:
            self._jobs[case_id] = {"case_id": case_id, "state": "queued",
                                   "current_agent": None, "log": [],
                                   "submitted_at": utc_now(), "finished_at": None,
                                   "error": None}

    def on_event(self, case_id: str):
        def handler(ev: dict):
            with self._lock:
                job = self._jobs[case_id]
                if ev["event"] == "agent_started":
                    job["state"], job["current_agent"] = "running", ev["agent"]
                elif ev["event"] == "log":
                    job["log"].append(ev["entry"])
        return handler

    def finish(self, case_id: str, result: dict):
        with self._lock:
            job = self._jobs[case_id]
            job["state"] = "done" if result["ok"] else "error"
            job["current_agent"], job["finished_at"] = None, utc_now()
            if not result["ok"]:
                job["error"] = result["failure"]

    def fail(self, case_id: str, message: str):
        with self._lock:
            job = self._jobs[case_id]
            job.update(state="error", current_agent=None, finished_at=utc_now(),
                       error={"message": message})

    def get(self, case_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(case_id)
            return json.loads(json.dumps(job)) if job else None


jobs = JobStore()
executor = ThreadPoolExecutor(max_workers=1)   # one case at a time (model + ledger)
_alloc_lock = threading.Lock()


def _case_dir(case_id: str) -> Path:
    return Path(config.OFFCHAIN_DIR) / case_id


def allocate_case_id(ledger) -> str:
    """Next free case ID: not used by a folder AND not already on the ledger."""
    config.OFFCHAIN_DIR.mkdir(parents=True, exist_ok=True)
    with _alloc_lock:
        for _ in range(50):
            cid = next_case_id()
            _case_dir(cid).mkdir(parents=True, exist_ok=True)
            try:
                taken = bool(ledger.history(cid))
            except Exception as exc:
                raise HTTPException(503, f"Ledger unreachable: {str(exc)[:150]}")
            if not taken:
                return cid
    raise HTTPException(500, "Could not allocate a case ID")


def _run_job(case_id: str, image_path: Path):
    try:
        result = Orchestrator().run_case(
            image_path, case_id=case_id, ledger=get_ledger(),
            on_event=jobs.on_event(case_id))
        jobs.finish(case_id, result)
    except Exception as exc:  # the worker must never die
        jobs.fail(case_id, f"{type(exc).__name__}: {str(exc)[:200]}")
    finally:
        shutil.rmtree(image_path.parent, ignore_errors=True)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ app
@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.environ.get("NEUROTRACE_SKIP_MODEL_LOAD") != "1":
        from ai.model_loader import load_bundle
        load_bundle()          # fail fast if the checkpoint is missing
    yield


app = FastAPI(title="NeuroTrace API", version="1.0", lifespan=lifespan)
router = APIRouter(dependencies=[Depends(require_key)])


@app.get("/health")
def health():
    return {"status": "ok",
            "ledger": os.environ.get("NEUROTRACE_LEDGER", "local").lower(),
            "demo_mode": demo_enabled(),
            "auth_required": bool(os.environ.get("NEUROTRACE_API_KEY"))}


@router.post("/cases", status_code=202)
def create_case(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, "Only .jpg, .jpeg and .png files are accepted")
    limit = config.MAX_FILE_MB * 1024 * 1024
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"File larger than {config.MAX_FILE_MB} MB")
    if not data:
        raise HTTPException(400, "Empty file")

    ledger = get_ledger()
    case_id = allocate_case_id(ledger)
    # client file name is never used: only the validated extension
    up_dir = Path(config.UPLOAD_DIR) / case_id
    up_dir.mkdir(parents=True, exist_ok=True)
    path = up_dir / f"upload{suffix}"
    path.write_bytes(data)

    jobs.create(case_id)
    executor.submit(_run_job, case_id, path)
    return {"case_id": case_id, "state": "queued"}


@router.get("/cases")
def list_cases():
    root = Path(config.OFFCHAIN_DIR)
    out = []
    if root.is_dir():
        for d in sorted(root.iterdir(), reverse=True):
            if not (d.is_dir() and re.match(CASE_RE, d.name)):
                continue
            item = {"case_id": d.name, "state": "incomplete"}
            job = jobs.get(d.name)
            final = d / "FINAL_RESULT.json"
            if final.is_file():
                r = _read_json(final)
                item.update(state="done", predicted_class=r.get("predicted_class"),
                            confidence=r.get("confidence"), status=r.get("status"),
                            status_fr=r.get("status_fr"))
            elif job:
                item["state"] = job["state"]
            out.append(item)
    return out


@router.get("/cases/{case_id}/status")
def case_status(case_id: str = CaseId):
    job = jobs.get(case_id)
    if job:
        return job
    log = _case_dir(case_id) / "AGENT_LOG.json"
    if not log.is_file():
        raise HTTPException(404, "Unknown case")
    done = (_case_dir(case_id) / "FINAL_RESULT.json").is_file()
    return {"case_id": case_id, "state": "done" if done else "incomplete",
            "current_agent": None, "log": _read_json(log)["entries"]}


@router.get("/cases/{case_id}/result")
def case_result(case_id: str = CaseId):
    final = _case_dir(case_id) / "FINAL_RESULT.json"
    if final.is_file():
        return _read_json(final)
    job = jobs.get(case_id)
    if job and job["state"] in ("queued", "running"):
        raise HTTPException(409, "Case not finished yet")
    raise HTTPException(404, "No result for this case")


@router.get("/cases/{case_id}/chain")
def case_chain(case_id: str = CaseId):
    try:
        return get_ledger().history(case_id)
    except Exception as exc:
        raise HTTPException(503, f"Ledger unreachable: {str(exc)[:150]}")


@router.get("/cases/{case_id}/files/{artifact_type}")
def case_file(case_id: str = CaseId, artifact_type: str = PathParam(...)):
    if artifact_type not in FILE_TYPES:
        raise HTTPException(404, "Unknown artifact type")
    matches = sorted(_case_dir(case_id).glob(f"{artifact_type}.*"))
    if not matches:
        raise HTTPException(404, "File not found")
    return FileResponse(matches[0])


@router.post("/cases/{case_id}/verify")
def verify(case_id: str = CaseId):
    try:
        res = verify_case(case_id, get_ledger())
    except Exception as exc:
        raise HTTPException(503, f"Ledger unreachable: {str(exc)[:150]}")
    if not res["artifacts"]:
        raise HTTPException(404, "No artifacts recorded for this case")
    return res


# ------------------------------------------------------------ tamper demo
def _backup_path(case_id: str) -> Path:
    return Path(config.OFFCHAIN_DIR) / "_demo_backups" / case_id / "PREDICTION.json"


@router.post("/cases/{case_id}/demo/tamper")
def demo_tamper(case_id: str = CaseId):
    """Deliberately modifies PREDICTION.json (off-chain). The ledger is untouched."""
    if not demo_enabled():
        raise HTTPException(403, "Demo mode is disabled (set NEUROTRACE_DEMO_MODE=1)")
    target = _case_dir(case_id) / "PREDICTION.json"
    if not target.is_file():
        raise HTTPException(404, "No PREDICTION.json for this case")
    backup = _backup_path(case_id)
    if backup.is_file():
        raise HTTPException(409, "Already tampered: restore first")
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(target, backup)

    data = _read_json(target)
    others = [c for c in data.get("probabilities", {}) if c != data.get("predicted_class")]
    data["predicted_class"] = others[0] if others else "tampered"
    target.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False),
                      encoding="utf-8")
    return {"case_id": case_id, "modified": "PREDICTION.json",
            "new_predicted_class": data["predicted_class"]}


@router.post("/cases/{case_id}/demo/restore")
def demo_restore(case_id: str = CaseId):
    if not demo_enabled():
        raise HTTPException(403, "Demo mode is disabled (set NEUROTRACE_DEMO_MODE=1)")
    backup = _backup_path(case_id)
    if not backup.is_file():
        raise HTTPException(404, "Nothing to restore")
    shutil.copyfile(backup, _case_dir(case_id) / "PREDICTION.json")
    backup.unlink()
    return {"case_id": case_id, "restored": "PREDICTION.json"}
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(Path(__file__).parent / "index.html")

app.include_router(router)
