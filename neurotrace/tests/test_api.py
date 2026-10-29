"""API tests (model and Fabric replaced by fakes)."""
import io
import os
import time

import numpy as np
import pytest
from PIL import Image

os.environ["NEUROTRACE_SKIP_MODEL_LOAD"] = "1"
os.environ.pop("NEUROTRACE_LEDGER", None)

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import config  # noqa: E402
from agents import steps  # noqa: E402
from api import app as api_app  # noqa: E402
from ledger.local_ledger import LocalLedger  # noqa: E402
from tests.test_orchestrator import _fake_explain, _fake_pred  # noqa: E402


def _png_bytes():
    rng = np.random.default_rng(1)
    arr = np.zeros((300, 300), dtype=np.uint8)
    arr[75:225, 75:225] = rng.integers(60, 200, (150, 150))
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OFFCHAIN_DIR", tmp_path / "store")
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(config, "LEDGER_PATH", tmp_path / "ledger.json")
    ckpt = tmp_path / "c.pth"
    ckpt.write_bytes(b"x")
    monkeypatch.setattr(config, "CHECKPOINT_PATH", ckpt)
    monkeypatch.setattr(steps, "predict", _fake_pred())
    monkeypatch.setattr(steps, "explain", _fake_explain)
    monkeypatch.setenv("NEUROTRACE_DEMO_MODE", "1")
    monkeypatch.delenv("NEUROTRACE_API_KEY", raising=False)
    with TestClient(api_app.app) as c:
        yield c


def _submit(client, data=None, name="mri.png"):
    r = client.post("/cases", files={"file": (name, data or _png_bytes(), "image/png")})
    assert r.status_code == 202, r.text
    return r.json()["case_id"]


def _wait(client, cid):
    for _ in range(100):
        s = client.get(f"/cases/{cid}/status").json()
        if s["state"] in ("done", "error"):
            return s
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_full_flow_and_tamper_demo(client):
    cid = _submit(client)
    assert cid == "CASE_001"
    assert _wait(client, cid)["state"] == "done"

    res = client.get(f"/cases/{cid}/result").json()
    assert res["status"] == "ACCEPTED" and res["predicted_class"] == "glioma"
    assert len(client.get(f"/cases/{cid}/chain").json()) == 6
    assert client.get(f"/cases/{cid}/files/EXPLANATION").status_code == 200
    assert client.get(f"/cases/{cid}/files/NOPE").status_code == 404

    v = client.post(f"/cases/{cid}/verify").json()
    assert v["overall"] == "INTEGRITY_VERIFIED"

    assert client.post(f"/cases/{cid}/demo/tamper").status_code == 200
    v = client.post(f"/cases/{cid}/verify").json()
    assert v["overall"] == "MODIFICATION_DETECTED"
    assert [a["artifact_type"] for a in v["artifacts"] if a["status"] != "INTEGRITY_VERIFIED"] \
        == ["PREDICTION"]
    assert client.post(f"/cases/{cid}/demo/tamper").status_code == 409

    assert client.post(f"/cases/{cid}/demo/restore").status_code == 200
    assert client.post(f"/cases/{cid}/verify").json()["overall"] == "INTEGRITY_VERIFIED"


def test_case_ids_increment_and_listing(client):
    a, b = _submit(client), _submit(client)
    assert (a, b) == ("CASE_001", "CASE_002")
    _wait(client, a), _wait(client, b)
    listed = client.get("/cases").json()
    assert [c["case_id"] for c in listed] == ["CASE_002", "CASE_001"]
    assert listed[0]["state"] == "done"


def test_invalid_upload_is_handled(client):
    assert client.post("/cases", files={"file": ("x.exe", b"abc", "a/b")}).status_code == 400
    assert client.post("/cases", files={"file": ("x.png", b"", "a/b")}).status_code == 400
    cid = _submit(client, data=b"not an image", name="bad.jpg")   # passes upload, fails preprocessing
    assert _wait(client, cid)["state"] == "done"
    assert client.get(f"/cases/{cid}/result").json()["status"] == "REJECTED"


def test_api_key_and_demo_mode(client, monkeypatch):
    monkeypatch.setenv("NEUROTRACE_API_KEY", "secret")
    assert client.get("/health").status_code == 200
    assert client.get("/cases").status_code == 401
    assert client.get("/cases", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/cases", headers={"X-API-Key": "secret"}).status_code == 200
    monkeypatch.setenv("NEUROTRACE_DEMO_MODE", "0")
    r = client.post("/cases/CASE_001/demo/tamper", headers={"X-API-Key": "secret"})
    assert r.status_code == 403


def test_bad_case_id_and_unknown_case(client):
    assert client.get("/cases/../etc/status").status_code in (404, 422)
    assert client.get("/cases/BAD/status").status_code == 422
    assert client.get("/cases/CASE_999/status").status_code == 404
