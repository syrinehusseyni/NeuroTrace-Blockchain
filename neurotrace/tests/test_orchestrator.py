"""Orchestrator tests. The model is replaced by fakes (fast, no GPU needed)."""
import json

import numpy as np
import pytest
from PIL import Image

import config
from agents import steps
from agents.orchestrator import Orchestrator
from ledger.local_ledger import LocalLedger
from verification.verify import OK, verify_case


def _fake_pred(conf=0.95, margin=0.9):
    def fn(image):
        return {"predicted_class": "glioma", "predicted_index": 0, "confidence": conf,
                "margin_top2": margin,
                "probabilities": {"glioma": conf, "notumor": 1 - conf},
                "model_id": "fake", "model_version": "fakever1"}
    return fn


def _fake_explain(image, pred, case_id, out_dir):
    p = out_dir / f"{case_id}_gradcam.png"
    Image.new("RGB", (20, 20), "red").save(p)
    return {"overlay_path": str(p), "heatmap_path": str(p), "method": "Grad-CAM"}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(steps, "predict", _fake_pred())
    monkeypatch.setattr(steps, "explain", _fake_explain)
    ckpt = tmp_path / "ckpt.pth"
    ckpt.write_bytes(b"checkpoint")
    monkeypatch.setattr(config, "CHECKPOINT_PATH", ckpt)
    rng = np.random.default_rng(0)
    arr = np.zeros((300, 300), dtype=np.uint8)
    arr[75:225, 75:225] = rng.integers(60, 200, (150, 150))
    img = tmp_path / "mri.png"
    Image.fromarray(arr).save(img)
    return {"img": img, "root": tmp_path / "store",
            "ledger": LocalLedger(tmp_path / "ledger.json"), "tmp": tmp_path}


def _run(env, **kw):
    return Orchestrator().run_case(env["img"], root=env["root"], ledger=env["ledger"], **kw)


def test_valid_case_full_chain(env):
    r = _run(env)
    assert r["ok"] and r["result"]["status"] == "ACCEPTED"
    types = [a["artifact_type"] for a in r["artifacts"]]
    assert types == ["ORIGINAL_IMAGE", "PREPROCESSED_IMAGE", "PREDICTION",
                     "VALIDATION", "EXPLANATION", "FINAL_RESULT"]
    agents = {e["agent"] for e in r["agent_log"]}
    assert {"IntakeAgent", "PreprocessingAgent", "PredictionAgent", "ValidationAgent",
            "ExplanationAgent", "FinalResultAgent", "ProvenanceAgent"} <= agents
    log_file = env["root"] / r["case_id"] / "AGENT_LOG.json"
    assert json.loads(log_file.read_text())["entries"]
    assert verify_case(r["case_id"], env["ledger"], env["root"])["overall"] == OK


def test_low_confidence_goes_to_review(env, monkeypatch):
    monkeypatch.setattr(steps, "predict", _fake_pred(conf=0.75))
    r = _run(env)
    assert r["result"]["status"] == "REVIEW_REQUIRED"
    assert "EXPLANATION" in [a["artifact_type"] for a in r["artifacts"]]


def test_rejected_prediction_has_no_explanation(env, monkeypatch):
    monkeypatch.setattr(steps, "predict", _fake_pred(conf=0.40, margin=0.1))
    r = _run(env)
    assert r["result"]["status"] == "REJECTED"
    assert "EXPLANATION" not in [a["artifact_type"] for a in r["artifacts"]]
    skipped = [e for e in r["agent_log"] if e["status"] == "skipped"]
    assert skipped and skipped[0]["agent"] == "ExplanationAgent"


def test_invalid_image_does_not_crash(env):
    bad = env["tmp"] / "bad.jpg"
    bad.write_text("not an image")
    r = Orchestrator().run_case(bad, root=env["root"], ledger=env["ledger"])
    assert r["ok"] and r["result"]["status"] == "REJECTED"
    assert [a["artifact_type"] for a in r["artifacts"]] == \
        ["ORIGINAL_IMAGE", "VALIDATION", "FINAL_RESULT"]


def test_agent_exception_is_recorded(env, monkeypatch):
    def boom(image):
        raise RuntimeError("model crashed")
    monkeypatch.setattr(steps, "predict", boom)
    r = _run(env)
    assert not r["ok"] and r["failure"]["agent"] == "PredictionAgent"
    assert r["result"]["status"] == "ERROR"
    assert [a["artifact_type"] for a in r["artifacts"]][-1] == "FINAL_RESULT"
    assert any(e["status"] == "error" for e in r["agent_log"])


def test_ledger_failure_is_handled(env):
    class FlakyLedger(LocalLedger):
        calls = 0
        def record(self, artifact):
            FlakyLedger.calls += 1
            if FlakyLedger.calls == 3:
                raise RuntimeError("peer unreachable")
            return super().record(artifact)
    led = FlakyLedger(env["tmp"] / "flaky.json")
    r = Orchestrator().run_case(env["img"], root=env["root"], ledger=led)
    assert not r["ok"] and r["failure"]["ledger"] is True
    assert r["failure"]["agent"] == "ProvenanceAgent"


def test_case_summary_registered_when_supported(env):
    class CaseLedger(LocalLedger):
        got = None
        def register_case(self, **kw):
            CaseLedger.got = kw
    led = CaseLedger(env["tmp"] / "c.json")
    r = Orchestrator().run_case(env["img"], root=env["root"], ledger=led)
    assert r["ok"] and CaseLedger.got["predicted_class"] == "glioma"
    assert CaseLedger.got["status"] == "ACCEPTED"
    assert len(CaseLedger.got["model_hash"]) == 64


def test_events_are_emitted(env):
    events = []
    _run(env, on_event=events.append)
    assert events[0]["event"] == "agent_started"
    assert any(e["event"] == "log" for e in events)
