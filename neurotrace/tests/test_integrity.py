import json
import pytest

from hashing.artifacts import create_artifact
from ledger.local_ledger import LocalLedger
from storage.offchain import next_case_id, store_json, resolve
from verification.verify import verify_case, OK, TAMPERED, MISSING


def _case(tmp_path):
    ledger = LocalLedger(tmp_path / "ledger.json")
    root = tmp_path / "store"
    cid = next_case_id(root)
    ref = store_json(cid, {"predicted_class": "glioma"}, "PREDICTION", root)
    a1 = ledger.record(create_artifact(cid, "PREDICTION", ref, None, "v1", root))
    ref2 = store_json(cid, {"status": "ACCEPTED"}, "VALIDATION", root)
    ledger.record(create_artifact(cid, "VALIDATION", ref2, a1["artifact_id"], "v1", root))
    return cid, ledger, root


def test_case_ids(tmp_path):
    assert next_case_id(tmp_path / "s") == "CASE_001"


def test_untouched_artifacts_verified(tmp_path):
    cid, ledger, root = _case(tmp_path)
    v = verify_case(cid, ledger, root)
    assert v["overall"] == OK and v["chain_ok"]


def test_modification_detected(tmp_path):                 # required test of the spec
    cid, ledger, root = _case(tmp_path)
    p = resolve(f"{cid}/PREDICTION.json", root)
    p.write_text(json.dumps({"predicted_class": "notumor"}))
    v = verify_case(cid, ledger, root)
    assert v["overall"] == TAMPERED
    assert [a["status"] for a in v["artifacts"]] == [TAMPERED, OK]


def test_missing_file(tmp_path):
    cid, ledger, root = _case(tmp_path)
    resolve(f"{cid}/VALIDATION.json", root).unlink()
    assert verify_case(cid, ledger, root)["artifacts"][1]["status"] == MISSING


def test_ledger_rejects_duplicate(tmp_path):
    cid, ledger, root = _case(tmp_path)
    ref = f"{cid}/PREDICTION.json"
    with pytest.raises(ValueError):
        ledger.record(create_artifact(cid, "PREDICTION", ref, None, "v1", root))


def test_path_traversal_blocked(tmp_path):
    with pytest.raises(ValueError):
        resolve("../secret.txt", tmp_path / "store")
