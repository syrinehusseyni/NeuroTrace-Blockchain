from validation.validate import validate, ACCEPTED, REVIEW, REJECTED


def _pre(tmp_path, flags=()):
    f = tmp_path / "p.png"
    f.write_bytes(b"x")
    return {"valid": True, "errors": [], "quality_flags": list(flags),
            "preprocessed_path": str(f)}


def _pred(conf=0.95, margin=0.9):
    return {"predicted_class": "glioma", "confidence": conf, "margin_top2": margin,
            "probabilities": {"glioma": conf, "other": 1 - conf},
            "model_version": "abc123"}


def test_accepted(tmp_path):
    assert validate(_pre(tmp_path), _pred())["status"] == ACCEPTED


def test_low_confidence_goes_to_review(tmp_path):   # required test of the spec
    r = validate(_pre(tmp_path), _pred(conf=0.75))
    assert r["status"] == REVIEW and r["status_fr"] == "Révision requise"


def test_very_low_confidence_rejected(tmp_path):
    assert validate(_pre(tmp_path), _pred(conf=0.40, margin=0.1))["status"] == REJECTED


def test_small_margin_review(tmp_path):
    assert validate(_pre(tmp_path), _pred(conf=0.95, margin=0.1))["status"] == REVIEW


def test_quality_flag_review(tmp_path):
    assert validate(_pre(tmp_path, ["blurry"]), _pred())["status"] == REVIEW


def test_invalid_image_rejected():
    pre = {"valid": False, "errors": ["invalid_or_corrupted_image"], "quality_flags": []}
    r = validate(pre, None)
    assert r["status"] == REJECTED and r["status_fr"] == "Rejeté"