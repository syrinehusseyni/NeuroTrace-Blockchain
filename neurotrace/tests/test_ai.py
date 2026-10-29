"""Minimal tests (run from neurotrace/: python -m pytest -q)."""
import numpy as np
from PIL import Image

from ai.predict import predict
from ai.explain import explain


def _fake_mri():
    return Image.fromarray(np.random.randint(0, 255, (300, 300), dtype=np.uint8))


def test_predict_contract():
    p = predict(_fake_mri())
    assert abs(sum(p["probabilities"].values()) - 1) < 1e-4
    assert p["predicted_class"] in p["probabilities"]
    assert 0 <= p["confidence"] <= 1
    assert p["model_version"]


def test_explain_creates_files(tmp_path):
    img = _fake_mri()
    out = explain(img, predict(img), "CASE_TEST", tmp_path)
    assert (tmp_path / "CASE_TEST_gradcam.png").is_file()
    assert (tmp_path / "CASE_TEST_heatmap.png").is_file()
    assert out["method"] == "Grad-CAM"
