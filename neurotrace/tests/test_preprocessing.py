import numpy as np
from PIL import Image

from preprocessing.preprocess import preprocess_image


def _save_mri(tmp_path, size=(300, 300), name="ok.png"):
    arr = np.zeros(size[::-1], dtype=np.uint8)
    h, w = arr.shape
    arr[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = np.random.randint(60, 200, (h // 2, w // 2))
    p = tmp_path / name
    Image.fromarray(arr).save(p)
    return p


def test_valid_image(tmp_path):
    res, img = preprocess_image(_save_mri(tmp_path), "CASE_T1", tmp_path)
    assert res["valid"] and img is not None
    assert (tmp_path / "CASE_T1_preprocessed.png").is_file()
    assert any(o["step"] == "resize" for o in res["operations"])


def test_invalid_file_does_not_crash(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_text("this is not an image")
    res, img = preprocess_image(bad, "CASE_T2", tmp_path)
    assert not res["valid"] and img is None
    assert "invalid_or_corrupted_image" in res["errors"]


def test_missing_file(tmp_path):
    res, img = preprocess_image(tmp_path / "nope.jpg", "CASE_T3", tmp_path)
    assert res["errors"] == ["file_not_found"]


def test_too_small(tmp_path):
    res, img = preprocess_image(_save_mri(tmp_path, (32, 32), "s.png"), "CASE_T4", tmp_path)
    assert "image_too_small" in res["errors"]
