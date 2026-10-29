"""Preprocessing module (PIL + NumPy only).

preprocess_image(path, case_id) -> (result_dict, model_image)
 - result_dict is JSON-serialisable (this is the preprocessing artifact summary)
 - model_image is the PIL image to give to ai.predict.predict (or None if invalid)
Never raises on a bad image: returns valid=False with the reason.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError

import config


def _validate(path: Path) -> tuple[Image.Image | None, list[str]]:
    errors: list[str] = []
    if not path.is_file():
        return None, ["file_not_found"]
    if path.stat().st_size > config.MAX_FILE_MB * 1024 * 1024:
        return None, ["file_too_large"]
    try:
        with Image.open(path) as probe:
            probe.verify()  # detects truncated / corrupted files
        img = Image.open(path)
        img.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        return None, ["invalid_or_corrupted_image"]

    if img.format not in config.ALLOWED_FORMATS:
        errors.append(f"unsupported_format:{img.format}")
    if min(img.size) < config.MIN_SIDE_PX:
        errors.append("image_too_small")
    return (None if errors else img), errors


def _brain_crop_box(gray: np.ndarray, margin: float = 0.02):
    """Bounding box of the non-black area (removes empty borders)."""
    thr = max(10, int(gray.mean() * 0.25))
    ys, xs = np.where(gray > thr)
    if len(xs) == 0:
        return None
    h, w = gray.shape
    mx, my = int(w * margin), int(h * margin)
    x0, x1 = max(xs.min() - mx, 0), min(xs.max() + mx + 1, w)
    y0, y1 = max(ys.min() - my, 0), min(ys.max() + my + 1, h)
    # ignore degenerate crops (would destroy the image)
    if (x1 - x0) < 0.3 * w or (y1 - y0) < 0.3 * h:
        return None
    return int(x0), int(y0), int(x1), int(y1)


def _sharpness(gray: np.ndarray) -> float:
    """Variance of a 4-neighbour Laplacian (higher = sharper)."""
    g = gray.astype(np.float32)
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1]
           + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.var())


def preprocess_image(path: str | Path, case_id: str,
                     out_dir: str | Path | None = None):
    path = Path(path)
    result: dict = {
        "case_id": case_id,
        "source_file": path.name,
        "valid": False,
        "errors": [],
        "quality_flags": [],
        "operations": [],
        "preprocessed_path": None,
        "model_input": "enhanced" if config.USE_ENHANCED_FOR_MODEL else "raw",
    }

    img, errors = _validate(path)
    if img is None:
        result["errors"] = errors
        return result, None

    rgb = img.convert("RGB")
    result["original"] = {"format": img.format, "mode": img.mode,
                          "width": rgb.width, "height": rgb.height}
    result["operations"].append({"step": "format_check", "ok": True})

    gray = np.asarray(rgb.convert("L"))
    box = _brain_crop_box(gray)
    work = rgb
    if box:
        work = rgb.crop(box)
        result["operations"].append({"step": "crop_borders", "box": list(box)})
    else:
        result["operations"].append({"step": "crop_borders", "skipped": True})

    work = ImageOps.autocontrast(work, cutoff=1)
    result["operations"].append({"step": "autocontrast", "cutoff_percent": 1})

    work = work.filter(ImageFilter.MedianFilter(size=3))
    result["operations"].append({"step": "denoise_median", "kernel": 3})

    size = config.IMAGE_SIZE
    work = work.resize((size, size), Image.Resampling.BILINEAR)
    result["operations"].append({"step": "resize", "size": [size, size]})
    result["operations"].append({
        "step": "normalization_at_inference",
        "mean": list(config.IMAGENET_MEAN), "std": list(config.IMAGENET_STD)})

    # Quality metrics on the ORIGINAL grayscale image
    q = {"mean_intensity": float(gray.mean()),
         "contrast_std": float(gray.std()),
         "sharpness": _sharpness(gray)}
    result["quality"] = q
    if q["contrast_std"] < config.MIN_CONTRAST_STD:
        result["quality_flags"].append("low_contrast")
    if q["sharpness"] < config.MIN_SHARPNESS:
        result["quality_flags"].append("blurry")

    out_dir = Path(out_dir) if out_dir else config.PREPROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{case_id}_preprocessed.png"
    work.save(out_path)  # lossless: the file is hashed later, must not change
    result["preprocessed_path"] = str(out_path)
    result["valid"] = True

    model_image = work if config.USE_ENHANCED_FOR_MODEL else rgb
    return result, model_image
