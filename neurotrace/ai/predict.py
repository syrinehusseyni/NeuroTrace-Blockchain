"""Prediction function: image in -> JSON-serialisable dict out."""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

import config
from ai.model_loader import load_bundle


def _to_rgb(image: Image.Image | str | Path) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    with Image.open(image) as src:
        return src.convert("RGB")


def predict(image: Image.Image | str | Path) -> dict:
    bundle = load_bundle()
    tensor = bundle.preprocess(_to_rgb(image)).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        probs = torch.softmax(bundle.model(tensor), dim=1)[0].cpu()

    idx = int(probs.argmax())
    ranked = probs.argsort(descending=True)
    top1, top2 = float(probs[ranked[0]]), float(probs[ranked[1]])

    return {
        "predicted_class": bundle.class_names[idx],
        "predicted_index": idx,
        "confidence": top1,
        "margin_top2": top1 - top2,  # useful for the validation module
        "probabilities": {c: float(p) for c, p in zip(bundle.class_names, probs)},
        "model_id": bundle.info["model_id"],
        "model_version": bundle.info["model_version"],
    }
