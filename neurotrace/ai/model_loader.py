"""Load the trained EfficientNet-B0 ONCE and expose model + metadata."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0

import config


@dataclass(frozen=True)
class ModelBundle:
    model: nn.Module
    class_names: list[str]
    image_size: int
    preprocess: transforms.Compose
    info: dict  # traceability metadata (goes into the prediction artifact)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=1)
def load_bundle(checkpoint_path: str | None = None) -> ModelBundle:
    path = Path(checkpoint_path) if checkpoint_path else config.CHECKPOINT_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # The checkpoint only contains tensors, dicts, lists, str, int, float,
    # so the safe loader is enough (no arbitrary pickle execution).
    ckpt = torch.load(path, map_location=config.DEVICE, weights_only=True)

    class_names = list(ckpt["class_names"])
    image_size = int(ckpt["image_size"])

    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(config.DEVICE).eval()

    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])

    ckpt_sha = _file_sha256(path)
    info = {
        "model_id": config.MODEL_ID,
        # version = short checkpoint hash -> a different file = a different version
        "model_version": ckpt_sha[:12],
        "checkpoint_sha256": ckpt_sha,
        "training_stage": ckpt.get("stage"),
        "best_epoch_in_stage": ckpt.get("best_epoch_in_stage"),
        "best_val_macro_f1": ckpt.get("best_val_macro_f1"),
        "seed": ckpt.get("seed"),
        "manifest_fingerprint": ckpt.get("manifest_fingerprint"),
        "class_names": class_names,
        "image_size": image_size,
    }
    return ModelBundle(model, class_names, image_size, preprocess, info)
