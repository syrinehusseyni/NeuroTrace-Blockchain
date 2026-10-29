"""Grad-CAM explanation (refactor of gradcam.py, no input(), model reused)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed (API / server use)
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import config
from ai.model_loader import load_bundle
from ai.predict import _to_rgb


def _compute_cam(bundle, input_tensor: torch.Tensor, target_index: int) -> np.ndarray:
    target_layer = bundle.model.features[-1]
    acts, grads = [], []
    h1 = target_layer.register_forward_hook(lambda m, i, o: acts.append(o))
    h2 = target_layer.register_full_backward_hook(lambda m, gi, go: grads.append(go[0]))
    try:
        bundle.model.zero_grad(set_to_none=True)
        logits = bundle.model(input_tensor)
        logits[:, target_index].sum().backward()

        a, g = acts[0].detach(), grads[0].detach()
        w = g.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * a).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(
            cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False
        )[0, 0].cpu().numpy()
    finally:
        h1.remove()
        h2.remove()

    lo, hi = float(cam.min()), float(cam.max())
    return (cam - lo) / (hi - lo) if hi > lo else np.zeros_like(cam)


def explain(
    image: Image.Image | str | Path,
    prediction: dict,
    case_id: str,
    out_dir: str | Path | None = None,
) -> dict:
    """Generate Grad-CAM files for an already-computed prediction.

    Returns the paths of the two files (these are the artifacts to hash).
    """
    bundle = load_bundle()
    out_dir = Path(out_dir) if out_dir else config.GRADCAM_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    original = _to_rgb(image)
    tensor = bundle.preprocess(original).unsqueeze(0).to(config.DEVICE)
    cam = _compute_cam(bundle, tensor, prediction["predicted_index"])

    heat = np.asarray(
        Image.fromarray(np.uint8(cam * 255)).resize(original.size, Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    rgb = np.asarray(original)

    heatmap_path = out_dir / f"{case_id}_heatmap.png"
    overlay_path = out_dir / f"{case_id}_gradcam.png"
    plt.imsave(heatmap_path, heat, cmap="jet")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rgb); axes[0].set_title("Original MRI")
    axes[1].imshow(heat, cmap="jet"); axes[1].set_title("Grad-CAM heatmap")
    axes[2].imshow(rgb); axes[2].imshow(heat, cmap="jet", alpha=0.4); axes[2].set_title("MRI + Grad-CAM")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Predicted: {prediction['predicted_class']} ({prediction['confidence']:.2%})")
    fig.tight_layout()
    fig.savefig(overlay_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "heatmap_path": str(heatmap_path),
        "overlay_path": str(overlay_path),
        "method": "Grad-CAM",
        "target_layer": "features[-1]",
        "disclaimer": "Interpretation aid, not proof of tumor location.",
    }
