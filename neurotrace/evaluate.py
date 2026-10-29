"""Evaluate the DEPLOYED model (same loader and preprocessing as the app).

    python evaluate.py --data "C:\\path\\to\\Testing"

--data must contain one sub-folder per class, named exactly like the classes
stored in the checkpoint (glioma, meningioma, notumor, pituitary).
Outputs (outputs/evaluation/): metrics.json, predictions.csv,
confusion_matrix.png, threshold_table.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import config
from evaluation.metrics import classification_report, confusion_matrix, threshold_table

EXTS = {".jpg", ".jpeg", ".png"}
THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]


def collect(data: Path, class_names: list[str]):
    folders = sorted(d.name for d in data.iterdir() if d.is_dir())
    if set(folders) != set(class_names):
        sys.exit("Class folders do not match the checkpoint.\n"
                 f"  folders found : {folders}\n  checkpoint    : {sorted(class_names)}")
    items = []
    for idx, name in enumerate(class_names):
        for p in sorted((data / name).rglob("*")):
            if p.suffix.lower() in EXTS:
                items.append((p, idx))
    if not items:
        sys.exit("No .jpg / .jpeg / .png images found.")
    return items


def run(items, bundle, batch_size=32):
    import torch
    rows, skipped = [], []
    batch, meta = [], []

    def flush():
        if not batch:
            return
        with torch.no_grad():
            probs = torch.softmax(bundle.model(torch.stack(batch).to(config.DEVICE)), dim=1).cpu().numpy()
        for (path, true_idx), pr in zip(meta, probs):
            order = np.argsort(-pr)
            rows.append({"path": str(path), "true": int(true_idx), "pred": int(order[0]),
                         "confidence": float(pr[order[0]]),
                         "margin": float(pr[order[0]] - pr[order[1]])})
        batch.clear(); meta.clear()

    for i, (path, true_idx) in enumerate(items, 1):
        try:
            with Image.open(path) as im:
                batch.append(bundle.preprocess(im.convert("RGB")))
            meta.append((path, true_idx))
        except Exception as exc:
            skipped.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"[:200]})
        if len(batch) == batch_size:
            flush()
        if i % 200 == 0:
            print(f"  {i}/{len(items)} images", flush=True)
    flush()
    return rows, skipped


def plot_cm(cm, class_names, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=30, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.set_title("Confusion matrix")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with one sub-folder per class")
    ap.add_argument("--out", default=str(config.PROJECT_DIR / "outputs" / "evaluation"))
    a = ap.parse_args()

    from ai.model_loader import load_bundle
    bundle = load_bundle()
    data, out = Path(a.data), Path(a.out)
    if not data.is_dir():
        sys.exit(f"Folder not found: {data}")
    out.mkdir(parents=True, exist_ok=True)

    items = collect(data, bundle.class_names)
    print(f"{len(items)} images, classes: {bundle.class_names}")
    rows, skipped = run(items, bundle)

    y_true = [r["true"] for r in rows]
    y_pred = [r["pred"] for r in rows]
    cm = confusion_matrix(y_true, y_pred, len(bundle.class_names))
    rep = classification_report(cm, bundle.class_names)
    rep.update(model=bundle.info["model_id"], model_version=bundle.info["model_version"],
               data_folder=str(data), skipped_images=skipped,
               confusion_matrix={"classes": bundle.class_names, "matrix": cm.tolist()})
    (out / "metrics.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")

    with open(out / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "true_class", "pred_class", "confidence", "margin_top2", "correct"])
        for r in rows:
            w.writerow([r["path"], bundle.class_names[r["true"]], bundle.class_names[r["pred"]],
                        f"{r['confidence']:.6f}", f"{r['margin']:.6f}", int(r["true"] == r["pred"])])

    correct = [r["true"] == r["pred"] for r in rows]
    table = threshold_table(correct, [r["confidence"] for r in rows],
                            [r["margin"] for r in rows], THRESHOLDS, config.MIN_MARGIN_TOP2)
    with open(out / "threshold_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader(); w.writerows(table)
    plot_cm(cm, bundle.class_names, out / "confusion_matrix.png")

    print(f"\nAccuracy : {rep['accuracy']:.4f}  ({rep['n_images']} images, {len(skipped)} skipped)")
    print(f"{'class':<12}{'precision':>10}{'recall':>9}{'f1':>8}{'support':>9}")
    for c, m in rep["per_class"].items():
        print(f"{c:<12}{m['precision']:>10.4f}{m['recall']:>9.4f}{m['f1']:>8.4f}{m['support']:>9}")
    print(f"{'macro avg':<12}{rep['macro_avg']['precision']:>10.4f}{rep['macro_avg']['recall']:>9.4f}{rep['macro_avg']['f1']:>8.4f}")
    print("\nConfusion matrix (rows = true, columns = predicted):")
    print(cm)
    print(f"\nFiles written to {out}")


if __name__ == "__main__":
    main()
