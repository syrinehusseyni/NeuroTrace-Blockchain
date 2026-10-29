# ===================== NeuroTrace: evaluation of the deployed model (Colab) =====================
# 1) Edit the two paths below.  2) Paste this whole cell into Colab and run it.
# Same checkpoint, same preprocessing and same metrics as the app (evaluate.py on the PC).

CKPT = "/content/drive/MyDrive/PATH/TO/efficientnet_b0_best.pth"   # <-- EDIT
TESTING = "/content/PATH/TO/Testing"                               # <-- EDIT (folder with glioma/, meningioma/, notumor/, pituitary/)
EXPECTED_VERSION = "2086692686b9"   # model_version shown in the app (first 12 chars of the checkpoint SHA-256)

# from google.colab import drive; drive.mount("/content/drive")     # uncomment if the paths are on Drive

import csv, hashlib, json, os
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MARGIN_MIN, THRESHOLDS = 0.30, [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

# ---- checkpoint identity (must match the deployed model) ----
h = hashlib.sha256()
with open(CKPT, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
version = h.hexdigest()[:12]
print("checkpoint version:", version, "| expected:", EXPECTED_VERSION,
      "| OK" if version == EXPECTED_VERSION else "| !!! DIFFERENT FILE FROM THE APP !!!")

ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=True)
class_names, image_size = list(ckpt["class_names"]), int(ckpt["image_size"])
model = efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
model.load_state_dict(ckpt["model_state_dict"]); model.to(DEVICE).eval()
preprocess = transforms.Compose([
    transforms.Resize((image_size, image_size)), transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

# ---- images ----
root = Path(TESTING)
folders = sorted(d.name for d in root.iterdir() if d.is_dir())
assert set(folders) == set(class_names), f"folders {folders} != checkpoint classes {sorted(class_names)}"
items = [(p, i) for i, c in enumerate(class_names)
         for p in sorted((root / c).rglob("*")) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
print(len(items), "images,", class_names)

# ---- inference ----
rows, skipped, batch, meta = [], [], [], []
def flush():
    if not batch: return
    with torch.no_grad():
        probs = torch.softmax(model(torch.stack(batch).to(DEVICE)), dim=1).cpu().numpy()
    for (p, t), pr in zip(meta, probs):
        o = np.argsort(-pr)
        rows.append((str(p), t, int(o[0]), float(pr[o[0]]), float(pr[o[0]] - pr[o[1]])))
    batch.clear(); meta.clear()
for p, t in items:
    try:
        with Image.open(p) as im: batch.append(preprocess(im.convert("RGB")))
        meta.append((p, t))
    except Exception as e:
        skipped.append((str(p), str(e)[:100]))
    if len(batch) == 32: flush()
flush()

# ---- metrics (NumPy, same formulas as evaluation/metrics.py) ----
y_true = np.array([r[1] for r in rows]); y_pred = np.array([r[2] for r in rows])
conf = np.array([r[3] for r in rows]); marg = np.array([r[4] for r in rows])
K = len(class_names)
cm = np.zeros((K, K), dtype=int)
for t, p in zip(y_true, y_pred): cm[t, p] += 1
div = lambda a, b: float(a) / float(b) if b else 0.0
print(f"\nAccuracy: {div(np.trace(cm), cm.sum()):.4f}  ({cm.sum()} images, {len(skipped)} skipped)")
print(f"{'class':<12}{'precision':>10}{'recall':>9}{'f1':>8}{'support':>9}")
per = {}
for i, c in enumerate(class_names):
    tp = cm[i, i]; pr_ = div(tp, cm[:, i].sum()); rc = div(tp, cm[i, :].sum())
    f1 = div(2 * pr_ * rc, pr_ + rc); per[c] = dict(precision=pr_, recall=rc, f1=f1, support=int(cm[i, :].sum()))
    print(f"{c:<12}{pr_:>10.4f}{rc:>9.4f}{f1:>8.4f}{int(cm[i, :].sum()):>9}")
print(f"{'macro avg':<12}{np.mean([v['precision'] for v in per.values()]):>10.4f}"
      f"{np.mean([v['recall'] for v in per.values()]):>9.4f}{np.mean([v['f1'] for v in per.values()]):>8.4f}")
print("\nConfusion matrix (rows = true, columns = predicted):\n", cm)

# ---- threshold table ----
ok = y_true == y_pred
print(f"\n{'conf>=':>7}{'+margin':>9}{'accepted':>10}{'coverage':>10}{'acc.accepted':>14}{'errors':>8}")
table = []
for t in THRESHOLDS:
    for use_m in (False, True):
        keep = conf >= t
        if use_m: keep &= marg >= MARGIN_MIN
        k = int(keep.sum())
        r = dict(conf_min=t, margin_min=MARGIN_MIN if use_m else None, accepted=k,
                 coverage=div(k, len(ok)), acc_accepted=div(ok[keep].sum(), k), errors=int((~ok[keep]).sum()))
        table.append(r)
        print(f"{t:>7.2f}{'yes' if use_m else 'no':>9}{k:>10}{r['coverage']:>10.3f}{r['acc_accepted']:>14.4f}{r['errors']:>8}")

# ---- save + figure ----
out = Path("/content/evaluation"); out.mkdir(exist_ok=True)
with open(out / "predictions.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["path", "true_class", "pred_class", "confidence", "margin_top2", "correct"])
    for p, t, pr_, c, m in rows: w.writerow([p, class_names[t], class_names[pr_], f"{c:.6f}", f"{m:.6f}", int(t == pr_)])
json.dump({"model_version": version, "classes": class_names, "per_class": per, "accuracy": div(np.trace(cm), cm.sum()),
           "confusion_matrix": cm.tolist(), "threshold_table": table, "skipped": skipped},
          open(out / "metrics.json", "w"), indent=2)
fig, ax = plt.subplots(figsize=(6, 5)); ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(K), class_names, rotation=30, ha="right"); ax.set_yticks(range(K), class_names)
ax.set_xlabel("Predicted class"); ax.set_ylabel("True class"); ax.set_title("Confusion matrix")
for i in range(K):
    for j in range(K):
        ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=150); plt.show()
print("\nFiles in", out)
try:
    import shutil; from google.colab import files
    files.download(shutil.make_archive("/content/evaluation", "zip", out))
except Exception:
    pass
