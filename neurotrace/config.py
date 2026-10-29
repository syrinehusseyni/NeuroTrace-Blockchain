"""Central configuration for NeuroTrace (paths + device)."""
from pathlib import Path
import torch

PROJECT_DIR = Path(__file__).resolve().parent

CHECKPOINT_PATH = PROJECT_DIR / "models" / "efficientnet_b0" / "efficientnet_b0_best.pth"
GRADCAM_DIR = PROJECT_DIR / "outputs" / "gradcam"
DEFAULT_IMAGE = PROJECT_DIR.parent / "Images" / "Te-gl_1.jpg"

MODEL_ID = "efficientnet_b0"
IMAGE_SIZE = 224  # must equal checkpoint["image_size"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same normalisation as used for training (ImageNet statistics).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---------------- Preprocessing ----------------
PREPROCESSED_DIR = PROJECT_DIR / "outputs" / "preprocessed"
ALLOWED_FORMATS = {"JPEG", "PNG"}
MAX_FILE_MB = 20
MIN_SIDE_PX = 64
# Quality thresholds: PLACEHOLDERS, to calibrate on your test set.
MIN_CONTRAST_STD = 15.0      # grayscale std below -> "low_contrast"
MIN_SHARPNESS = 20.0         # variance of Laplacian below -> "blurry"
# False = the model sees the raw image (same as training). True = enhanced one.
USE_ENHANCED_FOR_MODEL = False

# ---------------- Validation rules ----------------
RULES_VERSION = "v1"
REJECT_CONFIDENCE = 0.50   # below -> Rejeté
ACCEPT_CONFIDENCE = 0.90   # below -> Révision requise
MIN_MARGIN_TOP2 = 0.30     # top1 - top2 below -> Révision requise
# PLACEHOLDERS: calibrate on your validation set (see calibrate step).

# ---------------- Off-chain storage + local ledger ----------------
OFFCHAIN_DIR = PROJECT_DIR / "offchain_store"
LEDGER_PATH = PROJECT_DIR / "ledger_local" / "ledger.json"  # stand-in until Fabric
UPLOAD_DIR = PROJECT_DIR / "uploads_tmp"
