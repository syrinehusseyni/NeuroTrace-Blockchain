
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    PROJECT_DIR / "models" / "efficientnet_b0" / "efficientnet_b0_best.pth"
)

OUTPUT_DIR = PROJECT_DIR / "outputs"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# 2. SHA-256 helper
# --------------------------------------------------

def sha256_file(file_path: Path) -> str:
    """Calculate the SHA-256 hash of a file."""
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


# --------------------------------------------------
# 3. Load the trained model
# --------------------------------------------------

def load_model():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    # The checkpoint must be your own trusted training file.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    required_keys = [
        "model_state_dict",
        "class_names",
        "image_size"
    ]

    missing_keys = [
        key for key in required_keys if key not in checkpoint
    ]

    if missing_keys:
        raise ValueError(
            f"Checkpoint is missing required fields: {missing_keys}"
        )

    class_names = list(checkpoint["class_names"])
    image_size = int(checkpoint["image_size"])

    if not class_names:
        raise ValueError("No class names found in the checkpoint.")

    # Recreate the same EfficientNet-B0 architecture.
    model = efficientnet_b0(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(class_names)
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    # Match the ImageNet normalization used during training.
    weights = EfficientNet_B0_Weights.DEFAULT
    imagenet_transform = weights.transforms()

    # Keep the training pipeline's resize, tensor conversion,
    # and normalization. The dataset should already be RGB.
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=imagenet_transform.mean,
            std=imagenet_transform.std
        )
    ])

    return model, class_names, image_size, preprocess, checkpoint


# --------------------------------------------------
# 4. Predict one MRI image
# --------------------------------------------------

def predict_image(image_path: str, case_id: str = "CASE_001"):
    image_file = Path(image_path).expanduser()

    if not image_file.is_file():
        raise FileNotFoundError(
            f"MRI image not found: {image_file}"
        )

    model, class_names, image_size, preprocess, checkpoint = load_model()

    # Verify that the file can be opened as an image.
    with Image.open(image_file) as test_image:
        test_image.verify()

    # Reopen it and use RGB conversion, matching the training dataset.
    with Image.open(image_file) as source_image:
        image = source_image.convert("RGB")

    image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_index = int(torch.argmax(probabilities).item())
    predicted_class = class_names[predicted_index]

    scores = {
        class_names[i]: round(float(probabilities[i].item()), 6)
        for i in range(len(class_names))
    }

    image_hash = sha256_file(image_file)
    model_hash = sha256_file(CHECKPOINT_PATH)

    record = {
        "case_id": case_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "image_name": image_file.name,
        "image_sha256": image_hash,
        "model_name": "EfficientNet-B0",
        "model_checkpoint": str(CHECKPOINT_PATH.relative_to(PROJECT_DIR)),
        "model_sha256": model_hash,
        "model_stage": checkpoint.get("stage"),
        "best_epoch_in_stage": checkpoint.get("best_epoch_in_stage"),
        "best_validation_macro_f1": checkpoint.get("best_val_macro_f1"),
        "manifest_fingerprint": checkpoint.get("manifest_fingerprint"),
        "image_size": image_size,
        "class_names": class_names,
        "predicted_class": predicted_class,
        "top_score": round(float(probabilities[predicted_index].item()), 6),
        "scores_by_class": scores,
        "device": str(DEVICE),
        "note": (
            "Research prototype only. This output is not a medical diagnosis."
        )
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    safe_case_id = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in case_id
    )

    output_file = OUTPUT_DIR / f"{safe_case_id}_prediction.json"

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(record, file, indent=4, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("NEUROTRACE - MRI CLASSIFICATION")
    print("=" * 50)
    print(f"Device:           {DEVICE}")
    print(f"Case ID:          {case_id}")
    print(f"Predicted class:  {predicted_class}")
    print(f"Top score:        {record['top_score']:.2%}")

    print("\nScores by class:")
    for class_name, score in scores.items():
        print(f"  {class_name:<20} {score:.2%}")

    print("\nIntegrity:")
    print(f"Image SHA-256: {image_hash}")
    print(f"Model SHA-256: {model_hash}")

    print(f"\nJSON record saved to:\n{output_file}")
    print("\nResearch prototype only — not a medical diagnosis.")
    print("=" * 50)

    return record


# --------------------------------------------------
# 5. Run from the terminal
# --------------------------------------------------

if __name__ == "__main__":
    print("NeuroTrace EfficientNet-B0 Inference")
    print(f"Project directory: {PROJECT_DIR}")
    print(f"Checkpoint exists: {CHECKPOINT_PATH.is_file()}")

    image_path = input(
        "\nEnter the full path to an MRI image: "
    ).strip().strip('"')

    case_id = input(
        "Enter a case ID (press Enter for CASE_001): "
    ).strip()

    if not case_id:
        case_id = "CASE_001"

    try:
        predict_image(image_path, case_id)

    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"\nError: {error}")

    except Exception as error:
        print(f"\nUnexpected error: {type(error).__name__}: {error}")
        raise
