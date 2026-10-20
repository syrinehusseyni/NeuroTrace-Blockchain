
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    PROJECT_DIR / "models" / "efficientnet_b0" / "efficientnet_b0_best.pth"
)

OUTPUT_DIR = PROJECT_DIR / "outputs" / "gradcam"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_IMAGE = PROJECT_DIR / "Images" / "Te-gl_1.jpg"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# 2. Load the trained EfficientNet-B0
# --------------------------------------------------

def load_model():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    # Load only your own trusted checkpoint.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    class_names = list(checkpoint["class_names"])
    image_size = int(checkpoint["image_size"])

    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(class_names)
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    weights = EfficientNet_B0_Weights.DEFAULT
    base_transform = weights.transforms()

    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=base_transform.mean,
            std=base_transform.std
        )
    ])

    return model, class_names, preprocess


# --------------------------------------------------
# 3. Grad-CAM implementation
# --------------------------------------------------

def generate_gradcam(model, input_tensor, target_class):
    # The final convolutional block of EfficientNet-B0.
    target_layer = model.features[-1]

    activations = []
    gradients = []

    def save_activation(module, inputs, output):
        activations.append(output)

    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    activation_handle = target_layer.register_forward_hook(
        save_activation
    )
    gradient_handle = target_layer.register_full_backward_hook(
        save_gradient
    )

    try:
        model.zero_grad(set_to_none=True)

        logits = model(input_tensor)
        logits[:, target_class].sum().backward()

        activation = activations[0].detach()
        gradient = gradients[0].detach()

        # Average gradients spatially to obtain channel weights.
        weights = gradient.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of feature maps.
        cam = (weights * activation).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = torch.nn.functional.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        cam = cam[0, 0].detach().cpu().numpy()

        # Normalize the heatmap between 0 and 1.
        cam_min = float(cam.min())
        cam_max = float(cam.max())

        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, logits.detach()

    finally:
        activation_handle.remove()
        gradient_handle.remove()


# --------------------------------------------------
# 4. Run Grad-CAM and save visualization
# --------------------------------------------------

def main():
    print("NeuroTrace - Grad-CAM")
    print(f"Device: {DEVICE}")

    image_path = input(
        f"Enter MRI image path (Enter for {DEFAULT_IMAGE}): "
    ).strip().strip('"')

    if not image_path:
        image_path = str(DEFAULT_IMAGE)

    image_file = Path(image_path).expanduser()

    if not image_file.is_file():
        raise FileNotFoundError(f"Image not found: {image_file}")

    model, class_names, preprocess = load_model()

    with Image.open(image_file) as source:
        original_image = source.convert("RGB")

    input_tensor = preprocess(original_image).unsqueeze(0).to(DEVICE)

    # First obtain the predicted class.
    with torch.no_grad():
        probabilities = torch.softmax(model(input_tensor), dim=1)[0]

    predicted_index = int(probabilities.argmax().item())
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index].item())

    # Grad-CAM requires gradients, so do not wrap this call in no_grad().
    cam, _ = generate_gradcam(
        model,
        input_tensor,
        predicted_index
    )

    # Resize heatmap to the original image dimensions.
    heatmap_image = Image.fromarray(
        np.uint8(cam * 255)
    ).resize(original_image.size, Image.Resampling.BILINEAR)

    heatmap = np.asarray(heatmap_image, dtype=np.float32) / 255.0
    original = np.asarray(original_image)

    output_name = image_file.stem

    heatmap_path = OUTPUT_DIR / f"{output_name}_heatmap.png"
    overlay_path = OUTPUT_DIR / f"{output_name}_gradcam.png"

    plt.imsave(heatmap_path, heatmap, cmap="jet")

    # Save a side-by-side figure for easier interpretation.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original)
    axes[0].set_title("Original MRI")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM heatmap")

    axes[2].imshow(original)
    axes[2].imshow(heatmap, cmap="jet", alpha=0.4)
    axes[2].set_title("MRI + Grad-CAM")

    for axis in axes:
        axis.axis("off")

    fig.suptitle(
        f"Predicted: {predicted_class} ({confidence:.2%})",
        fontsize=14
    )
    fig.tight_layout()
    fig.savefig(overlay_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nPrediction")
    print(f"Class: {predicted_class}")
    print(f"Score: {confidence:.2%}")
    print(f"\nHeatmap saved: {heatmap_path}")
    print(f"Visualization saved: {overlay_path}")
    print("\nGrad-CAM is an explanatory visualization, not proof of tumor location.")


if __name__ == "__main__":
    main()
