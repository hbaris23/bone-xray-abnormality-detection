from pathlib import Path
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models

import numpy as np
import cv2
import matplotlib.pyplot as plt


# ============================================================
# AYARLAR
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "MURA-v1.1"

MODEL_PATH = (
    BASE_DIR
    / "outputs"
    / "models"
    / "final_balanced_resnet18.pth"
)

VALID_CSV = DATA_DIR / "valid_image_paths.csv"

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "figures"
    / "gradcam_class_comparison"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: "Normal",
    1: "Abnormal"
}


# ============================================================
# MODEL
# ============================================================

def load_model(device):

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model bulunamadı:\n{MODEL_PATH}"
        )

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        2
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model = model.to(device)
    model.eval()

    return model


# ============================================================
# PATH
# ============================================================

def get_full_path(path_text):

    if path_text.startswith("MURA-v1.1"):
        return BASE_DIR / "data" / path_text

    return DATA_DIR / path_text


# ============================================================
# TRANSFORM
# ============================================================

def get_transform():

    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        )
    ])


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = (
            target_layer.register_forward_hook(
                self.save_activation
            )
        )

        self.backward_handle = (
            target_layer.register_full_backward_hook(
                self.save_gradient
            )
        )

    def save_activation(
        self,
        module,
        input,
        output
    ):
        self.activations = output

    def save_gradient(
        self,
        module,
        grad_input,
        grad_output
    ):
        self.gradients = grad_output[0]

    def generate(
        self,
        input_tensor,
        target_class
    ):

        self.model.zero_grad(set_to_none=True)

        output = self.model(input_tensor)

        score = output[:, target_class].sum()

        score.backward()

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights * activations
        ).sum(dim=1)

        cam = torch.relu(cam)

        cam = cam.squeeze().detach().cpu().numpy()

        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max - cam_min > 1e-8:

            cam = (
                cam - cam_min
            ) / (
                cam_max - cam_min
            )

        else:

            cam = np.zeros_like(cam)

        return cam

    def close(self):

        self.forward_handle.remove()
        self.backward_handle.remove()


# ============================================================
# OVERLAY
# ============================================================

def create_overlay(
    original_image,
    cam
):

    image = np.array(
        original_image.convert("RGB")
    )

    h, w = image.shape[:2]

    cam_resized = cv2.resize(
        cam,
        (w, h)
    )

    heatmap = np.uint8(
        255 * cam_resized
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    overlay = (
        0.65 * image
        + 0.35 * heatmap
    )

    return np.uint8(
        np.clip(
            overlay,
            0,
            255
        )
    )


# ============================================================
# MURA'DAN İKİ ÖZEL GÖRÜNTÜYÜ BUL
# ============================================================

def find_target_images():

    paths = pd.read_csv(
        VALID_CSV,
        header=None
    )[0].tolist()

    targets = {
        "01_HAND_patient11530":
            "XR_HAND/patient11530/study1_positive/image2.png",

        "33_WRIST_patient11188":
            "XR_WRIST/patient11188/study1_positive/image1.png"
    }

    found = {}

    for name, target in targets.items():

        matches = [
            p for p in paths
            if target.lower() in p.lower()
        ]

        if not matches:

            raise FileNotFoundError(
                f"MURA görüntüsü bulunamadı:\n{target}"
            )

        found[name] = get_full_path(
            matches[0]
        )

    return found


# ============================================================
# TEK GÖRÜNTÜ - İKİ SINIF GRAD-CAM
# ============================================================

def analyze_image(
    model,
    gradcam,
    image_path,
    transform,
    device,
    output_name
):

    original = Image.open(
        image_path
    ).convert("RGB")

    tensor = transform(
        original
    ).unsqueeze(0).to(device)

    # --------------------------------------------------------
    # Model tahmini
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

    normal_prob = probabilities[0].item()
    abnormal_prob = probabilities[1].item()

    predicted_class = (
        1
        if abnormal_prob >= 0.35
        else 0
    )

    print("\n" + "=" * 70)
    print(f"Görüntü: {image_path}")
    print(
        f"Normal olasılığı   : "
        f"%{normal_prob * 100:.2f}"
    )
    print(
        f"Abnormal olasılığı : "
        f"%{abnormal_prob * 100:.2f}"
    )
    print(
        f"Model tahmini      : "
        f"{CLASS_NAMES[predicted_class]}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # NORMAL Grad-CAM
    # --------------------------------------------------------

    normal_cam = gradcam.generate(
        tensor,
        target_class=0
    )

    normal_overlay = create_overlay(
        original,
        normal_cam
    )

    # --------------------------------------------------------
    # ABNORMAL Grad-CAM
    # --------------------------------------------------------

    abnormal_cam = gradcam.generate(
        tensor,
        target_class=1
    )

    abnormal_overlay = create_overlay(
        original,
        abnormal_cam
    )

    # --------------------------------------------------------
    # Üçlü görsel
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(18, 6)
    )

    ax1 = plt.subplot(1, 3, 1)

    ax1.imshow(original)

    ax1.axis("off")

    ax1.set_title(
        "Original X-ray",
        fontsize=13
    )

    ax2 = plt.subplot(1, 3, 2)

    ax2.imshow(normal_overlay)

    ax2.axis("off")

    ax2.set_title(
        f"Grad-CAM → NORMAL\n"
        f"Normal: %{normal_prob * 100:.2f}",
        fontsize=13
    )

    ax3 = plt.subplot(1, 3, 3)

    ax3.imshow(abnormal_overlay)

    ax3.axis("off")

    ax3.set_title(
        f"Grad-CAM → ABNORMAL\n"
        f"Abnormal: %{abnormal_prob * 100:.2f}",
        fontsize=13
    )

    fig.suptitle(
        f"{output_name}\n"
        f"Model prediction: "
        f"{CLASS_NAMES[predicted_class]}",
        fontsize=15
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{output_name}_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # Ayrı overlay dosyaları da kaydet
    Image.fromarray(
        normal_overlay
    ).save(
        OUTPUT_DIR
        / f"{output_name}_normal_gradcam.png"
    )

    Image.fromarray(
        abnormal_overlay
    ).save(
        OUTPUT_DIR
        / f"{output_name}_abnormal_gradcam.png"
    )

    print(
        "\nKaydedildi:"
    )
    print(output_path)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "MURA - NORMAL / ABNORMAL ÇİFT GRAD-CAM ANALİZİ"
    )
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDevice: {device}")
    print(f"Model: {MODEL_PATH}")
    print("Grad-CAM katmanı: model.layer4[-1]")

    model = load_model(
        device
    )

    target_layer = model.layer4[-1]

    gradcam = GradCAM(
        model,
        target_layer
    )

    transform = get_transform()

    try:

        targets = find_target_images()

        print("\nBulunan hedef görüntüler:")

        for name, path in targets.items():

            print(
                f"\n{name}"
            )

            print(path)

        for name, image_path in targets.items():

            analyze_image(
                model=model,
                gradcam=gradcam,
                image_path=image_path,
                transform=transform,
                device=device,
                output_name=name
            )

    finally:

        gradcam.close()

    print("\n" + "=" * 70)
    print("ÇİFT GRAD-CAM ANALİZİ TAMAMLANDI")
    print("=" * 70)
    print(
        f"\nÇıktı klasörü:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()