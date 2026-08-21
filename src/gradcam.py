from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "MURA-v1.1"

MODEL_PATH = BASE_DIR / "outputs" / "models" / "finetuned_resnet18.pth"

FIGURE_DIR = BASE_DIR / "outputs" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

VALID_CSV = DATA_DIR / "valid_image_paths.csv"

ABNORMAL_THRESHOLD = 0.50

CLASS_NAMES = {
    0: "Normal",
    1: "Abnormal"
}


def get_full_path(path_text):
    if path_text.startswith("MURA-v1.1"):
        return BASE_DIR / "data" / path_text
    return DATA_DIR / path_text


def get_label_from_path(path_text):
    if "positive" in path_text.lower():
        return 1
    return 0


def load_model(device):
    if not MODEL_PATH.exists():
        print("HATA: Model dosyası bulunamadı.")
        print("Beklenen model yolu:", MODEL_PATH)
        print("Önce train_finetune.py dosyasını çalıştırman gerekiyor.")
        exit()

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def select_abnormal_sample():
    paths = pd.read_csv(VALID_CSV, header=None)[0].tolist()

    abnormal_paths = [p for p in paths if "positive" in p.lower()]

    selected_path = random.choice(abnormal_paths)

    return selected_path


def predict_image(model, image_tensor, device):
    image_tensor = image_tensor.unsqueeze(0).to(device)

    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1)

    normal_probability = probabilities[0][0].item()
    abnormal_probability = probabilities[0][1].item()

    if abnormal_probability >= ABNORMAL_THRESHOLD:
        predicted_label = 1
        confidence = abnormal_probability
    else:
        predicted_label = 0
        confidence = normal_probability

    return outputs, predicted_label, confidence, normal_probability, abnormal_probability


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self.forward_hook = self.target_layer.register_forward_hook(
            self.save_activations
        )

        self.backward_hook = self.target_layer.register_full_backward_hook(
            self.save_gradients
        )

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class):
        self.model.zero_grad()

        output = self.model(input_tensor)

        score = output[:, target_class]
        score.backward()

        gradients = self.gradients.detach()
        activations = self.activations.detach()

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

        cam = torch.sum(weights * activations, dim=1)
        cam = torch.relu(cam)

        cam = cam.squeeze().cpu().numpy()

        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)

        return cam

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


def create_gradcam_overlay(image_path, cam):
    original_image = Image.open(image_path).convert("RGB")
    original_image = original_image.resize((224, 224))

    original_np = np.array(original_image)

    cam_resized = cv2.resize(cam, (224, 224))
    heatmap = np.uint8(255 * cam_resized)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = heatmap * 0.4 + original_np * 0.6
    overlay = np.uint8(overlay)

    return original_np, heatmap, overlay


def save_gradcam_result(
    original_np,
    heatmap,
    overlay,
    true_label,
    predicted_label,
    confidence,
    normal_probability,
    abnormal_probability
):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original_np)
    plt.title("Original X-ray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title("Grad-CAM Heatmap")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title(
        f"True: {CLASS_NAMES[true_label]}\n"
        f"Pred: {CLASS_NAMES[predicted_label]}\n"
        f"Conf: {confidence * 100:.2f}%\n"
        f"Normal: {normal_probability * 100:.2f}% | "
        f"Abnormal: {abnormal_probability * 100:.2f}%"
    )
    plt.axis("off")

    plt.tight_layout()

    save_path = FIGURE_DIR / "gradcam_result.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Grad-CAM görseli kaydedildi:", save_path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Kullanılan cihaz:", device)

    model = load_model(device)
    transform = get_transform()

    selected_path_text = select_abnormal_sample()
    image_path = get_full_path(selected_path_text)
    true_label = get_label_from_path(selected_path_text)

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image)

    input_tensor = image_tensor.unsqueeze(0).to(device)

    outputs, predicted_label, confidence, normal_probability, abnormal_probability = predict_image(
        model,
        image_tensor,
        device
    )

    print("\nSeçilen görüntü:")
    print(image_path)

    print("\nGerçek etiket:", CLASS_NAMES[true_label])
    print("Model tahmini:", CLASS_NAMES[predicted_label])
    print(f"Güven oranı: %{confidence * 100:.2f}")
    print(f"Normal olasılığı: %{normal_probability * 100:.2f}")
    print(f"Abnormal olasılığı: %{abnormal_probability * 100:.2f}")

    if true_label == predicted_label:
        print("Sonuç: Model doğru tahmin yaptı.")
    else:
        print("Sonuç: Model yanlış tahmin yaptı.")

    target_layer = model.layer4[-1]
    gradcam = GradCAM(model, target_layer)

    cam = gradcam.generate(
        input_tensor=input_tensor,
        target_class=predicted_label
    )

    gradcam.remove_hooks()

    original_np, heatmap, overlay = create_gradcam_overlay(
        image_path=image_path,
        cam=cam
    )

    save_gradcam_result(
        original_np=original_np,
        heatmap=heatmap,
        overlay=overlay,
        true_label=true_label,
        predicted_label=predicted_label,
        confidence=confidence,
        normal_probability=normal_probability,
        abnormal_probability=abnormal_probability
    )


if __name__ == "__main__":
    main()