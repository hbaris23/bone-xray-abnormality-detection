from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "outputs" / "models" / "final_balanced_resnet18.pth"
TEST_DIR = BASE_DIR / "test_images"
OUTPUT_DIR = BASE_DIR / "outputs" / "figures" / "gradcam_bbox"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ABNORMAL_THRESHOLD = 0.35
CLASS_NAMES = {0: "Normal", 1: "Abnormal"}
SELECTED_FILES = [
    "mura_1_NORMAL.png",
    "mura_2_NORMAL.png",
    "mura_3_ABNORMAL.png",
    "mura_4_ABNORMAL.png"
]
ACTIVATION_PERCENTILE = 85
MIN_COMPONENT_AREA = 2


def load_model(device):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model bulunamadı: {MODEL_PATH}")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    return model.to(device).eval()


def get_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.fh = target_layer.register_forward_hook(self.save_activation)
        self.bh = target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, inp, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class):
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0, target_class].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam).squeeze(0).detach().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam

    def close(self):
        self.fh.remove()
        self.bh.remove()


def prepare_crop(original):
    resized = original.resize((256, 256), Image.Resampling.BILINEAR)
    crop = resized.crop((16, 16, 240, 240))
    return crop


def create_xray_mask(crop):
    gray = np.array(crop.convert("L"))
    mask = np.where(gray > 12, 255, 0).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        largest = 1
        largest_area = 0
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > largest_area:
                largest_area = area
                largest = i
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)
    return mask


def find_bounding_box(heatmap, xray_mask):
    cam = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_CUBIC)
    cam = np.clip(cam, 0, 1)
    mask01 = xray_mask.astype(np.float32) / 255.0
    cam *= mask01
    values = cam[mask01 > 0]
    if values.size < 20 or float(values.max()) <= 0.05:
        return None

    threshold = np.percentile(values, ACTIVATION_PERCENTILE)
    binary = np.where(cam >= threshold, 255, 0).astype(np.uint8)
    binary = cv2.bitwise_and(binary, xray_mask)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    components = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_COMPONENT_AREA:
            continue
        component = labels == i
        score = float(cam[component].sum())
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        components.append((score, x, y, w, h))

    if not components:
        return None

    _, x, y, w, h = max(components, key=lambda z: z[0])
    px = max(4, int(w * 0.25))
    py = max(4, int(h * 0.25))
    return (max(0, x-px), max(0, y-py), min(224, x+w+px), min(224, y+h+py))


def crop_box_to_original(box, original_width, original_height):
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return (
        max(0, min(original_width-1, int((x1+16) / 256 * original_width))),
        max(0, min(original_height-1, int((y1+16) / 256 * original_height))),
        max(1, min(original_width, int((x2+16) / 256 * original_width))),
        max(1, min(original_height, int((y2+16) / 256 * original_height)))
    )


def process_image(model, gradcam, image_path, device, transform):
    print(f"\nİşleniyor: {image_path.name}")
    original = Image.open(image_path).convert("RGB")
    original_np = np.array(original)
    h, w = original_np.shape[:2]
    input_tensor = transform(original).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(input_tensor), dim=1)[0]
    normal_prob = probs[0].item()
    abnormal_prob = probs[1].item()
    predicted = 1 if abnormal_prob >= ABNORMAL_THRESHOLD else 0
    confidence = abnormal_prob if predicted else normal_prob

    print(f"Tahmin: {CLASS_NAMES[predicted]}")
    print(f"Güven: %{confidence*100:.2f}")
    print(f"Normal olasılığı: %{normal_prob*100:.2f}")
    print(f"Abnormal olasılığı: %{abnormal_prob*100:.2f}")

    heatmap = gradcam.generate(input_tensor, predicted)
    crop = prepare_crop(original)
    mask = create_xray_mask(crop)
    box_crop = find_bounding_box(heatmap, mask)
    box = crop_box_to_original(box_crop, w, h)

    plt.figure(figsize=(8, 8))
    plt.imshow(original_np, cmap="gray")
    plt.axis("off")

    if box is not None:
        x1, y1, x2, y2 = box
        plt.gca().add_patch(plt.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            fill=False, edgecolor="red", linewidth=3
        ))
        plt.text(x1, max(20, y1-10), "Modelin odaklandigi bolge",
                 fontsize=10, color="black",
                 bbox=dict(facecolor="white", alpha=0.85))
        print("Kutulu aktivasyon bulundu.")
    else:
        plt.text(10, 30, "Guvenilir aktivasyon bolgesi bulunamadi",
                 fontsize=10, color="black",
                 bbox=dict(facecolor="white", alpha=0.85))
        print("Güvenilir aktivasyon bölgesi bulunamadı.")

    plt.title(
        f"Prediction: {CLASS_NAMES[predicted]}\n"
        f"Confidence: {confidence*100:.2f}%\n"
        f"Normal: {normal_prob*100:.2f}% | Abnormal: {abnormal_prob*100:.2f}%\n"
        f"Threshold: {ABNORMAL_THRESHOLD}", fontsize=11
    )
    save_path = OUTPUT_DIR / f"{image_path.stem}_bbox.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Kutulu Grad-CAM kaydedildi:", save_path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Kullanılan cihaz:", device)
    print("Kullanılan model:", MODEL_PATH)
    print("Abnormal threshold:", ABNORMAL_THRESHOLD)
    print("Grad-CAM katmanı: model.layer3[-1]")
    print("Aktivasyon percentile:", ACTIVATION_PERCENTILE)

    model = load_model(device)
    transform = get_transform()
    gradcam = GradCAM(model, model.layer3[-1])

    image_paths = [TEST_DIR / f for f in SELECTED_FILES] if SELECTED_FILES else [
        p for p in TEST_DIR.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]
    print("Toplam görüntü sayısı:", len(image_paths))

    try:
        for image_path in image_paths:
            if not image_path.exists():
                print("UYARI: Dosya bulunamadı:", image_path)
                continue
            process_image(model, gradcam, image_path, device, transform)
    finally:
        gradcam.close()

    print("\nTüm kutulu Grad-CAM işlemleri tamamlandı.")
    print("Çıktı klasörü:", OUTPUT_DIR)
    print("NOT: Kutu, modelin karar verirken en yüksek aktivasyon gösterdiği bölgedir; tek başına lezyon/kırık tanısı değildir.")


if __name__ == "__main__":
    main()