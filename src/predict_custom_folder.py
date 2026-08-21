from pathlib import Path

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "outputs" / "models" / "finetuned_resnet18.pth"
TEST_IMAGE_DIR = BASE_DIR / "test_images"

FIGURE_DIR = BASE_DIR / "outputs" / "figures"
REPORT_DIR = BASE_DIR / "outputs" / "reports"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

ABNORMAL_THRESHOLD = 0.35

CLASS_NAMES = {
    0: "Normal",
    1: "Abnormal"
}


def load_model(device):
    if not MODEL_PATH.exists():
        print("HATA: Model dosyası bulunamadı.")
        print("Beklenen yol:", MODEL_PATH)
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


def get_image_files():
    allowed_extensions = [".jpg", ".jpeg", ".png"]

    image_files = [
        file for file in TEST_IMAGE_DIR.iterdir()
        if file.suffix.lower() in allowed_extensions
    ]

    return image_files


def predict_image(model, image_path, device, transform):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
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

    return predicted_label, confidence, normal_probability, abnormal_probability


def save_prediction_grid(results):
    total = len(results)

    if total == 0:
        return

    cols = 2
    rows = (total + cols - 1) // cols

    plt.figure(figsize=(10, rows * 5))

    for i, result in enumerate(results):
        image = Image.open(result["image_path"]).convert("L")

        plt.subplot(rows, cols, i + 1)
        plt.imshow(image, cmap="gray")
        plt.axis("off")

        title = (
            f"{result['filename']}\n"
            f"Prediction: {result['prediction']}\n"
            f"Confidence: {result['confidence_percent']:.2f}%\n"
            f"Normal: {result['normal_probability_percent']:.2f}% | "
            f"Abnormal: {result['abnormal_probability_percent']:.2f}%"
        )

        plt.title(title, fontsize=9)

    plt.tight_layout()

    save_path = FIGURE_DIR / "custom_folder_predictions.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Toplu tahmin görseli kaydedildi:", save_path)


def save_prediction_report(results):
    report_rows = []

    for result in results:
        report_rows.append({
            "filename": result["filename"],
            "image_path": str(result["image_path"]),
            "prediction": result["prediction"],
            "confidence_percent": round(result["confidence_percent"], 2),
            "normal_probability_percent": round(result["normal_probability_percent"], 2),
            "abnormal_probability_percent": round(result["abnormal_probability_percent"], 2),
            "threshold": ABNORMAL_THRESHOLD
        })

    df = pd.DataFrame(report_rows)

    save_path = REPORT_DIR / "custom_folder_predictions.csv"
    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print("Toplu tahmin raporu kaydedildi:", save_path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Kullanılan cihaz:", device)

    if not TEST_IMAGE_DIR.exists():
        print("HATA: test_images klasörü bulunamadı.")
        print("Beklenen klasör:", TEST_IMAGE_DIR)
        exit()

    image_files = get_image_files()

    if len(image_files) == 0:
        print("HATA: test_images klasöründe görüntü bulunamadı.")
        print("Desteklenen formatlar: .jpg, .jpeg, .png")
        exit()

    print("Bulunan görüntü sayısı:", len(image_files))

    model = load_model(device)
    transform = get_transform()

    results = []

    print("\nTahmin sonuçları:\n")

    for image_path in image_files:
        predicted_label, confidence, normal_probability, abnormal_probability = predict_image(
            model=model,
            image_path=image_path,
            device=device,
            transform=transform
        )

        prediction_name = CLASS_NAMES[predicted_label]

        result = {
            "filename": image_path.name,
            "image_path": image_path,
            "prediction": prediction_name,
            "confidence_percent": confidence * 100,
            "normal_probability_percent": normal_probability * 100,
            "abnormal_probability_percent": abnormal_probability * 100
        }

        results.append(result)

        print("Görüntü:", image_path.name)
        print("Model tahmini:", prediction_name)
        print(f"Güven oranı: %{confidence * 100:.2f}")
        print(f"Normal olasılığı: %{normal_probability * 100:.2f}")
        print(f"Abnormal olasılığı: %{abnormal_probability * 100:.2f}")
        print("-" * 40)

    print("\nUYARI: Bu sonuçlar tanı değildir. Sadece modelin ön değerlendirme tahminidir.")

    save_prediction_grid(results)
    save_prediction_report(results)


if __name__ == "__main__":
    main()