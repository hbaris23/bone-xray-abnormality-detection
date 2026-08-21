from pathlib import Path
import random

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models

import matplotlib.pyplot as plt


# ============================================================
# PROJE YOLLARI
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "MURA-v1.1"

# FINAL BALANCED MODEL
MODEL_PATH = (
    BASE_DIR
    / "outputs"
    / "models"
    / "final_balanced_resnet18.pth"
)

FIGURE_DIR = BASE_DIR / "outputs" / "figures"
FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

VALID_CSV = DATA_DIR / "valid_image_paths.csv"


# ============================================================
# FINAL THRESHOLD
# ============================================================

# Validation setindeki 3197 görüntü üzerinde yapılan
# değerlendirme sonucunda dengeli kullanım için 0.35 seçildi.

ABNORMAL_THRESHOLD = 0.35


# ============================================================
# SINIFLAR
# ============================================================

CLASS_NAMES = {
    0: "Normal",
    1: "Abnormal"
}


# ============================================================
# DOSYA YOLUNU BUL
# ============================================================

def get_full_path(path_text):

    if path_text.startswith("MURA-v1.1"):
        return BASE_DIR / "data" / path_text

    return DATA_DIR / path_text


# ============================================================
# GERÇEK ETİKETİ BUL
# ============================================================

def get_label_from_path(path_text):

    if "positive" in path_text.lower():
        return 1

    return 0


# ============================================================
# MODELİ YÜKLE
# ============================================================

def load_model(device):

    if not MODEL_PATH.exists():

        print("HATA: Model dosyası bulunamadı.")
        print("Beklenen model yolu:")
        print(MODEL_PATH)

        raise FileNotFoundError(
            f"Model bulunamadı: {MODEL_PATH}"
        )

    print("\nKullanılan model:")
    print(MODEL_PATH)

    model = models.resnet18(
        weights=None
    )

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
# VALIDATION SETTEN ÖRNEK SEÇ
# ============================================================

def select_sample_image():

    if not VALID_CSV.exists():

        raise FileNotFoundError(
            f"Validation CSV bulunamadı: {VALID_CSV}"
        )

    paths = pd.read_csv(
        VALID_CSV,
        header=None
    )[0].tolist()

    paths = [
        p for p in paths
        if isinstance(p, str)
        and p.strip()
    ]

    if len(paths) == 0:

        raise ValueError(
            "Validation setinde görüntü bulunamadı."
        )

    # Özellikle abnormal örnek seçmeye çalışıyoruz.
    abnormal_paths = [
        p for p in paths
        if "positive" in p.lower()
    ]

    if len(abnormal_paths) > 0:

        selected_path = random.choice(
            abnormal_paths
        )

    else:

        selected_path = random.choice(
            paths
        )

    return selected_path


# ============================================================
# GÖRÜNTÜ TAHMİNİ
# ============================================================

def predict_image(
    model,
    image_path,
    device
):

    transform = transforms.Compose([

        transforms.Resize(
            (256, 256)
        ),

        transforms.CenterCrop(
            (224, 224)
        ),

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

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = transform(
        image
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        normal_probability = (
            probabilities[0][0].item()
        )

        abnormal_probability = (
            probabilities[0][1].item()
        )

    # ========================================================
    # FINAL THRESHOLD KARARI
    # ========================================================

    if abnormal_probability >= ABNORMAL_THRESHOLD:

        predicted_class = 1
        confidence = abnormal_probability

    else:

        predicted_class = 0
        confidence = normal_probability

    return (
        predicted_class,
        confidence,
        normal_probability,
        abnormal_probability
    )


# ============================================================
# SONUCU GÖRSEL OLARAK KAYDET
# ============================================================

def save_prediction_result(
    image_path,
    true_label,
    predicted_label,
    confidence,
    normal_probability,
    abnormal_probability
):

    image = Image.open(
        image_path
    ).convert("L")

    plt.figure(
        figsize=(7, 7)
    )

    plt.imshow(
        image,
        cmap="gray"
    )

    plt.axis("off")

    if true_label == predicted_label:
        result_text = "CORRECT"
    else:
        result_text = "WRONG"

    title = (

        f"Gerçek: "
        f"{CLASS_NAMES[true_label]}\n"

        f"Tahmin: "
        f"{CLASS_NAMES[predicted_label]}\n"

        f"Güven: "
        f"%{confidence * 100:.2f}\n"

        f"Normal olasılığı: "
        f"%{normal_probability * 100:.2f}\n"

        f"Abnormal olasılığı: "
        f"%{abnormal_probability * 100:.2f}\n"

        f"Threshold: "
        f"{ABNORMAL_THRESHOLD}\n"

        f"Sonuç: {result_text}"
    )

    plt.title(
        title,
        fontsize=10
    )

    save_path = (
        FIGURE_DIR
        / "prediction_result_final.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "\nTahmin görseli kaydedildi:"
    )

    print(
        save_path
    )


# ============================================================
# ANA PROGRAM
# ============================================================

def main():

    # CPU veya GPU
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Kullanılan cihaz:",
        device
    )

    print(
        "Kullanılan threshold:",
        ABNORMAL_THRESHOLD
    )

    # Model
    model = load_model(
        device
    )

    # Validation görüntüsü
    selected_path_text = (
        select_sample_image()
    )

    image_path = get_full_path(
        selected_path_text
    )

    true_label = get_label_from_path(
        selected_path_text
    )

    # Tahmin
    (
        predicted_label,
        confidence,
        normal_probability,
        abnormal_probability
    ) = predict_image(
        model,
        image_path,
        device
    )

    # ========================================================
    # TERMINAL SONUÇLARI
    # ========================================================

    print("\n" + "=" * 60)

    print(
        "Seçilen görüntü:"
    )

    print(
        image_path
    )

    print(
        "\nGerçek etiket:",
        CLASS_NAMES[true_label]
    )

    print(
        "Model tahmini:",
        CLASS_NAMES[predicted_label]
    )

    print(
        f"Güven oranı: "
        f"%{confidence * 100:.2f}"
    )

    print(
        f"\nNormal olasılığı: "
        f"%{normal_probability * 100:.2f}"
    )

    print(
        f"Abnormal olasılığı: "
        f"%{abnormal_probability * 100:.2f}"
    )

    print(
        f"Kullanılan abnormal threshold: "
        f"{ABNORMAL_THRESHOLD}"
    )

    if true_label == predicted_label:

        print(
            "\nSonuç: Model doğru tahmin yaptı."
        )

    else:

        print(
            "\nSonuç: Model yanlış tahmin yaptı."
        )

    print("=" * 60)

    # Görseli kaydet
    save_prediction_result(

        image_path=image_path,

        true_label=true_label,

        predicted_label=predicted_label,

        confidence=confidence,

        normal_probability=normal_probability,

        abnormal_probability=abnormal_probability
    )


# ============================================================
# PROGRAMI ÇALIŞTIR
# ============================================================

if __name__ == "__main__":
    main()