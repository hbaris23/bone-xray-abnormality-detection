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

REPORT_DIR = BASE_DIR / "outputs" / "reports"

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

VALID_CSV = DATA_DIR / "valid_image_paths.csv"


# ============================================================
# FINAL THRESHOLD
# ============================================================

# 3197 validation görüntüsü üzerinde yapılan
# değerlendirmeye göre final threshold = 0.35

ABNORMAL_THRESHOLD = 0.35


# ============================================================
# SINIFLAR
# ============================================================

CLASS_NAMES = {
    0: "Normal",
    1: "Abnormal"
}


# ============================================================
# RANDOM SEED
# ============================================================

SEED = 42

random.seed(SEED)


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

        print(
            "HATA: Model dosyası bulunamadı."
        )

        print(
            "Beklenen model yolu:"
        )

        print(
            MODEL_PATH
        )

        raise FileNotFoundError(
            f"Model bulunamadı: {MODEL_PATH}"
        )


    print(
        "\nKullanılan model:"
    )

    print(
        MODEL_PATH
    )


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


    model = model.to(
        device
    )


    model.eval()


    return model


# ============================================================
# 12 ÖRNEK GÖRÜNTÜ SEÇ
# ============================================================

def select_sample_images():

    if not VALID_CSV.exists():

        raise FileNotFoundError(
            f"Validation CSV bulunamadı: {VALID_CSV}"
        )


    paths = pd.read_csv(
        VALID_CSV,
        header=None
    )[0].tolist()


    paths = [
        p
        for p in paths
        if isinstance(p, str)
        and p.strip()
    ]


    normal_paths = [
        p
        for p in paths
        if "negative" in p.lower()
    ]


    abnormal_paths = [
        p
        for p in paths
        if "positive" in p.lower()
    ]


    if len(normal_paths) < 6:

        raise ValueError(
            "Validation setinde yeterli Normal görüntü yok."
        )


    if len(abnormal_paths) < 6:

        raise ValueError(
            "Validation setinde yeterli Abnormal görüntü yok."
        )


    # 6 Normal
    selected_normal = random.sample(
        normal_paths,
        6
    )


    # 6 Abnormal
    selected_abnormal = random.sample(
        abnormal_paths,
        6
    )


    selected_paths = (
        selected_normal
        + selected_abnormal
    )


    # Görüntüleri karıştır
    random.shuffle(
        selected_paths
    )


    return selected_paths


# ============================================================
# TRANSFORM
# ============================================================

def get_transform():

    return transforms.Compose([

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


# ============================================================
# TEK GÖRÜNTİ TAHMİNİ
# ============================================================

def predict_image(
    model,
    image_path,
    device,
    transform
):

    image = Image.open(
        image_path
    ).convert("RGB")


    image_tensor = transform(
        image
    ).unsqueeze(0).to(
        device
    )


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
# SONUÇLARI GRID OLARAK KAYDET
# ============================================================

def save_prediction_grid(results):

    plt.figure(
        figsize=(14, 10)
    )


    for i, result in enumerate(
        results
    ):

        image = Image.open(
            result["image_path"]
        ).convert("L")


        true_label = (
            result["true_label_name"]
        )


        predicted_label = (
            result["predicted_label_name"]
        )


        confidence = (
            result["confidence"]
        )


        abnormal_prob = (
            result["abnormal_probability"]
        )


        status = (
            result["status"]
        )


        plt.subplot(
            3,
            4,
            i + 1
        )


        plt.imshow(
            image,
            cmap="gray"
        )


        plt.axis(
            "off"
        )


        title = (

            f"Gerçek: {true_label}\n"

            f"Tahmin: {predicted_label}\n"

            f"Güven: %{confidence:.2f}\n"

            f"Abnormal: %{abnormal_prob:.2f}\n"

            f"{status}"
        )


        plt.title(
            title,
            fontsize=8
        )


    plt.suptitle(
        "Final Model - Multiple Prediction",
        fontsize=14
    )


    plt.tight_layout()


    save_path = (
        FIGURE_DIR
        / "prediction_grid_final.png"
    )


    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        "\nÇoklu tahmin görseli kaydedildi:"
    )


    print(
        save_path
    )


# ============================================================
# CSV RAPORU
# ============================================================

def save_prediction_report(
    results
):

    report_rows = []


    for result in results:

        report_rows.append({

            "image_path":
                str(result["image_path"]),

            "true_label":
                result["true_label_name"],

            "predicted_label":
                result["predicted_label_name"],

            "confidence_percent":
                round(
                    result["confidence"],
                    2
                ),

            "normal_probability_percent":
                round(
                    result["normal_probability"],
                    2
                ),

            "abnormal_probability_percent":
                round(
                    result["abnormal_probability"],
                    2
                ),

            "threshold":
                ABNORMAL_THRESHOLD,

            "status":
                result["status"]
        })


    df = pd.DataFrame(
        report_rows
    )


    save_path = (
        REPORT_DIR
        / "prediction_results_final.csv"
    )


    df.to_csv(
        save_path,
        index=False,
        encoding="utf-8-sig"
    )


    print(
        "Çoklu tahmin CSV raporu kaydedildi:"
    )


    print(
        save_path
    )


# ============================================================
# ANA PROGRAM
# ============================================================

def main():

    # --------------------------------------------------------
    # CİHAZ
    # --------------------------------------------------------

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
        "Kullanılan model:",
        MODEL_PATH
    )


    print(
        "Kullanılan abnormal threshold:",
        ABNORMAL_THRESHOLD
    )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = load_model(
        device
    )


    # --------------------------------------------------------
    # TRANSFORM
    # --------------------------------------------------------

    transform = get_transform()


    # --------------------------------------------------------
    # GÖRÜNTÜLER
    # --------------------------------------------------------

    selected_paths = (
        select_sample_images()
    )


    print(
        "\nToplam test görüntüsü:",
        len(selected_paths)
    )


    print(
        "6 Normal + 6 Abnormal"
    )


    # --------------------------------------------------------
    # TAHMİNLER
    # --------------------------------------------------------

    results = []


    correct_count = 0


    print(
        "\n"
        + "=" * 60
    )


    print(
        "FINAL MODEL ÇOKLU TAHMİN SONUÇLARI"
    )


    print(
        "=" * 60
    )


    for path_text in selected_paths:

        image_path = get_full_path(
            path_text
        )


        true_label = (
            get_label_from_path(
                path_text
            )
        )


        (
            predicted_label,
            confidence,
            normal_probability,
            abnormal_probability
        ) = predict_image(

            model=model,

            image_path=image_path,

            device=device,

            transform=transform
        )


        is_correct = (
            true_label == predicted_label
        )


        if is_correct:

            correct_count += 1

            status = "Correct"

        else:

            status = "Wrong"


        result = {

            "image_path":
                image_path,

            "true_label":
                true_label,

            "predicted_label":
                predicted_label,

            "true_label_name":
                CLASS_NAMES[
                    true_label
                ],

            "predicted_label_name":
                CLASS_NAMES[
                    predicted_label
                ],

            "confidence":
                confidence * 100,

            "normal_probability":
                normal_probability * 100,

            "abnormal_probability":
                abnormal_probability * 100,

            "status":
                status
        }


        results.append(
            result
        )


        # ----------------------------------------------------
        # TERMINAL
        # ----------------------------------------------------

        print(
            "\nGörüntü:",
            image_path.name
        )


        print(
            "Gerçek:",
            result[
                "true_label_name"
            ]
        )


        print(
            "Tahmin:",
            result[
                "predicted_label_name"
            ]
        )


        print(
            f"Normal olasılığı: "
            f"%{result['normal_probability']:.2f}"
        )


        print(
            f"Abnormal olasılığı: "
            f"%{result['abnormal_probability']:.2f}"
        )


        print(
            f"Güven: "
            f"%{result['confidence']:.2f}"
        )


        print(
            "Threshold:",
            ABNORMAL_THRESHOLD
        )


        print(
            "Sonuç:",
            status
        )


        print(
            "-" * 40
        )


    # --------------------------------------------------------
    # GENEL SONUÇ
    # --------------------------------------------------------

    total = len(
        results
    )


    accuracy = (
        correct_count / total
    )


    print(
        "\n"
        + "=" * 60
    )


    print(
        "ÇOKLU TAHMİN ÖZETİ"
    )


    print(
        "=" * 60
    )


    print(
        f"Toplam doğru: "
        f"{correct_count}/{total}"
    )


    print(
        f"Bu örnekler üzerindeki doğruluk: "
        f"%{accuracy * 100:.2f}"
    )


    print(
        f"Kullanılan threshold: "
        f"{ABNORMAL_THRESHOLD}"
    )


    # --------------------------------------------------------
    # DOSYALARI KAYDET
    # --------------------------------------------------------

    save_prediction_grid(
        results
    )


    save_prediction_report(
        results
    )


    print(
        "\nTüm çoklu tahmin işlemleri tamamlandı."
    )


# ============================================================
# PROGRAMI ÇALIŞTIR
# ============================================================

if __name__ == "__main__":
    main()