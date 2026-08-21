from pathlib import Path
import random

import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms, models

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


# ============================================================
# AYARLAR
# ============================================================

ImageFile.LOAD_TRUNCATED_IMAGES = True

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# PROJE YOLLARI
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "MURA-v1.1"

# ARTIK FINAL BALANCED MODELİ KULLANIYORUZ
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
# BATCH AYARI
# ============================================================

BATCH_SIZE = 16


# ============================================================
# SINIFLAR
# ============================================================

CLASS_NAMES = [
    "Normal",
    "Abnormal"
]


# ============================================================
# THRESHOLD ARALIĞI
# ============================================================

# Abnormal olasılığı bu değerin üzerindeyse
# görüntüyü Abnormal kabul edeceğiz.

THRESHOLDS = [
    round(x / 100, 2)
    for x in range(20, 71, 5)
]


# ============================================================
# DOSYA YOLU
# ============================================================

def get_full_path(path_text):

    if path_text.startswith("MURA-v1.1"):

        return BASE_DIR / "data" / path_text

    return DATA_DIR / path_text


# ============================================================
# ETİKET BULMA
# ============================================================

def get_label_from_path(path_text):

    if "positive" in path_text.lower():

        return 1

    return 0


# ============================================================
# DATASET
# ============================================================

class MURADataset(torch.utils.data.Dataset):

    def __init__(
        self,
        image_paths,
        transform=None
    ):

        self.image_paths = image_paths
        self.transform = transform


    def __len__(self):

        return len(self.image_paths)


    def __getitem__(self, idx):

        path_text = self.image_paths[idx]

        image_path = get_full_path(
            path_text
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        label = get_label_from_path(
            path_text
        )

        if self.transform:

            image = self.transform(
                image
            )

        return (
            image,
            torch.tensor(
                label,
                dtype=torch.long
            )
        )


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
# MODELİ YÜKLE
# ============================================================

def load_model(device):

    if not MODEL_PATH.exists():

        print(
            "HATA: Model bulunamadı."
        )

        print(
            "Beklenen model:"
        )

        print(
            MODEL_PATH
        )

        raise FileNotFoundError(
            MODEL_PATH
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
# TÜM VALIDATION VERİSİNİ AL
# ============================================================

def load_validation_paths():

    if not VALID_CSV.exists():

        raise FileNotFoundError(
            f"Validation CSV bulunamadı: {VALID_CSV}"
        )


    paths = pd.read_csv(
        VALID_CSV,
        header=None
    )[0].tolist()


    # Boş satırları temizle
    paths = [
        p for p in paths
        if isinstance(p, str)
        and p.strip()
    ]


    print(
        "\nValidation CSV'deki toplam görüntü:",
        len(paths)
    )


    return paths


# ============================================================
# PROBABILITY TOPLA
# ============================================================

def collect_probabilities(
    model,
    dataloader,
    device
):

    all_labels = []

    all_abnormal_probs = []


    with torch.no_grad():

        for images, labels in tqdm(
            dataloader,
            desc="Validation değerlendiriliyor"
        ):

            images = images.to(
                device
            )


            outputs = model(
                images
            )


            probabilities = torch.softmax(
                outputs,
                dim=1
            )


            # 1. sınıf = Abnormal
            abnormal_probs = (
                probabilities[:, 1]
            )


            all_labels.extend(
                labels.numpy()
            )


            all_abnormal_probs.extend(
                abnormal_probs
                .cpu()
                .numpy()
            )


    return (
        all_labels,
        all_abnormal_probs
    )


# ============================================================
# THRESHOLD DEĞERLENDİR
# ============================================================

def evaluate_thresholds(
    labels,
    abnormal_probs
):

    results = []


    for threshold in THRESHOLDS:

        predictions = [

            1
            if prob >= threshold
            else 0

            for prob in abnormal_probs
        ]


        # ----------------------------------------------------
        # ACCURACY
        # ----------------------------------------------------

        accuracy = accuracy_score(
            labels,
            predictions
        )


        # ----------------------------------------------------
        # BALANCED ACCURACY
        # ----------------------------------------------------

        balanced_accuracy = (
            balanced_accuracy_score(
                labels,
                predictions
            )
        )


        # ----------------------------------------------------
        # NORMAL RECALL
        # ----------------------------------------------------

        normal_recall = recall_score(
            labels,
            predictions,
            pos_label=0,
            zero_division=0
        )


        # ----------------------------------------------------
        # ABNORMAL RECALL
        # ----------------------------------------------------

        abnormal_recall = recall_score(
            labels,
            predictions,
            pos_label=1,
            zero_division=0
        )


        # ----------------------------------------------------
        # NORMAL PRECISION
        # ----------------------------------------------------

        normal_precision = precision_score(
            labels,
            predictions,
            pos_label=0,
            zero_division=0
        )


        # ----------------------------------------------------
        # ABNORMAL PRECISION
        # ----------------------------------------------------

        abnormal_precision = precision_score(
            labels,
            predictions,
            pos_label=1,
            zero_division=0
        )


        # ----------------------------------------------------
        # MACRO F1
        # ----------------------------------------------------

        macro_f1 = f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0
        )


        results.append({

            "threshold": threshold,

            "accuracy": accuracy,

            "balanced_accuracy":
                balanced_accuracy,

            "normal_recall":
                normal_recall,

            "abnormal_recall":
                abnormal_recall,

            "normal_precision":
                normal_precision,

            "abnormal_precision":
                abnormal_precision,

            "macro_f1":
                macro_f1
        })


    return results


# ============================================================
# SONUÇLARI KAYDET
# ============================================================

def save_threshold_results(
    results
):

    df = pd.DataFrame(
        results
    )


    csv_path = (
        REPORT_DIR
        / "final_threshold_results.csv"
    )


    txt_path = (
        REPORT_DIR
        / "final_threshold_results.txt"
    )


    df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig"
    )


    with open(
        txt_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            df.to_string(
                index=False
            )
        )


    print(
        "\nThreshold sonuçları kaydedildi:"
    )

    print(
        csv_path
    )

    print(
        txt_path
    )


# ============================================================
# THRESHOLD GRAFİĞİ
# ============================================================

def save_threshold_plot(
    results
):

    thresholds = [
        r["threshold"]
        for r in results
    ]


    accuracy = [
        r["accuracy"]
        for r in results
    ]


    balanced_accuracy = [
        r["balanced_accuracy"]
        for r in results
    ]


    normal_recall = [
        r["normal_recall"]
        for r in results
    ]


    abnormal_recall = [
        r["abnormal_recall"]
        for r in results
    ]


    macro_f1 = [
        r["macro_f1"]
        for r in results
    ]


    plt.figure(
        figsize=(10, 6)
    )


    plt.plot(
        thresholds,
        accuracy,
        marker="o",
        label="Accuracy"
    )


    plt.plot(
        thresholds,
        balanced_accuracy,
        marker="o",
        label="Balanced Accuracy"
    )


    plt.plot(
        thresholds,
        normal_recall,
        marker="o",
        label="Normal Recall"
    )


    plt.plot(
        thresholds,
        abnormal_recall,
        marker="o",
        label="Abnormal Recall"
    )


    plt.plot(
        thresholds,
        macro_f1,
        marker="o",
        label="Macro F1"
    )


    plt.xlabel(
        "Abnormal Threshold"
    )


    plt.ylabel(
        "Score"
    )


    plt.title(
        "Final Balanced Model - Threshold Comparison"
    )


    plt.legend()

    plt.grid(True)


    save_path = (
        FIGURE_DIR
        / "final_threshold_comparison.png"
    )


    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        "\nThreshold grafiği kaydedildi:"
    )

    print(
        save_path
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    labels,
    abnormal_probs,
    threshold,
    filename_suffix
):

    predictions = [

        1
        if prob >= threshold
        else 0

        for prob in abnormal_probs
    ]


    cm = confusion_matrix(
        labels,
        predictions
    )


    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES
    )


    display.plot(
        values_format="d"
    )


    plt.title(
        f"Confusion Matrix - Threshold {threshold:.2f}"
    )


    save_path = (
        FIGURE_DIR
        / f"confusion_matrix_{filename_suffix}.png"
    )


    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        "Confusion Matrix kaydedildi:"
    )

    print(
        save_path
    )


    return cm


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

def save_classification_report(
    labels,
    abnormal_probs,
    threshold,
    filename_suffix
):

    predictions = [

        1
        if prob >= threshold
        else 0

        for prob in abnormal_probs
    ]


    report = classification_report(
        labels,
        predictions,
        target_names=CLASS_NAMES,
        zero_division=0
    )


    report_path = (
        REPORT_DIR
        / f"classification_report_{filename_suffix}.txt"
    )


    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)


    print(
        "Classification report kaydedildi:"
    )

    print(
        report_path
    )


# ============================================================
# SONUÇLARI TERMİNALE YAZ
# ============================================================

def print_results(
    results
):

    print(
        "\n"
        + "=" * 90
    )

    print(
        "THRESHOLD SONUÇLARI"
    )

    print(
        "=" * 90
    )


    for result in results:

        print(

            f"Threshold: "
            f"{result['threshold']:.2f} | "

            f"Accuracy: "
            f"{result['accuracy']:.4f} | "

            f"Balanced Acc: "
            f"{result['balanced_accuracy']:.4f} | "

            f"Normal Recall: "
            f"{result['normal_recall']:.4f} | "

            f"Abnormal Recall: "
            f"{result['abnormal_recall']:.4f} | "

            f"Macro F1: "
            f"{result['macro_f1']:.4f}"
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


    # --------------------------------------------------------
    # TÜM VALIDATION PATHLERİ
    # --------------------------------------------------------

    valid_paths = (
        load_validation_paths()
    )


    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    transform = get_transform()


    valid_dataset = MURADataset(
        valid_paths,
        transform=transform
    )


    valid_loader = torch.utils.data.DataLoader(

        valid_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0
    )


    print(
        "Değerlendirilecek görüntü sayısı:",
        len(valid_dataset)
    )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = load_model(
        device
    )


    # --------------------------------------------------------
    # MODEL PROBABILITYLERİ
    # --------------------------------------------------------

    print(
        "\nModel validation seti üzerinde test ediliyor..."
    )


    labels, abnormal_probs = (
        collect_probabilities(
            model,
            valid_loader,
            device
        )
    )


    # --------------------------------------------------------
    # THRESHOLD TESTİ
    # --------------------------------------------------------

    results = evaluate_thresholds(
        labels,
        abnormal_probs
    )


    print_results(
        results
    )


    # --------------------------------------------------------
    # EN DENGELİ THRESHOLD
    # --------------------------------------------------------

    best_balanced = max(

        results,

        key=lambda x:
            (
                x["balanced_accuracy"],
                x["macro_f1"]
            )
    )


    best_balanced_threshold = (
        best_balanced["threshold"]
    )


    # --------------------------------------------------------
    # YÜKSEK ABNORMAL RECALL THRESHOLD
    # --------------------------------------------------------

    # Önceliğimiz abnormal görüntüleri
    # kaçırmamak.
    #
    # Ancak Normal Recall'ın da %70'in
    # altına düşmesine izin vermiyoruz.

    candidates = [

        r
        for r in results

        if r["normal_recall"] >= 0.70
    ]


    if len(candidates) > 0:

        best_recall = max(

            candidates,

            key=lambda x:
                (
                    x["abnormal_recall"],
                    x["macro_f1"]
                )
        )

    else:

        best_recall = max(

            results,

            key=lambda x:
                (
                    x["abnormal_recall"],
                    x["macro_f1"]
                )
        )


    best_recall_threshold = (
        best_recall["threshold"]
    )


    # --------------------------------------------------------
    # EN DENGELİ SONUÇ
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 90
    )

    print(
        "1) EN DENGELİ THRESHOLD"
    )

    print(
        "=" * 90
    )


    print(
        f"Threshold: "
        f"{best_balanced_threshold:.2f}"
    )


    print(
        f"Accuracy: "
        f"{best_balanced['accuracy']:.4f}"
    )


    print(
        f"Balanced Accuracy: "
        f"{best_balanced['balanced_accuracy']:.4f}"
    )


    print(
        f"Normal Recall: "
        f"{best_balanced['normal_recall']:.4f}"
    )


    print(
        f"Abnormal Recall: "
        f"{best_balanced['abnormal_recall']:.4f}"
    )


    print(
        f"Macro F1: "
        f"{best_balanced['macro_f1']:.4f}"
    )


    # --------------------------------------------------------
    # ABNORMAL RECALL ÖNCELİKLİ SONUÇ
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 90
    )

    print(
        "2) ABNORMAL RECALL ÖNCELİKLİ THRESHOLD"
    )

    print(
        "=" * 90
    )


    print(
        f"Threshold: "
        f"{best_recall_threshold:.2f}"
    )


    print(
        f"Accuracy: "
        f"{best_recall['accuracy']:.4f}"
    )


    print(
        f"Balanced Accuracy: "
        f"{best_recall['balanced_accuracy']:.4f}"
    )


    print(
        f"Normal Recall: "
        f"{best_recall['normal_recall']:.4f}"
    )


    print(
        f"Abnormal Recall: "
        f"{best_recall['abnormal_recall']:.4f}"
    )


    print(
        f"Macro F1: "
        f"{best_recall['macro_f1']:.4f}"
    )


    # --------------------------------------------------------
    # DOSYALARI KAYDET
    # --------------------------------------------------------

    save_threshold_results(
        results
    )


    save_threshold_plot(
        results
    )


    # --------------------------------------------------------
    # EN DENGELİ MODELİN CONFUSION MATRIX'İ
    # --------------------------------------------------------

    save_confusion_matrix(

        labels,

        abnormal_probs,

        best_balanced_threshold,

        "best_balanced"
    )


    save_classification_report(

        labels,

        abnormal_probs,

        best_balanced_threshold,

        "best_balanced"
    )


    # --------------------------------------------------------
    # ABNORMAL RECALL ÖNCELİKLİ CONFUSION MATRIX
    # --------------------------------------------------------

    save_confusion_matrix(

        labels,

        abnormal_probs,

        best_recall_threshold,

        "abnormal_recall"
    )


    save_classification_report(

        labels,

        abnormal_probs,

        best_recall_threshold,

        "abnormal_recall"
    )


    # --------------------------------------------------------
    # FİNAL ÖZET
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 90
    )

    print(
        "DEĞERLENDİRME TAMAMLANDI"
    )

    print(
        "=" * 90
    )


    print(
        "\nKullanılan model:"
    )

    print(
        MODEL_PATH
    )


    print(
        "\nValidation görüntü sayısı:",
        len(labels)
    )


    print(
        "\nEn dengeli threshold:"
    )

    print(
        f"{best_balanced_threshold:.2f}"
    )


    print(
        "\nAbnormal Recall öncelikli threshold:"
    )

    print(
        f"{best_recall_threshold:.2f}"
    )


    print(
        "\nÖNEMLİ:"
    )

    print(
        "Bu değerlendirme modelin hiçbir zaman"
    )

    print(
        "yanlış tahmin yapmayacağını garanti etmez."
    )

    print(
        "Ama hangi threshold'un bizim proje"
    )

    print(
        "amacımız için daha uygun olduğunu gösterir."
    )


# ============================================================
# PROGRAMI BAŞLAT
# ============================================================

if __name__ == "__main__":
    main()