from pathlib import Path
import random

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

BASE_DIR = Path(__file__).resolve().parents[2]

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
    / "gradcam_mura_40"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# MURA 40 sonuç raporları
REPORT_DIR = BASE_DIR / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# 20 Normal + 20 Abnormal
NUM_NORMAL = 20
NUM_ABNORMAL = 20

# Bizim seçtiğimiz threshold
ABNORMAL_THRESHOLD = 0.35

# Grad-CAM için son convolution katmanı
# Önceki denemede layer4 kullanıyoruz.
GRADCAM_LAYER = "layer4"

SEED = 42

CLASS_NAMES = {
    0: "Normal",
    1: "Abnormal"
}


# ============================================================
# SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


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
# LABEL
# ============================================================

def get_label(path_text):

    if "positive" in path_text.lower():

        return 1

    return 0


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
# MURA'DAN 20 + 20 ÖRNEK SEÇ
# ============================================================

def select_mura_samples():

    print("\nMURA validation CSV okunuyor...")

    paths = pd.read_csv(
        VALID_CSV,
        header=None
    )[0].tolist()

    normal_paths = [
        p for p in paths
        if "negative" in p.lower()
    ]

    abnormal_paths = [
        p for p in paths
        if "positive" in p.lower()
    ]

    print(
        f"Validation Normal görüntü: "
        f"{len(normal_paths)}"
    )

    print(
        f"Validation Abnormal görüntü: "
        f"{len(abnormal_paths)}"
    )

    if len(normal_paths) < NUM_NORMAL:

        raise ValueError(
            "Yeterli Normal görüntü yok."
        )

    if len(abnormal_paths) < NUM_ABNORMAL:

        raise ValueError(
            "Yeterli Abnormal görüntü yok."
        )

    selected_normal = random.sample(
        normal_paths,
        NUM_NORMAL
    )

    selected_abnormal = random.sample(
        abnormal_paths,
        NUM_ABNORMAL
    )

    selected = (
        selected_normal
        + selected_abnormal
    )

    random.shuffle(selected)

    return selected


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(
            self.save_activation
        )

        self.backward_handle = target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def save_activation(
        self,
        module,
        input,
        output
    ):

        self.activations = output.detach()

    def save_gradient(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor,
        target_class
    ):

        self.model.zero_grad()

        output = self.model(input_tensor)

        score = output[:, target_class]

        score.backward()

        gradients = self.gradients
        activations = self.activations

        # Global Average Pooling
        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights * activations
        ).sum(
            dim=1
        )

        cam = torch.relu(cam)

        cam = cam.squeeze().cpu().numpy()

        # Normalize
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
# IMAGE ÜZERİNE GRAD-CAM
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

    overlay = np.uint8(
        np.clip(
            overlay,
            0,
            255
        )
    )

    return overlay


# ============================================================
# TEK GÖRÜNTÜ İŞLE
# ============================================================

def process_image(
    model,
    gradcam,
    image_path,
    device,
    transform
):

    original_image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = transform(
        original_image
    ).unsqueeze(0).to(device)

    # --------------------------------------------------------
    # Tahmin
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

    normal_prob = probabilities[0][0].item()

    abnormal_prob = probabilities[0][1].item()

    # Threshold
    if abnormal_prob >= ABNORMAL_THRESHOLD:

        predicted_class = 1

        confidence = abnormal_prob

    else:

        predicted_class = 0

        confidence = normal_prob

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    cam = gradcam.generate(
        image_tensor,
        predicted_class
    )

    overlay = create_overlay(
        original_image,
        cam
    )

    return (
        predicted_class,
        confidence,
        normal_prob,
        abnormal_prob,
        overlay
    )


# ============================================================
# SONUÇ GÖRSELİ
# ============================================================

def save_result(
    image_path,
    true_class,
    predicted_class,
    confidence,
    normal_prob,
    abnormal_prob,
    overlay
):

    original = Image.open(
        image_path
    ).convert("RGB")

    fig = plt.figure(
        figsize=(12, 5)
    )

    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    ax1 = plt.subplot(
        1,
        2,
        1
    )

    ax1.imshow(original)

    ax1.axis("off")

    ax1.set_title(
        "Original X-ray"
    )

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    ax2 = plt.subplot(
        1,
        2,
        2
    )

    ax2.imshow(overlay)

    ax2.axis("off")

    ax2.set_title(
        "Grad-CAM - Layer4"
    )

    # --------------------------------------------------------
    # Başlık
    # --------------------------------------------------------

    status = (
        "CORRECT"
        if true_class == predicted_class
        else "WRONG"
    )

    fig.suptitle(

        f"True: {CLASS_NAMES[true_class]} | "
        f"Prediction: {CLASS_NAMES[predicted_class]} | "
        f"{status}\n"
        f"Confidence: {confidence * 100:.2f}% | "
        f"Normal: {normal_prob * 100:.2f}% | "
        f"Abnormal: {abnormal_prob * 100:.2f}%",

        fontsize=12
    )

    plt.tight_layout()

    output_name = (
        image_path.stem
        + "_gradcam.png"
    )

    output_path = (
        OUTPUT_DIR
        / output_name
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    return output_path


# ============================================================
# SUMMARY
# ============================================================

def save_summary(results):

    if len(results) == 0:

        return

    fig = plt.figure(
        figsize=(16, 20)
    )

    for i, result in enumerate(results):

        ax = plt.subplot(
            5,
            8,
            i + 1
        )

        ax.imshow(
            result["overlay"]
        )

        ax.axis("off")

        status = (
            "✓"
            if result["correct"]
            else "✗"
        )

        ax.set_title(

            f"{result['true']}\n"
            f"{result['predicted']} {status}\n"
            f"{result['confidence']:.1f}%",

            fontsize=7
        )

    plt.tight_layout()

    summary_path = (
        OUTPUT_DIR
        / "mura_40_summary.png"
    )

    plt.savefig(
        summary_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "\nToplu özet kaydedildi:"
    )

    print(summary_path)


# ============================================================
# MAIN
# ============================================================

def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)

    print(
        "MURA 40 GÖRÜNTÜ GRAD-CAM ANALİZİ"
    )

    print("=" * 70)

    print(
        "\nKullanılan model:"
    )

    print(MODEL_PATH)

    print(
        f"\nDevice: {device}"
    )

    print(
        f"Threshold: {ABNORMAL_THRESHOLD}"
    )

    print(
        "Grad-CAM katmanı: model.layer4[-1]"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = load_model(
        device
    )

    # --------------------------------------------------------
    # Layer4
    # --------------------------------------------------------

    target_layer = (
        model.layer4[-1]
    )

    gradcam = GradCAM(
        model,
        target_layer
    )

    transform = get_transform()

    # --------------------------------------------------------
    # 20 + 20 seç
    # --------------------------------------------------------

    selected_paths = (
        select_mura_samples()
    )

    print(
        "\nToplam seçilen görüntü:",
        len(selected_paths)
    )

    print(
        "20 Normal + 20 Abnormal"
    )

    print("\nİşlem başlıyor...\n")

    results = []

    correct_count = 0

    # --------------------------------------------------------
    # Görüntüler
    # --------------------------------------------------------

    for index, path_text in enumerate(
        selected_paths,
        start=1
    ):

        image_path = get_full_path(
            path_text
        )

        true_class = get_label(
            path_text
        )

        print(
            f"[{index}/40] "
            f"{image_path.name}"
        )

        try:

            (
                predicted_class,
                confidence,
                normal_prob,
                abnormal_prob,
                overlay

            ) = process_image(

                model=model,

                gradcam=gradcam,

                image_path=image_path,

                device=device,

                transform=transform
            )

            correct = (
                true_class
                == predicted_class
            )

            if correct:

                correct_count += 1

            # Aynı image1.png adı MURA'da birçok kez bulunabildiği için
            # çıktı dosyasını sıra numarasıyla benzersiz yapıyoruz.
            output_path = save_result(

                image_path=image_path,

                true_class=true_class,

                predicted_class=predicted_class,

                confidence=confidence,

                normal_prob=normal_prob,

                abnormal_prob=abnormal_prob,

                overlay=overlay
            )

            # Aynı isimli görüntüler birbirinin üzerine yazılmasın.
            unique_output_path = (
                OUTPUT_DIR
                / f"{index:02d}_{image_path.stem}_gradcam.png"
            )

            if output_path.exists():
                output_path.rename(unique_output_path)
                output_path = unique_output_path

            results.append({

                # Sadece image1.png gibi tekrar eden dosya adını değil,
                # MURA'daki gerçek ve benzersiz yolu da saklıyoruz.
                "image": image_path.name,

                "image_path": str(image_path),

                "mura_path": path_text,

                "true":
                    CLASS_NAMES[
                        true_class
                    ],

                "predicted":
                    CLASS_NAMES[
                        predicted_class
                    ],

                "confidence":
                    confidence * 100,

                "normal_prob":
                    normal_prob * 100,

                "abnormal_prob":
                    abnormal_prob * 100,

                "correct":
                    correct,

                "overlay":
                    overlay

            })

            print(
                f"    Gerçek: "
                f"{CLASS_NAMES[true_class]}"
            )

            print(
                f"    Tahmin: "
                f"{CLASS_NAMES[predicted_class]}"
            )

            print(
                f"    Normal: "
                f"%{normal_prob * 100:.2f}"
            )

            print(
                f"    Abnormal: "
                f"%{abnormal_prob * 100:.2f}"
            )

            print(
                f"    Sonuç: "
                f"{'CORRECT' if correct else 'WRONG'}"
            )

            print(
                f"    Kaydedildi: "
                f"{output_path.name}"
            )

            if not correct:
                print(
                    "    YANLIŞ TAHMİN - Gerçek MURA yolu:"
                )
                print(
                    f"    {path_text}"
                )

        except Exception as e:

            print(
                f"    HATA: {e}"
            )

        print("-" * 60)

    # --------------------------------------------------------
    # Grad-CAM kapat
    # --------------------------------------------------------

    gradcam.close()

    # --------------------------------------------------------
    # Sonuç
    # --------------------------------------------------------

    total = len(results)

    if total > 0:

        accuracy = (
            correct_count
            / total
        )

        print("\n")
        print("=" * 70)

        print(
            "MURA 40 SONUÇ"
        )

        print("=" * 70)

        print(
            f"İşlenen görüntü: "
            f"{total}"
        )

        print(
            f"Doğru tahmin: "
            f"{correct_count}/{total}"
        )

        print(
            f"Doğruluk: "
            f"%{accuracy * 100:.2f}"
        )

        print("=" * 70)

    # --------------------------------------------------------
    # DETAYLI CSV RAPORU + METRİKLER
    # --------------------------------------------------------

    if total > 0:

        # 1) Her görüntünün sonucunu CSV'ye kaydet
        report_rows = []

        for result in results:
            report_rows.append({
                "image": result["image"],
                "image_path": result["image_path"],
                "mura_path": result["mura_path"],
                "true_label": result["true"],
                "predicted_label": result["predicted"],
                "confidence_percent": round(result["confidence"], 2),
                "normal_probability_percent": round(result["normal_prob"], 2),
                "abnormal_probability_percent": round(result["abnormal_prob"], 2),
                "threshold": ABNORMAL_THRESHOLD,
                "status": "Correct" if result["correct"] else "Wrong"
            })

        report_df = pd.DataFrame(report_rows)

        csv_path = REPORT_DIR / "mura_40_results.csv"
        report_df.to_csv(
            csv_path,
            index=False,
            encoding="utf-8-sig"
        )

        # Sadece yanlış tahminleri ayrıca kaydet.
        wrong_df = report_df[
            report_df["status"] == "Wrong"
        ].copy()

        wrong_csv_path = (
            REPORT_DIR
            / "mura_40_wrong_predictions.csv"
        )

        wrong_df.to_csv(
            wrong_csv_path,
            index=False,
            encoding="utf-8-sig"
        )

        print("Yanlış tahminlerin ayrıntılı raporu:")
        print(wrong_csv_path)

        # 2) Confusion matrix değerleri
        tp = sum(
            1 for r in results
            if r["true"] == "Abnormal"
            and r["predicted"] == "Abnormal"
        )

        fn = sum(
            1 for r in results
            if r["true"] == "Abnormal"
            and r["predicted"] == "Normal"
        )

        tn = sum(
            1 for r in results
            if r["true"] == "Normal"
            and r["predicted"] == "Normal"
        )

        fp = sum(
            1 for r in results
            if r["true"] == "Normal"
            and r["predicted"] == "Abnormal"
        )

        # 3) Recall / Precision / F1
        abnormal_recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        normal_recall = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else 0.0
        )

        abnormal_precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        abnormal_f1 = (
            2 * abnormal_precision * abnormal_recall
            / (abnormal_precision + abnormal_recall)
            if (abnormal_precision + abnormal_recall) > 0
            else 0.0
        )

        # Normal sınıf için precision / F1
        normal_precision = (
            tn / (tn + fn)
            if (tn + fn) > 0
            else 0.0
        )

        normal_f1 = (
            2 * normal_precision * normal_recall
            / (normal_precision + normal_recall)
            if (normal_precision + normal_recall) > 0
            else 0.0
        )

        balanced_accuracy = (
            (normal_recall + abnormal_recall) / 2
        )

        macro_f1 = (
            (normal_f1 + abnormal_f1) / 2
        )

        # 4) Metrikleri terminale yazdır
        print("\n")
        print("=" * 70)
        print("MURA 40 DETAYLI PERFORMANS")
        print("=" * 70)
        print(f"Toplam görüntü        : {total}")
        print(f"Doğru tahmin          : {correct_count}/{total}")
        print(f"Accuracy              : %{accuracy * 100:.2f}")
        print(f"Normal Recall         : %{normal_recall * 100:.2f}")
        print(f"Abnormal Recall       : %{abnormal_recall * 100:.2f}")
        print(f"Abnormal Precision    : %{abnormal_precision * 100:.2f}")
        print(f"Abnormal F1           : %{abnormal_f1 * 100:.2f}")
        print(f"Normal Precision      : %{normal_precision * 100:.2f}")
        print(f"Normal F1             : %{normal_f1 * 100:.2f}")
        print(f"Balanced Accuracy     : %{balanced_accuracy * 100:.2f}")
        print(f"Macro F1              : %{macro_f1 * 100:.2f}")
        print("-" * 70)
        print("Confusion Matrix")
        print(f"True Normal / Pred Normal       (TN): {tn}")
        print(f"True Normal / Pred Abnormal     (FP): {fp}")
        print(f"True Abnormal / Pred Normal     (FN): {fn}")
        print(f"True Abnormal / Pred Abnormal   (TP): {tp}")
        print("-" * 70)
        print("CSV raporu:")
        print(csv_path)
        print("=" * 70)

        # 5) Confusion matrix görseli
        cm = np.array([
            [tn, fp],
            [fn, tp]
        ])

        fig_cm, ax_cm = plt.subplots(
            figsize=(7, 6)
        )

        ax_cm.imshow(cm, cmap="Blues")

        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])

        ax_cm.set_xticklabels(
            ["Normal", "Abnormal"]
        )

        ax_cm.set_yticklabels(
            ["Normal", "Abnormal"]
        )

        ax_cm.set_xlabel("Model Tahmini")
        ax_cm.set_ylabel("Gerçek Sınıf")
        ax_cm.set_title(
            f"MURA 40 Confusion Matrix\n"
            f"Accuracy: %{accuracy * 100:.2f}"
        )

        for i in range(2):
            for j in range(2):
                ax_cm.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=16
                )

        plt.tight_layout()

        cm_path = (
            OUTPUT_DIR
            / "mura_40_confusion_matrix.png"
        )

        plt.savefig(
            cm_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()

        # 6) Metrikleri ayrıca TXT dosyasına kaydet
        metrics_path = (
            REPORT_DIR
            / "mura_40_metrics.txt"
        )

        with open(
            metrics_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write("MURA 40 PERFORMANS RAPORU\n")
            f.write("=" * 60 + "\n")
            f.write(f"Toplam görüntü: {total}\n")
            f.write(f"Doğru tahmin: {correct_count}/{total}\n")
            f.write(f"Threshold: {ABNORMAL_THRESHOLD}\n")
            f.write(f"Accuracy: %{accuracy * 100:.2f}\n")
            f.write(f"Normal Recall: %{normal_recall * 100:.2f}\n")
            f.write(f"Abnormal Recall: %{abnormal_recall * 100:.2f}\n")
            f.write(f"Abnormal Precision: %{abnormal_precision * 100:.2f}\n")
            f.write(f"Abnormal F1: %{abnormal_f1 * 100:.2f}\n")
            f.write(f"Normal Precision: %{normal_precision * 100:.2f}\n")
            f.write(f"Normal F1: %{normal_f1 * 100:.2f}\n")
            f.write(f"Balanced Accuracy: %{balanced_accuracy * 100:.2f}\n")
            f.write(f"Macro F1: %{macro_f1 * 100:.2f}\n")
            f.write("\nConfusion Matrix\n")
            f.write(f"TN: {tn}\n")
            f.write(f"FP: {fp}\n")
            f.write(f"FN: {fn}\n")
            f.write(f"TP: {tp}\n")

        print("Confusion matrix:")
        print(cm_path)
        print("Metrik raporu:")
        print(metrics_path)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    save_summary(
        results
    )

    print(
        "\nTüm işlemler tamamlandı."
    )

    print(
        "\nÇıktı klasörü:"
    )

    print(
        OUTPUT_DIR
    )


# ============================================================
# ÇALIŞTIR
# ============================================================

if __name__ == "__main__":

    main()