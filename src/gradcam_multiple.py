from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

import matplotlib.pyplot as plt


# ============================================================
# PROJE YOLLARI
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

# FINAL BALANCED MODEL
MODEL_PATH = (
    BASE_DIR
    / "outputs"
    / "models"
    / "final_balanced_resnet18.pth"
)

# Kendi test görüntülerini buraya koy
TEST_IMAGE_DIR = BASE_DIR / "test_images"

# Grad-CAM çıktıları
FIGURE_DIR = (
    BASE_DIR
    / "outputs"
    / "figures"
    / "gradcam_multiple"
)

# CSV raporları
REPORT_DIR = (
    BASE_DIR
    / "outputs"
    / "reports"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FINAL THRESHOLD
# ============================================================

# 3197 validation görüntüsü üzerindeki değerlendirmeye göre
# final balanced sistem için threshold = 0.35

ABNORMAL_THRESHOLD = 0.35


# ============================================================
# SINIFLAR
# ============================================================

CLASS_NAMES = {
    0: "Normal",
    1: "Abnormal"
}


# ============================================================
# GRAD-CAM AYARI
# ============================================================

# None bırakırsan modelin tahmin ettiği sınıf açıklanır.
#
# Örneğin:
# Model Abnormal → Abnormal Grad-CAM
# Model Normal   → Normal Grad-CAM

GRADCAM_TARGET_CLASS = None


# ============================================================
# SADECE BELİRLİ DOSYALARI ÇALIŞTIRMA
# ============================================================

# Boş bırakırsan test_images içindeki bütün görüntüler işlenir.

SELECTED_FILES = [
    "mura_1_NORMAL.png",
    "mura_2_NORMAL.png",
    "mura_3_ABNORMAL.png",
    "mura_4_ABNORMAL.png"
]


# Örnek kullanım:
#
# SELECTED_FILES = [
#     "img1.jpg",
#     "img3.jpg"
# ]


# ============================================================
# MODELİ YÜKLE
# ============================================================

def load_model(device):

    if not MODEL_PATH.exists():

        print(
            "HATA: Model dosyası bulunamadı."
        )

        print(
            "Beklenen model:"
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
# TEST GÖRSELLERİNİ BUL
# ============================================================

def get_image_files():

    if not TEST_IMAGE_DIR.exists():

        raise FileNotFoundError(
            f"test_images klasörü bulunamadı: "
            f"{TEST_IMAGE_DIR}"
        )


    allowed_extensions = [
        ".jpg",
        ".jpeg",
        ".png"
    ]


    image_files = [

        file

        for file in TEST_IMAGE_DIR.iterdir()

        if file.is_file()
        and file.suffix.lower()
        in allowed_extensions
    ]


    image_files = sorted(
        image_files,
        key=lambda x: x.name.lower()
    )


    # Eğer özel dosya listesi verilmişse
    # sadece onları kullan.

    if len(SELECTED_FILES) > 0:

        selected_set = set(
            SELECTED_FILES
        )

        image_files = [

            file

            for file in image_files

            if file.name in selected_set
        ]


    if len(image_files) == 0:

        raise ValueError(
            "İşlenecek görüntü bulunamadı."
        )


    return image_files


# ============================================================
# GRAD-CAM SINIFI
# ============================================================

class GradCAM:

    def __init__(
        self,
        model,
        target_layer
    ):

        self.model = model

        self.target_layer = target_layer

        self.gradients = None

        self.activations = None


        # Forward hook
        self.forward_hook = (
            self.target_layer.register_forward_hook(
                self.save_activations
            )
        )


        # Backward hook
        self.backward_hook = (
            self.target_layer.register_full_backward_hook(
                self.save_gradients
            )
        )


    def save_activations(
        self,
        module,
        input,
        output
    ):

        self.activations = output


    def save_gradients(
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

        self.model.zero_grad()


        output = self.model(
            input_tensor
        )


        class_score = (
            output[0, target_class]
        )


        class_score.backward()


        gradients = (
            self.gradients.detach()
        )


        activations = (
            self.activations.detach()
        )


        # Global Average Pooling
        weights = torch.mean(
            gradients,
            dim=(2, 3),
            keepdim=True
        )


        # Ağırlıklı aktivasyonlar
        cam = torch.sum(
            weights * activations,
            dim=1
        )


        # Negatif değerleri kaldır
        cam = torch.relu(
            cam
        )


        cam = (
            cam
            .squeeze()
            .cpu()
            .numpy()
        )


        # Normalize et
        cam = (
            cam - np.min(cam)
        )


        cam = (
            cam
            / (np.max(cam) + 1e-8)
        )


        return cam


    def remove_hooks(self):

        self.forward_hook.remove()

        self.backward_hook.remove()


# ============================================================
# MODEL TAHMİNİ
# ============================================================

def predict_image(
    model,
    image_tensor,
    device
):

    input_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(device)
    )


    with torch.no_grad():

        outputs = model(
            input_tensor
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


    # --------------------------------------------------------
    # FINAL THRESHOLD
    # --------------------------------------------------------

    if (
        abnormal_probability
        >= ABNORMAL_THRESHOLD
    ):

        predicted_label = 1

        confidence = abnormal_probability

    else:

        predicted_label = 0

        confidence = normal_probability


    return (
        predicted_label,
        confidence,
        normal_probability,
        abnormal_probability
    )


# ============================================================
# HEATMAP BOYUTLANDIR
# ============================================================

def resize_heatmap(
    heatmap,
    size=(224, 224)
):

    heatmap_uint8 = np.uint8(
        255 * heatmap
    )


    heatmap_image = Image.fromarray(
        heatmap_uint8
    )


    heatmap_image = (
        heatmap_image.resize(
            size,
            resample=Image.Resampling.BILINEAR
        )
    )


    heatmap_resized = (
        np.array(
            heatmap_image
        ).astype(
            np.float32
        )
        / 255.0
    )


    return heatmap_resized


# ============================================================
# HEATMAP + ORİJİNAL GÖRÜNTÜ
# ============================================================

def create_overlay(
    original_image,
    heatmap
):

    original_resized = (
        original_image
        .resize((224, 224))
        .convert("RGB")
    )


    original_np = (
        np.array(
            original_resized
        ).astype(
            np.float32
        )
        / 255.0
    )


    heatmap_resized = resize_heatmap(
        heatmap
    )


    cmap = plt.get_cmap(
        "jet"
    )


    colored_heatmap = cmap(
        heatmap_resized
    )[:, :, :3]


    # Orijinal görüntü + heatmap
    overlay = (
        0.6 * original_np
        + 0.4 * colored_heatmap
    )


    overlay = np.clip(
        overlay,
        0,
        1
    )


    return (
        original_np,
        heatmap_resized,
        overlay
    )


# ============================================================
# GRAD-CAM GÖRSELİNİ KAYDET
# ============================================================

def save_single_gradcam_result(
    image_path,
    original_np,
    heatmap_resized,
    overlay,
    prediction_name,
    confidence,
    normal_probability,
    abnormal_probability,
    gradcam_target_name
):

    save_path = (
        FIGURE_DIR
        / f"{image_path.stem}_gradcam.png"
    )


    plt.figure(
        figsize=(15, 5)
    )


    # --------------------------------------------------------
    # 1 - ORIGINAL
    # --------------------------------------------------------

    plt.subplot(
        1,
        3,
        1
    )


    plt.imshow(
        original_np
    )


    plt.title(
        "Original X-ray"
    )


    plt.axis(
        "off"
    )


    # --------------------------------------------------------
    # 2 - HEATMAP
    # --------------------------------------------------------

    plt.subplot(
        1,
        3,
        2
    )


    plt.imshow(
        heatmap_resized,
        cmap="jet"
    )


    plt.title(
        f"Grad-CAM\nTarget: {gradcam_target_name}"
    )


    plt.axis(
        "off"
    )


    # --------------------------------------------------------
    # 3 - OVERLAY
    # --------------------------------------------------------

    plt.subplot(
        1,
        3,
        3
    )


    plt.imshow(
        overlay
    )


    plt.title(

        f"Prediction: {prediction_name}\n"

        f"Confidence: "
        f"{confidence * 100:.2f}%\n"

        f"Normal: "
        f"{normal_probability * 100:.2f}% | "

        f"Abnormal: "
        f"{abnormal_probability * 100:.2f}%\n"

        f"Grad-CAM Target: "
        f"{gradcam_target_name}"
    )


    plt.axis(
        "off"
    )


    plt.tight_layout()


    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    return save_path


# ============================================================
# ÖZET GRİD
# ============================================================

def save_summary_grid(
    results
):

    total = len(
        results
    )


    if total == 0:
        return


    cols = 2


    rows = (
        total + cols - 1
    ) // cols


    plt.figure(
        figsize=(
            12,
            rows * 5
        )
    )


    for i, result in enumerate(
        results
    ):

        image = Image.open(
            result["gradcam_path"]
        ).convert("RGB")


        plt.subplot(
            rows,
            cols,
            i + 1
        )


        plt.imshow(
            image
        )


        plt.axis(
            "off"
        )


        plt.title(
            result["filename"],
            fontsize=10
        )


    plt.tight_layout()


    save_path = (
        FIGURE_DIR
        / "gradcam_summary_grid.png"
    )


    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        "\nToplu Grad-CAM özet görseli kaydedildi:"
    )


    print(
        save_path
    )


# ============================================================
# TEK GÖRÜNTİYİ İŞLE
# ============================================================

def process_single_image(
    model,
    image_path,
    device,
    transform
):

    # --------------------------------------------------------
    # ORİJİNAL GÖRÜNTÜ
    # --------------------------------------------------------

    original_image = (
        Image.open(
            image_path
        ).convert("RGB")
    )


    # --------------------------------------------------------
    # MODEL GİRDİSİ
    # --------------------------------------------------------

    image_tensor = transform(
        original_image
    )


    # --------------------------------------------------------
    # TAHMİN
    # --------------------------------------------------------

    (
        predicted_label,
        confidence,
        normal_probability,
        abnormal_probability
    ) = predict_image(

        model=model,

        image_tensor=image_tensor,

        device=device
    )


    prediction_name = (
        CLASS_NAMES[
            predicted_label
        ]
    )


    # --------------------------------------------------------
    # GRAD-CAM HEDEF SINIFI
    # --------------------------------------------------------

    if GRADCAM_TARGET_CLASS is None:

        # Model hangi sınıfı tahmin ettiyse
        # Grad-CAM onu açıklayacak.

        target_class = (
            predicted_label
        )

    else:

        target_class = (
            GRADCAM_TARGET_CLASS
        )


    gradcam_target_name = (
        CLASS_NAMES[
            target_class
        ]
    )


    # --------------------------------------------------------
    # INPUT TENSOR
    # --------------------------------------------------------

    input_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(device)
    )


    # --------------------------------------------------------
    # GRAD-CAM LAYER
    # --------------------------------------------------------

    # ResNet18'in son convolutional bölümlerinden
    # detaylı Grad-CAM almak için layer3[-1] kullanıyoruz.

    target_layer = (
        model.layer4[-1]
    )


    gradcam = GradCAM(
        model,
        target_layer
    )


    # --------------------------------------------------------
    # HEATMAP
    # --------------------------------------------------------

    heatmap = gradcam.generate(

        input_tensor=input_tensor,

        target_class=target_class
    )


    gradcam.remove_hooks()


    # --------------------------------------------------------
    # OVERLAY
    # --------------------------------------------------------

    (
        original_np,
        heatmap_resized,
        overlay
    ) = create_overlay(

        original_image=original_image,

        heatmap=heatmap
    )


    # --------------------------------------------------------
    # GÖRSELİ KAYDET
    # --------------------------------------------------------

    gradcam_path = (
        save_single_gradcam_result(

            image_path=image_path,

            original_np=original_np,

            heatmap_resized=heatmap_resized,

            overlay=overlay,

            prediction_name=prediction_name,

            confidence=confidence,

            normal_probability=normal_probability,

            abnormal_probability=abnormal_probability,

            gradcam_target_name=gradcam_target_name
        )
    )


    # --------------------------------------------------------
    # SONUÇ
    # --------------------------------------------------------

    result = {

        "filename":
            image_path.name,

        "image_path":
            str(image_path),

        "prediction":
            prediction_name,

        "confidence_percent":
            confidence * 100,

        "normal_probability_percent":
            normal_probability * 100,

        "abnormal_probability_percent":
            abnormal_probability * 100,

        "threshold":
            ABNORMAL_THRESHOLD,

        "gradcam_target_class":
            gradcam_target_name,

        "gradcam_layer":
            "model.layer3[-1]",

        "gradcam_path":
            str(gradcam_path)
    }


    return result


# ============================================================
# CSV RAPORU
# ============================================================

def save_csv_report(
    results
):

    df = pd.DataFrame(
        results
    )


    save_path = (
        REPORT_DIR
        / "gradcam_multiple_results.csv"
    )


    df.to_csv(
        save_path,
        index=False,
        encoding="utf-8-sig"
    )


    print(
        "\nGrad-CAM CSV raporu kaydedildi:"
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
    # TEST GÖRSELLERİ
    # --------------------------------------------------------

    image_files = (
        get_image_files()
    )


    print(
        "Toplam görüntü sayısı:",
        len(image_files)
    )


    print(
        "Kullanılan abnormal threshold:",
        ABNORMAL_THRESHOLD
    )


    if GRADCAM_TARGET_CLASS is None:

        print(
            "Grad-CAM hedefi: "
            "Modelin tahmin ettiği sınıf"
        )

    else:

        print(
            "Grad-CAM hedefi:",
            CLASS_NAMES[
                GRADCAM_TARGET_CLASS
            ]
        )


    print(
        "Grad-CAM katmanı:",
        "model.layer3[-1]"
    )


    print()


    # --------------------------------------------------------
    # TÜM GÖRSELLERİ İŞLE
    # --------------------------------------------------------

    results = []


    for image_path in image_files:

        print(
            "İşleniyor:",
            image_path.name
        )


        try:

            result = (
                process_single_image(

                    model=model,

                    image_path=image_path,

                    device=device,

                    transform=transform
                )
            )


            results.append(
                result
            )


            print(
                "Tahmin:",
                result["prediction"]
            )


            print(
                f"Güven: "
                f"%{result['confidence_percent']:.2f}"
            )


            print(
                f"Normal olasılığı: "
                f"%{result['normal_probability_percent']:.2f}"
            )


            print(
                f"Abnormal olasılığı: "
                f"%{result['abnormal_probability_percent']:.2f}"
            )


            print(
                "Grad-CAM hedefi:",
                result[
                    "gradcam_target_class"
                ]
            )


            print(
                "Grad-CAM kaydedildi:",
                result[
                    "gradcam_path"
                ]
            )


        except Exception as error:

            print(
                "HATA:",
                error
            )


        print(
            "-" * 60
        )


    # --------------------------------------------------------
    # RAPORLAR
    # --------------------------------------------------------

    save_csv_report(
        results
    )


    save_summary_grid(
        results
    )


    # --------------------------------------------------------
    # BİTİŞ
    # --------------------------------------------------------

    print()
    print(
        "Tüm Grad-CAM işlemleri tamamlandı."
    )

    print(
        "Grad-CAM klasörü:",
        FIGURE_DIR
    )

    print(
        "\nUYARI:"
    )

    print(
        "Grad-CAM modeli tanı koydurmaz."
    )

    print(
        "Sadece modelin karar verirken"
    )

    print(
        "hangi görüntü bölgelerine"
    )

    print(
        "odaklandığını görselleştirir."
    )


# ============================================================
# PROGRAMI BAŞLAT
# ============================================================

if __name__ == "__main__":
    main()