from pathlib import Path
import random

import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    recall_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score
)


# ============================================================
# 1. GENEL AYARLAR
# ============================================================

ImageFile.LOAD_TRUNCATED_IMAGES = True

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 2. KLASÖRLER
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "MURA-v1.1"

MODEL_DIR = BASE_DIR / "outputs" / "models"
FIGURE_DIR = BASE_DIR / "outputs" / "figures"
REPORT_DIR = BASE_DIR / "outputs" / "reports"


MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


TRAIN_CSV = DATA_DIR / "train_image_paths.csv"
VALID_CSV = DATA_DIR / "valid_image_paths.csv"


# ============================================================
# 3. EĞİTİM AYARLARI
# ============================================================

# Önceki modelimiz 3000 görüntü ile eğitiliyordu.
# Şimdi daha fazla görüntü kullanıyoruz.

MAX_TRAIN_SAMPLES = 12000

# Validation'ın tamamını kullanacağız.
MAX_VALID_SAMPLES = None

BATCH_SIZE = 16

EPOCHS = 10

LEARNING_RATE = 0.00005

WEIGHT_DECAY = 0.0001

# Validation gelişmezse bu kadar epoch bekleyip duracağız.
EARLY_STOPPING_PATIENCE = 3


# Yeni modelin adı.
# Eski final_balanced_resnet18.pth dosyasına dokunmuyoruz.
MODEL_PATH = MODEL_DIR / "final_balanced_resnet18_v2.pth"


CLASS_NAMES = [
    "Normal",
    "Abnormal"
]


# ============================================================
# 4. BALANCED DATA SEÇİMİ
# ============================================================

def select_balanced_random_paths(csv_path, max_samples=None):

    paths = pd.read_csv(csv_path, header=None)[0].tolist()

    negative_paths = [
        p for p in paths
        if "negative" in p.lower()
    ]

    positive_paths = [
        p for p in paths
        if "positive" in p.lower()
    ]

    random.shuffle(negative_paths)
    random.shuffle(positive_paths)

    # Validation'ın tamamını kullan
    if max_samples is None:

        selected_paths = negative_paths + positive_paths

        random.shuffle(selected_paths)

        return selected_paths

    # Dengeli seçim
    half = max_samples // 2

    selected_negative = negative_paths[:half]
    selected_positive = positive_paths[:half]

    selected_paths = (
        selected_negative +
        selected_positive
    )

    random.shuffle(selected_paths)

    return selected_paths


# ============================================================
# 5. DATASET
# ============================================================

class MURADataset(Dataset):

    def __init__(self, image_paths, transform=None):

        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):

        return len(self.image_paths)

    def get_full_path(self, path):

        if path.startswith("MURA-v1.1"):

            return BASE_DIR / "data" / path

        return DATA_DIR / path

    def __getitem__(self, idx):

        image_path_text = self.image_paths[idx]

        image_path = self.get_full_path(
            image_path_text
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        # negative = Normal = 0
        # positive = Abnormal = 1

        label = (
            1
            if "positive" in image_path_text.lower()
            else 0
        )

        if self.transform:

            image = self.transform(image)

        return (
            image,
            torch.tensor(
                label,
                dtype=torch.long
            )
        )


# ============================================================
# 6. IMAGE TRANSFORMS
# ============================================================

# Medical görüntülerde aşırı augmentation kullanmıyoruz.
# Amacımız görüntünün anatomik yapısını bozmamak.

train_transform = transforms.Compose([

    transforms.Resize(
        (256, 256)
    ),

    transforms.RandomRotation(
        degrees=5
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.03, 0.03),
        scale=(0.97, 1.03)
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


valid_transform = transforms.Compose([

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
# 7. TRAIN
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0

    for images, labels in tqdm(
        dataloader,
        desc="Training"
    ):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        _, predicted = torch.max(
            outputs,
            1
        )

        correct += (
            predicted == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = (
        running_loss / total
    )

    epoch_acc = (
        correct / total
    )

    return epoch_loss, epoch_acc


# ============================================================
# 8. VALIDATION
# ============================================================

def validate(
    model,
    dataloader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in tqdm(
            dataloader,
            desc="Validation"
        ):

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item()
                * images.size(0)
            )

            total += labels.size(0)

            _, predicted = torch.max(
                outputs,
                1
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predicted.cpu().numpy()
            )

    epoch_loss = (
        running_loss / total
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            all_labels,
            all_predictions
        )
    )

    normal_recall = recall_score(
        all_labels,
        all_predictions,
        pos_label=0,
        zero_division=0
    )

    abnormal_recall = recall_score(
        all_labels,
        all_predictions,
        pos_label=1,
        zero_division=0
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return (
        epoch_loss,
        accuracy,
        balanced_accuracy,
        normal_recall,
        abnormal_recall,
        macro_f1,
        all_labels,
        all_predictions
    )


# ============================================================
# 9. TRAINING GRAPH
# ============================================================

def save_training_curves(history):

    epochs = range(
        1,
        len(history["train_loss"]) + 1
    )

    plt.figure(
        figsize=(12, 7)
    )

    plt.plot(
        epochs,
        history["train_loss"],
        marker="o",
        label="Train Loss"
    )

    plt.plot(
        epochs,
        history["valid_loss"],
        marker="o",
        label="Valid Loss"
    )

    plt.plot(
        epochs,
        history["train_acc"],
        marker="o",
        label="Train Accuracy"
    )

    plt.plot(
        epochs,
        history["valid_acc"],
        marker="o",
        label="Valid Accuracy"
    )

    plt.plot(
        epochs,
        history["normal_recall"],
        marker="o",
        label="Normal Recall"
    )

    plt.plot(
        epochs,
        history["abnormal_recall"],
        marker="o",
        label="Abnormal Recall"
    )

    plt.plot(
        epochs,
        history["macro_f1"],
        marker="o",
        label="Macro F1"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Score / Loss"
    )

    plt.title(
        "Final Balanced ResNet18 v2 Training"
    )

    plt.legend()

    plt.grid(True)

    save_path = (
        FIGURE_DIR /
        "training_curves_final_balanced_v2.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Eğitim grafiği kaydedildi:"
    )

    print(
        save_path
    )


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    labels,
    predictions
):

    cm = confusion_matrix(
        labels,
        predictions
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Normal",
            "Abnormal"
        ]
    )

    display.plot(
        values_format="d"
    )

    plt.title(
        "Final Balanced ResNet18 v2"
    )

    save_path = (
        FIGURE_DIR /
        "confusion_matrix_final_balanced_v2.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Confusion matrix kaydedildi:"
    )

    print(
        save_path
    )


# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

def save_classification_report(
    labels,
    predictions
):

    report = classification_report(
        labels,
        predictions,
        target_names=[
            "Normal",
            "Abnormal"
        ],
        zero_division=0
    )

    save_path = (
        REPORT_DIR /
        "classification_report_final_balanced_v2.txt"
    )

    with open(
        save_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)

    print(
        "Classification report kaydedildi:"
    )

    print(
        save_path
    )


# ============================================================
# 12. MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "FINAL BALANCED RESNET18 V2 EĞİTİMİ"
    )
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Kullanılan cihaz:",
        device
    )

    print()

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    train_paths = select_balanced_random_paths(
        TRAIN_CSV,
        MAX_TRAIN_SAMPLES
    )

    valid_paths = select_balanced_random_paths(
        VALID_CSV,
        MAX_VALID_SAMPLES
    )

    train_dataset = MURADataset(
        train_paths,
        transform=train_transform
    )

    valid_dataset = MURADataset(
        valid_paths,
        transform=valid_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print(
        "Train görüntü sayısı:",
        len(train_dataset)
    )

    print(
        "Validation görüntü sayısı:",
        len(valid_dataset)
    )

    print()

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print(
        "ResNet18 hazırlanıyor..."
    )

    try:

        model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        print(
            "ImageNet pretrained ağırlıkları yüklendi."
        )

    except Exception:

        model = models.resnet18(
            weights=None
        )

        print(
            "Pretrained ağırlıklar yüklenemedi."
        )

        print(
            "Model sıfırdan başlatıldı."
        )

    # --------------------------------------------------------
    # LAYER FREEZE
    # --------------------------------------------------------

    # Önce tüm katmanları dondur.

    for param in model.parameters():

        param.requires_grad = False

    # layer3 ve layer4'ü eğitime açıyoruz.
    # Önceki modelde yalnızca layer4 açıktı.

    for param in model.layer3.parameters():

        param.requires_grad = True

    for param in model.layer4.parameters():

        param.requires_grad = True

    # Son sınıflandırma katmanı.

    model.fc = nn.Linear(
        model.fc.in_features,
        2
    )

    model = model.to(device)

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    # Eğitim datası zaten dengeli:
    # %50 Normal
    # %50 Abnormal
    #
    # Bu nedenle aşırı class weight kullanmıyoruz.

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(

        [
            {
                "params": model.layer3.parameters(),
                "lr": LEARNING_RATE
            },

            {
                "params": model.layer4.parameters(),
                "lr": LEARNING_RATE
            },

            {
                "params": model.fc.parameters(),
                "lr": LEARNING_RATE * 2
            }
        ],

        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # LR SCHEDULER
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="max",

        factor=0.5,

        patience=1
    )

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    history = {

        "train_loss": [],
        "valid_loss": [],

        "train_acc": [],
        "valid_acc": [],

        "balanced_acc": [],

        "normal_recall": [],
        "abnormal_recall": [],

        "macro_f1": []
    }

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    best_macro_f1 = -1.0

    best_labels = None
    best_predictions = None
    best_metrics = None

    epochs_without_improvement = 0

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(EPOCHS):

        print()
        print("=" * 70)

        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        print("=" * 70)

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_loss, train_acc = train_one_epoch(

            model,

            train_loader,

            criterion,

            optimizer,

            device
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        (
            valid_loss,
            valid_acc,
            balanced_acc,
            normal_recall,
            abnormal_recall,
            macro_f1,
            labels,
            predictions

        ) = validate(

            model,

            valid_loader,

            criterion,

            device
        )

        # ----------------------------------------------------
        # SAVE HISTORY
        # ----------------------------------------------------

        history["train_loss"].append(
            train_loss
        )

        history["valid_loss"].append(
            valid_loss
        )

        history["train_acc"].append(
            train_acc
        )

        history["valid_acc"].append(
            valid_acc
        )

        history["balanced_acc"].append(
            balanced_acc
        )

        history["normal_recall"].append(
            normal_recall
        )

        history["abnormal_recall"].append(
            abnormal_recall
        )

        history["macro_f1"].append(
            macro_f1
        )

        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

        print()

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Train Accuracy: {train_acc:.4f}"
        )

        print(
            f"Validation Loss: {valid_loss:.4f}"
        )

        print(
            f"Validation Accuracy: {valid_acc:.4f}"
        )

        print(
            f"Balanced Accuracy: {balanced_acc:.4f}"
        )

        print(
            f"Normal Recall: {normal_recall:.4f}"
        )

        print(
            f"Abnormal Recall: {abnormal_recall:.4f}"
        )

        print(
            f"Macro F1: {macro_f1:.4f}"
        )

        # ----------------------------------------------------
        # SCHEDULER
        # ----------------------------------------------------

        scheduler.step(
            macro_f1
        )

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        if macro_f1 > best_macro_f1:

            best_macro_f1 = macro_f1

            best_labels = labels
            best_predictions = predictions

            best_metrics = {

                "valid_acc": valid_acc,

                "balanced_acc":
                    balanced_acc,

                "normal_recall":
                    normal_recall,

                "abnormal_recall":
                    abnormal_recall,

                "macro_f1":
                    macro_f1
            }

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print()

            print(
                ">>> YENİ EN İYİ MODEL KAYDEDİLDİ"
            )

            print(
                MODEL_PATH
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

            print()

            print(
                "Validation Macro F1 gelişmedi."
            )

            print(
                "Gelişmeyen epoch sayısı:",
                epochs_without_improvement
            )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print()

            print(
                "Early stopping çalıştı."
            )

            print(
                "Eğitim erken durduruldu."
            )

            break

    # ========================================================
    # TRAINING FINISHED
    # ========================================================

    print()
    print("=" * 70)
    print(
        "EĞİTİM TAMAMLANDI"
    )
    print("=" * 70)

    print()

    print(
        "Yeni model:"
    )

    print(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # SAVE GRAPHS
    # --------------------------------------------------------

    save_training_curves(
        history
    )

    # --------------------------------------------------------
    # SAVE REPORTS
    # --------------------------------------------------------

    if (
        best_labels is not None
        and best_predictions is not None
    ):

        save_confusion_matrix(
            best_labels,
            best_predictions
        )

        save_classification_report(
            best_labels,
            best_predictions
        )

    # --------------------------------------------------------
    # BEST RESULTS
    # --------------------------------------------------------

    if best_metrics is not None:

        print()

        print(
            "EN İYİ MODEL SONUÇLARI"
        )

        print("-" * 50)

        print(
            "Validation Accuracy:",
            round(
                best_metrics["valid_acc"],
                4
            )
        )

        print(
            "Balanced Accuracy:",
            round(
                best_metrics["balanced_acc"],
                4
            )
        )

        print(
            "Normal Recall:",
            round(
                best_metrics["normal_recall"],
                4
            )
        )

        print(
            "Abnormal Recall:",
            round(
                best_metrics["abnormal_recall"],
                4
            )
        )

        print(
            "Macro F1:",
            round(
                best_metrics["macro_f1"],
                4
            )
        )

    print()

    print("=" * 70)

    print(
        "ÖNEMLİ:"
    )

    print(
        "Bu model yanlış tahminleri azaltmak için"
    )

    print(
        "daha fazla veri ve daha kontrollü eğitim"
    )

    print(
        "kullanılarak yeniden eğitildi."
    )

    print()

    print(
        "Modelin %100 hatasız olması garanti edilemez."
    )

    print(
        "Grad-CAM kutuları ayrıca lokalizasyon"
    )

    print(
        "performansı açısından değerlendirilmelidir."
    )

    print("=" * 70)


# ============================================================
# PROGRAMI ÇALIŞTIR
# ============================================================

if __name__ == "__main__":

    main()