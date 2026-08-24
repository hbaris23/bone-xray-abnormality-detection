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
    recall_score
)


ImageFile.LOAD_TRUNCATED_IMAGES = True

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

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

# Abnormal yakalamayı artırmak için daha fazla veri kullanıyoruz.
# Bilgisayar çok yavaşlarsa bu sayıları azaltırız.
MAX_TRAIN_SAMPLES = 4000
MAX_VALID_SAMPLES = 1000

BATCH_SIZE = 16
EPOCHS = 7
LEARNING_RATE = 0.0001


def select_balanced_random_paths(csv_path, max_samples):
    paths = pd.read_csv(csv_path, header=None)[0].tolist()

    negative_paths = [p for p in paths if "negative" in p.lower()]
    positive_paths = [p for p in paths if "positive" in p.lower()]

    random.shuffle(negative_paths)
    random.shuffle(positive_paths)

    half = max_samples // 2

    selected_paths = negative_paths[:half] + positive_paths[:half]
    random.shuffle(selected_paths)

    return selected_paths


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
        image_path = self.get_full_path(image_path_text)

        image = Image.open(image_path).convert("RGB")

        # negative = normal = 0
        # positive = abnormal = 1
        label = 1 if "positive" in image_path_text.lower() else 0

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomRotation(8),
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

valid_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    abnormal_recall = recall_score(
        all_labels,
        all_predictions,
        pos_label=1,
        zero_division=0
    )

    return epoch_loss, epoch_acc, abnormal_recall, all_labels, all_predictions


def save_training_curves(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 5))

    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["valid_loss"], label="Valid Loss")
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["valid_acc"], label="Valid Accuracy")
    plt.plot(epochs, history["abnormal_recall"], label="Abnormal Recall")

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Recall-Focused Training Results")
    plt.legend()
    plt.grid(True)

    save_path = FIGURE_DIR / "training_curves_recall_focus.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Eğitim grafiği kaydedildi:", save_path)


def save_confusion_matrix(labels, predictions):
    cm = confusion_matrix(labels, predictions)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Abnormal"]
    )

    display.plot(values_format="d")
    plt.title("Recall-Focused Confusion Matrix")

    save_path = FIGURE_DIR / "confusion_matrix_recall_focus.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Confusion matrix kaydedildi:", save_path)


def save_classification_report(labels, predictions):
    report = classification_report(
        labels,
        predictions,
        target_names=["Normal", "Abnormal"],
        zero_division=0
    )

    save_path = REPORT_DIR / "classification_report_recall_focus.txt"

    with open(save_path, "w", encoding="utf-8") as file:
        file.write(report)

    print("Classification report kaydedildi:", save_path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Kullanılan cihaz:", device)

    train_paths = select_balanced_random_paths(TRAIN_CSV, MAX_TRAIN_SAMPLES)
    valid_paths = select_balanced_random_paths(VALID_CSV, MAX_VALID_SAMPLES)

    train_dataset = MURADataset(train_paths, transform=train_transform)
    valid_dataset = MURADataset(valid_paths, transform=valid_transform)

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

    print("Train görüntü sayısı:", len(train_dataset))
    print("Valid görüntü sayısı:", len(valid_dataset))

    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        print("Hazır ResNet18 ağırlıkları yüklendi.")
    except Exception:
        model = models.resnet18(weights=None)
        print("Hazır ağırlıklar yüklenemedi. Model sıfırdan başlatıldı.")

    # Önce tüm katmanları donduruyoruz.
    for param in model.parameters():
        param.requires_grad = False

    # Son bloğu eğitime açıyoruz.
    for param in model.layer4.parameters():
        param.requires_grad = True

    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    # Burada abnormal sınıfına daha fazla ağırlık veriyoruz.
    # Amaç: Abnormal görüntüleri normal sanma hatasını azaltmak.
    class_weights = torch.tensor([1.0, 2.0], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        list(model.layer4.parameters()) + list(model.fc.parameters()),
        lr=LEARNING_RATE,
        weight_decay=0.0001
    )

    history = {
        "train_loss": [],
        "valid_loss": [],
        "train_acc": [],
        "valid_acc": [],
        "abnormal_recall": []
    }

    best_abnormal_recall = -1.0
    best_valid_acc = -1.0
    best_labels = None
    best_predictions = None

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        valid_loss, valid_acc, abnormal_recall, labels, predictions = validate(
            model,
            valid_loader,
            criterion,
            device
        )

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        history["train_acc"].append(train_acc)
        history["valid_acc"].append(valid_acc)
        history["abnormal_recall"].append(abnormal_recall)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.4f}")
        print(f"Abnormal Recall: {abnormal_recall:.4f}")

        # Bu sefer en iyi modeli accuracy'ye göre değil,
        # abnormal recall değerine göre kaydediyoruz.
        if abnormal_recall > best_abnormal_recall:
            best_abnormal_recall = abnormal_recall
            best_valid_acc = valid_acc
            best_labels = labels
            best_predictions = predictions

            save_path = MODEL_DIR / "recall_focus_resnet18.pth"
            torch.save(model.state_dict(), save_path)
            print("Abnormal recall odaklı en iyi model kaydedildi:", save_path)

    save_training_curves(history)

    if best_labels is not None and best_predictions is not None:
        save_confusion_matrix(best_labels, best_predictions)
        save_classification_report(best_labels, best_predictions)

    print("\nRecall-focused eğitim tamamlandı.")
    print("En iyi abnormal recall:", round(best_abnormal_recall, 4))
    print("Bu modeldeki validation accuracy:", round(best_valid_acc, 4))


if __name__ == "__main__":
    main()