from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, models


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "MURA-v1.1"
OUTPUT_DIR = BASE_DIR / "outputs" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = DATA_DIR / "train_image_paths.csv"
VALID_CSV = DATA_DIR / "valid_image_paths.csv"

MAX_TRAIN_SAMPLES = 300
MAX_VALID_SAMPLES = 100

BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 0.001


def select_balanced_paths(csv_path, max_samples):
    paths = pd.read_csv(csv_path, header=None)[0].tolist()

    negative_paths = [p for p in paths if "negative" in p.lower()]
    positive_paths = [p for p in paths if "positive" in p.lower()]

    half = max_samples // 2

    selected_paths = negative_paths[:half] + positive_paths[:half]

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

        # MURA'da:
        # negative = normal = 0
        # positive = abnormal = 1
        if "positive" in image_path_text.lower():
            label = 1
        else:
            label = 0

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
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

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Kullanılan cihaz:", device)

    train_paths = select_balanced_paths(TRAIN_CSV, MAX_TRAIN_SAMPLES)
    valid_paths = select_balanced_paths(VALID_CSV, MAX_VALID_SAMPLES)

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

    # İlk denemede modeli hızlı çalıştırmak için ana katmanları donduruyoruz.
    for param in model.parameters():
        param.requires_grad = False

    # Son sınıflandırma katmanını 2 sınıfa göre değiştiriyoruz.
    model.fc = nn.Linear(model.fc.in_features, 2)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    best_valid_acc = 0.0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        valid_loss, valid_acc = validate(
            model,
            valid_loader,
            criterion,
            device
        )

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.4f}")

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            save_path = OUTPUT_DIR / "baseline_resnet18.pth"
            torch.save(model.state_dict(), save_path)
            print("Model kaydedildi:", save_path)

    print("\nİlk eğitim denemesi tamamlandı.")


if __name__ == "__main__":
    main()