from pathlib import Path
import pandas as pd
from PIL import Image

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "MURA-v1.1"

train_csv = DATA_DIR / "train_image_paths.csv"
valid_csv = DATA_DIR / "valid_image_paths.csv"

print("Proje klasörü:", BASE_DIR)
print("Veri klasörü:", DATA_DIR)

if not DATA_DIR.exists():
    print("HATA: MURA-v1.1 klasörü bulunamadı.")
    exit()

if not train_csv.exists():
    print("HATA: train_image_paths.csv bulunamadı.")
    exit()

if not valid_csv.exists():
    print("HATA: valid_image_paths.csv bulunamadı.")
    exit()

train_paths = pd.read_csv(train_csv, header=None)[0].tolist()
valid_paths = pd.read_csv(valid_csv, header=None)[0].tolist()

def summarize(paths, name):
    positive = sum("positive" in p.lower() for p in paths)
    negative = sum("negative" in p.lower() for p in paths)

    print(f"\n{name} görüntü sayısı: {len(paths)}")
    print(f"{name} positive/anormal: {positive}")
    print(f"{name} negative/normal: {negative}")

summarize(train_paths, "Train")
summarize(valid_paths, "Valid")

first_path = train_paths[0]

if first_path.startswith("MURA-v1.1"):
    first_image_path = BASE_DIR / "data" / first_path
else:
    first_image_path = DATA_DIR / first_path

print("\nİlk görüntü yolu:")
print(first_image_path)

if first_image_path.exists():
    img = Image.open(first_image_path)
    print("İlk görüntü başarıyla açıldı.")
    print("Görüntü boyutu:", img.size)
    print("\nVeri kontrolü başarılı.")
else:
    print("HATA: İlk görüntü bulunamadı.")