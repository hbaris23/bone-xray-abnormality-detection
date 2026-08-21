from pathlib import Path
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "MURA-v1.1"
OUTPUT_DIR = BASE_DIR / "outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_csv = DATA_DIR / "train_image_paths.csv"

paths = pd.read_csv(train_csv, header=None)[0].tolist()

negative_paths = [p for p in paths if "negative" in p.lower()]
positive_paths = [p for p in paths if "positive" in p.lower()]

sample_paths = negative_paths[:3] + positive_paths[:3]

def get_full_path(p):
    if p.startswith("MURA-v1.1"):
        return BASE_DIR / "data" / p
    return DATA_DIR / p

plt.figure(figsize=(12, 8))

for i, path in enumerate(sample_paths):
    image_path = get_full_path(path)
    img = Image.open(image_path).convert("L")

    label = "Normal" if "negative" in path.lower() else "Abnormal"

    plt.subplot(2, 3, i + 1)
    plt.imshow(img, cmap="gray")
    plt.title(label)
    plt.axis("off")

plt.tight_layout()

save_path = OUTPUT_DIR / "sample_images.png"
plt.savefig(save_path, dpi=300)
plt.show()

print("Örnek görüntüler kaydedildi:")
print(save_path)