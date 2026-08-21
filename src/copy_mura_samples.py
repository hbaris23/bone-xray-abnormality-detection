from pathlib import Path
import random
import shutil
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "MURA-v1.1"
VALID_CSV = DATA_DIR / "valid_image_paths.csv"

TEST_DIR = BASE_DIR / "test_images"
TEST_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)

# Kaç tane Normal ve Abnormal alınacak?
NORMAL_COUNT = 2
ABNORMAL_COUNT = 2


def get_full_path(path_text):
    if path_text.startswith("MURA-v1.1"):
        return BASE_DIR / "data" / path_text

    return DATA_DIR / path_text


def main():

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

    selected_normal = random.sample(
        normal_paths,
        NORMAL_COUNT
    )

    selected_abnormal = random.sample(
        abnormal_paths,
        ABNORMAL_COUNT
    )

    selected = (
        selected_normal +
        selected_abnormal
    )

    print("MURA'dan seçilen görüntüler:\n")

    # Önce eski test görüntülerini temizle
    for file in TEST_DIR.iterdir():

        if file.is_file():
            file.unlink()

    # Görüntüleri kopyala
    for index, path_text in enumerate(selected, start=1):

        source = get_full_path(path_text)

        if "positive" in path_text.lower():
            label = "ABNORMAL"
        else:
            label = "NORMAL"

        extension = source.suffix

        destination = (
            TEST_DIR /
            f"mura_{index}_{label}{extension}"
        )

        shutil.copy2(
            source,
            destination
        )

        print(f"{destination.name}")
        print(f"Gerçek etiket: {label}")
        print(f"Kaynak: {source}")
        print("-" * 60)

    print("\nToplam kopyalanan görüntü:", len(selected))
    print("Normal:", NORMAL_COUNT)
    print("Abnormal:", ABNORMAL_COUNT)
    print("\nGörüntüler test_images klasörüne kopyalandı.")


if __name__ == "__main__":
    main()