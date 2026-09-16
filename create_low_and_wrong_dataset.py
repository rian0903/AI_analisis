"""
create_low_and_wrong_dataset.py
================================
Script untuk mengumpulkan semua gambar dari folder 'gambar_model_bingung' 
(Low Confidence & Misclassified) dan menyusunnya ke dalam folder dataset retraining
'dataset_low_and_wrong' lengkap dengan struktur split (train, validation, test) 
sehingga kompatibel 100% dengan src/train.py dan src/dataset.py.
"""

import json
import random
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CONFUSED_DIR = BASE_DIR / "models" / "gambar_model_bingung"

# Target dataset root
DATASET_TARGET = BASE_DIR / "dataset_low_and_wrong"
DATASET_TARGET_TRAIN = BASE_DIR / "dataset_low_and_wrong_train"

def main():
    if not CONFUSED_DIR.exists():
        print(f"Error: Folder '{CONFUSED_DIR}' tidak ditemukan. Jalankan 'extract_confused_images.py' terlebih dahulu.")
        return

    # Prepare dataset_low_and_wrong/{train, validation, test}/{Safe, Violence, NSFW}
    splits = ["train", "validation", "test"]
    categories = ["Safe", "Violence", "NSFW"]

    for target_dir in [DATASET_TARGET, DATASET_TARGET_TRAIN]:
        if target_dir.exists():
            shutil.rmtree(target_dir)

    for split in splits:
        for cat in categories:
            (DATASET_TARGET / split / cat).mkdir(parents=True, exist_ok=True)
            
    for cat in categories:
        (DATASET_TARGET_TRAIN / cat).mkdir(parents=True, exist_ok=True)

    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    # Collect unique images per class: true_label -> list of (filename, src_file_path)
    class_images = {"Safe": [], "Violence": [], "NSFW": []}
    unique_seen = set()

    json_files = list(CONFUSED_DIR.rglob("*.json"))
    print(f"Ditemukan {len(json_files)} metadata JSON file.")

    for json_p in json_files:
        if json_p.name == "ringkasan_gambar_bingung.json":
            continue

        try:
            with open(json_p, "r", encoding="utf-8") as f:
                meta = json.load(f)

            filename = meta.get("filename")
            true_label = meta.get("true_label")

            if filename in unique_seen or true_label not in class_images:
                continue

            img_file = json_p.parent / filename
            if not img_file.exists():
                possible_imgs = [p for p in json_p.parent.iterdir() if p.stem == json_p.stem and p.suffix.lower() in supported_exts]
                if possible_imgs:
                    img_file = possible_imgs[0]
                else:
                    continue

            unique_seen.add(filename)
            class_images[true_label].append((filename, img_file))

        except Exception as e:
            print(f"Error reading {json_p}: {e}")

    # Split ratios: Train 80%, Validation 10%, Test 10%
    random.seed(42)
    
    counts_split = {split: {"Safe": 0, "Violence": 0, "NSFW": 0} for split in splits}

    for cat, items in class_images.items():
        random.shuffle(items)
        n_total = len(items)
        n_val = max(1, int(n_total * 0.10)) if n_total >= 10 else (1 if n_total >= 3 else 0)
        n_test = max(1, int(n_total * 0.10)) if n_total >= 10 else (1 if n_total >= 3 else 0)
        n_train = n_total - n_val - n_test

        train_items = items[:n_train]
        val_items = items[n_train:n_train + n_val]
        test_items = items[n_train + n_val:]

        for filename, src_path in train_items:
            shutil.copy2(src_path, DATASET_TARGET / "train" / cat / filename)
            shutil.copy2(src_path, DATASET_TARGET_TRAIN / cat / filename)
            counts_split["train"][cat] += 1

        for filename, src_path in val_items:
            shutil.copy2(src_path, DATASET_TARGET / "validation" / cat / filename)
            counts_split["validation"][cat] += 1

        for filename, src_path in test_items:
            shutil.copy2(src_path, DATASET_TARGET / "test" / cat / filename)
            counts_split["test"][cat] += 1

    total_images = len(unique_seen)
    print("\n=======================================================")
    print(" Pembuatan Dataset Low & Wrong Complete!")
    print(f" Total Gambar Unik Terkumpul : {total_images}")
    print("\n Rincian Pembagian Dataset (Train 80% / Val 10% / Test 10%):")
    for split in splits:
        t_cat = counts_split[split]
        s_total = sum(t_cat.values())
        print(f"   * Split '{split:10s}': Total {s_total:3d} gambar | Safe: {t_cat['Safe']:3d}, Violence: {t_cat['Violence']:3d}, NSFW: {t_cat['NSFW']:3d}")

    print(f"    - {DATASET_TARGET.resolve()}")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
