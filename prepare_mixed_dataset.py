"""
prepare_mixed_dataset.py
========================
Prepares a balanced, high-quality mixed-domain dataset (50% Real-World Photos + 50% 2D Anime/Illustration)
for Safe, Violence, and NSFW classes into dataset_large/{train, validation, test}/.
"""

import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw_data"
TARGET_DIR = BASE_DIR / "dataset_large"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

TARGET_PER_CLASS = {
    "train": 1200,      # 600 real + 600 anime per class
    "validation": 250,  # 125 real + 125 anime per class
    "test": 250,        # 125 real + 125 anime per class
}

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def collect_images_from_dir(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    valid = []
    for p in directory.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            valid.append(p)
    return valid


def build_mixed_dataset():
    print("=" * 60)
    print(" BUILDING BALANCED MIXED DATASET (REAL PHOTOS + 2D ANIME)")
    print(" Target: dataset_large/")
    print("=" * 60)

    # Clean target directory first
    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)

    # 1. Source Dictionaries
    sources = {
        "Safe": {
            "real": [
                RAW_DIR / "safe_raw",
                RAW_DIR / "kaggle_downloads" / "safe",
                RAW_DIR / "hf_downloads" / "safe",
            ],
            "anime": [
                RAW_DIR / "anime_safe_raw",
                RAW_DIR / "hf_downloads" / "anime_safe",
            ],
        },
        "Violence": {
            "real": [
                RAW_DIR / "violence_raw",
                RAW_DIR / "kaggle_downloads" / "violence",
                RAW_DIR / "hf_downloads" / "violence",
            ],
            "anime": [
                RAW_DIR / "anime_violence_raw",
                RAW_DIR / "hf_downloads" / "anime_violence",
            ],
        },
        "NSFW": {
            "real": [
                RAW_DIR / "nsfw_raw",
                RAW_DIR / "kaggle_downloads" / "nsfw",
                RAW_DIR / "hf_downloads" / "nsfw",
            ],
            "anime": [
                RAW_DIR / "anime_nsfw_raw",
                RAW_DIR / "hf_downloads" / "civitai_nsfw",
                RAW_DIR / "hf_downloads" / "anime_nsfw",
            ],
        },
    }

    # Clean & recreate target structure
    for split in ["train", "validation", "test"]:
        for cat in ["Safe", "Violence", "NSFW"]:
            (TARGET_DIR / split / cat).mkdir(parents=True, exist_ok=True)

    for cat, domains in sources.items():
        print(f"\nProcessing category: [{cat}]")

        # Collect real
        real_files: List[Path] = []
        for d in domains["real"]:
            real_files.extend(collect_images_from_dir(d))
        real_files = list(set(real_files))
        random.shuffle(real_files)

        # Collect anime
        anime_files: List[Path] = []
        for d in domains["anime"]:
            anime_files.extend(collect_images_from_dir(d))
        anime_files = list(set(anime_files))
        random.shuffle(anime_files)

        print(f"  Available Pool -> Real: {len(real_files)} | Anime: {len(anime_files)}")

        # Split Real
        n_train_real = TARGET_PER_CLASS["train"] // 2
        n_val_real = TARGET_PER_CLASS["validation"] // 2
        n_test_real = TARGET_PER_CLASS["test"] // 2

        real_train = real_files[:n_train_real]
        real_val = real_files[n_train_real : n_train_real + n_val_real]
        real_test = real_files[n_train_real + n_val_real : n_train_real + n_val_real + n_test_real]

        # Split Anime
        n_train_anime = TARGET_PER_CLASS["train"] // 2
        n_val_anime = TARGET_PER_CLASS["validation"] // 2
        n_test_anime = TARGET_PER_CLASS["test"] // 2

        anime_train = anime_files[:n_train_anime]
        anime_val = anime_files[n_train_anime : n_train_anime + n_val_anime]
        anime_test = anime_files[n_train_anime + n_val_anime : n_train_anime + n_val_anime + n_test_anime]

        splits_data = {
            "train": [("real", p) for p in real_train] + [("anime", p) for p in anime_train],
            "validation": [("real", p) for p in real_val] + [("anime", p) for p in anime_val],
            "test": [("real", p) for p in real_test] + [("anime", p) for p in anime_test],
        }

        for split, items in splits_data.items():
            dest_dir = TARGET_DIR / split / cat
            count = 0
            for domain_tag, src_path in items:
                dest_name = f"{domain_tag}_{src_path.name}"
                dest_path = dest_dir / dest_name
                try:
                    shutil.copy2(src_path, dest_path)
                    count += 1
                except Exception:
                    pass
            print(f"  -> {split:10s} / {cat:8s}: {count} files copied.")

    print("\n" + "=" * 60)
    print(" DATASET PREPARATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    build_mixed_dataset()
