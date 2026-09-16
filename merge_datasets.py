"""
merge_datasets.py
=================
Script untuk menggabungkan kembali seluruh dataset yang sebelumnya dipisah 
(dataset_low_and_wrong & gambar_model_bingung) ke dalam folder dataset utama 'dataset_large'.
"""

import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_LARGE = BASE_DIR / "dataset_large"
DATASET_LOW_WRONG = BASE_DIR / "dataset_low_and_wrong"

def count_images_in_dataset(dataset_dir: Path):
    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    counts = {}
    splits = ["train", "validation", "test"]
    categories = ["Safe", "Violence", "NSFW"]

    total = 0
    for split in splits:
        counts[split] = {}
        for cat in categories:
            folder = dataset_dir / split / cat
            if folder.exists():
                c = len([p for p in folder.iterdir() if p.suffix.lower() in supported_exts])
            else:
                c = 0
            counts[split][cat] = c
            total += c
    counts["total"] = total
    return counts

def main():
    if not DATASET_LARGE.exists():
        print(f"Error: Folder '{DATASET_LARGE}' tidak ditemukan.")
        return

    print("Mengecek jumlah gambar awal pada 'dataset_large'...")
    before_counts = count_images_in_dataset(DATASET_LARGE)
    print(f"Total awal dataset_large: {before_counts['total']} gambar")

    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    splits = ["train", "validation", "test"]
    categories = ["Safe", "Violence", "NSFW"]

    added_count = 0
    skipped_count = 0

    if DATASET_LOW_WRONG.exists():
        print("\nMenggabungkan data dari 'dataset_low_and_wrong' ke 'dataset_large'...")
        for split in splits:
            for cat in categories:
                src_folder = DATASET_LOW_WRONG / split / cat
                dest_folder = DATASET_LARGE / split / cat
                dest_folder.mkdir(parents=True, exist_ok=True)

                if not src_folder.exists():
                    continue

                images = [p for p in src_folder.iterdir() if p.suffix.lower() in supported_exts]
                for img_p in images:
                    target_p = dest_folder / img_p.name
                    if not target_p.exists():
                        shutil.copy2(img_p, target_p)
                        added_count += 1
                    else:
                        skipped_count += 1

    print("\nMengecek jumlah gambar akhir pada 'dataset_large'...")
    after_counts = count_images_in_dataset(DATASET_LARGE)

    print("\n=======================================================")
    print(" Penggabungan Dataset Selesai!")
    print(f" Total Gambar Baru Ditambahkan : {added_count}")
    print(f" Total Gambar Sudah Ada/Skip   : {skipped_count}")
    print(f" Total Gambar Akhir dataset_large : {after_counts['total']}")
    print("\n Rincian per Split pada dataset_large:")
    for split in splits:
        s_counts = after_counts[split]
        s_total = sum(s_counts.values())
        print(f"   * {split:10s}: Total {s_total:5d} gambar | Safe: {s_counts['Safe']:4d}, Violence: {s_counts['Violence']:4d}, NSFW: {s_counts['NSFW']:4d}")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
