import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def simple_perceptual_hash(path: Path, size: Tuple[int, int] = (8, 8)) -> str:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return ""
    img = cv2.resize(img, size)
    mean = float(img.mean())
    bits = "".join("1" if pixel > mean else "0" for pixel in img.flatten())
    return bits


def normalize_source_name(name: str) -> str:
    n = name.lower()
    n = re.sub(r"\.(jpg|jpeg|png|bmp|webp)$", "", n)
    n = re.sub(r"(?:_frame|_frame_\d+|_f\d+|_img\d+|_\d{2,})$", "", n)
    n = re.sub(r"[_-]+", "_", n)
    return n.strip("_")


def collect_dataset_summary(dataset_dir: str) -> Dict[str, object]:
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    class_names = [p.name for p in sorted(root.iterdir()) if p.is_dir() and p.name in {"train", "validation", "test"}]
    allowed_splits = ["train", "validation", "test"]
    split_dirs = {split: root / split for split in allowed_splits if (root / split).exists()}

    summary = {
        "dataset_dir": str(root),
        "splits": {split: {} for split in allowed_splits},
        "duplicates": [],
        "near_duplicates": [],
        "corrupted_images": [],
        "resolution_summary": {},
        "source_leakage": [],
    }

    all_files: List[Tuple[str, Path, str]] = []

    for split, split_dir in split_dirs.items():
        split_summary = {}
        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            files = sorted(p for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
            split_summary[cls_dir.name] = len(files)
            for file_path in files:
                all_files.append((split, file_path, cls_dir.name))
        summary["splits"][split] = split_summary

    size_counter = Counter()
    resolutions = []
    seen_hashes = {}
    hashes_by_file = {}

    for split, file_path, cls_name in all_files:
        if not file_path.exists():
            summary["corrupted_images"].append({"split": split, "class": cls_name, "file": str(file_path), "reason": "missing"})
            continue

        img = cv2.imread(str(file_path))
        if img is None:
            summary["corrupted_images"].append({"split": split, "class": cls_name, "file": str(file_path), "reason": "cv2_decode_failed"})
            continue

        h, w = img.shape[:2]
        resolutions.append((w, h))
        size_counter[(w, h)] += 1
        hashes_by_file[str(file_path)] = sha256_file(file_path)

        existing = seen_hashes.get(hashes_by_file[str(file_path)])
        if existing is not None:
            summary["duplicates"].append({"sha256": hashes_by_file[str(file_path)], "files": [existing, str(file_path)]})
        else:
            seen_hashes[hashes_by_file[str(file_path)]] = str(file_path)

    if resolutions:
        widths = [w for w, _ in resolutions]
        heights = [h for _, h in resolutions]
        summary["resolution_summary"] = {
            "min_width": min(widths),
            "max_width": max(widths),
            "mean_width": round(sum(widths) / len(widths), 2),
            "min_height": min(heights),
            "max_height": max(heights),
            "mean_height": round(sum(heights) / len(heights), 2),
            "unique_resolutions": dict(size_counter),
        }

    perceptual_by_file: Dict[str, str] = {}
    for split, file_path, cls_name in all_files:
        phash = simple_perceptual_hash(file_path)
        if not phash:
            continue
        perceptual_by_file[str(file_path)] = phash

    by_hash: Dict[str, List[str]] = defaultdict(list)
    for file_name, phash in perceptual_by_file.items():
        by_hash[phash].append(file_name)

    for phash, paths in by_hash.items():
        if len(paths) > 1:
            summary["near_duplicates"].append({"perceptual_hash": phash, "files": sorted(paths)})

    source_map: Dict[str, List[str]] = defaultdict(list)
    for split, file_path, cls_name in all_files:
        source_name = normalize_source_name(file_path.stem)
        source_map[source_name].append(f"{split}:{cls_name}:{file_path.name}")

    for source_name, occurrences in source_map.items():
        split_names = {entry.split(":", 3)[0] for entry in occurrences}
        if len(split_names) > 1:
            summary["source_leakage"].append({"source": source_name, "splits": sorted(split_names), "files": occurrences})

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Audit dataset structure and quality checks.")
    parser.add_argument("--dataset_dir", default="dataset_large", help="Dataset root directory (contains train/validation/test).")
    args = parser.parse_args()

    report = collect_dataset_summary(args.dataset_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
