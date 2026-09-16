"""
setup_dataset.py
================
Script otomatis untuk mengunduh, mengorganisir, dan membagi dataset
Image-Based Inappropriate Content Detection.

Sumber Dataset
--------------
├── Safe     : Intel Image Classification (Kaggle)
│               kaggle.com/datasets/puneet6060/intel-image-classification
├── Violence : Violence vs Non-Violence 11K Images (Kaggle)
│               kaggle.com/datasets/ahmedmoorsy/violence-vs-non-violence-11k-images
└── NSFW     : deepghs/nsfw_detect (HuggingFace)  [fallback: manual folder]

Target
------
Total ~6,000 gambar (~2,000 per kelas)
Split  : 70% train  | 15% validation | 15% test
Hasil  : dataset/train|validation|test/{Safe,Violence,NSFW}/

Prasyarat
---------
1. Akun Kaggle + kaggle.json  (lihat STEP 1 di bawah)
2. pip install kaggle scikit-learn Pillow tqdm datasets huggingface_hub

Cara Jalankan
-------------
    python setup_dataset.py

Opsi lanjutan:
    python setup_dataset.py --safe_count 2000 --violence_count 2000 --nsfw_count 2000
    python setup_dataset.py --skip_download   (jika sudah punya raw images di raw_data/)
"""

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------
PROJECT_ROOT   = Path(__file__).resolve().parent
RAW_DATA_DIR   = PROJECT_ROOT / "raw_data"          # Tempat download sementara
DATASET_DIR    = PROJECT_ROOT / "dataset"           # Output akhir

# Jumlah gambar per kelas
DEFAULT_SAFE_COUNT     = 2000
DEFAULT_VIOLENCE_COUNT = 2000
DEFAULT_NSFW_COUNT     = 2000

# Rasio split
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Ekstensi gambar yang valid
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Seed untuk reproducibility
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Informasi dataset Kaggle
# ---------------------------------------------------------------------------
KAGGLE_DATASETS = {
    "safe": {
        "owner": "puneet6060",
        "dataset": "intel-image-classification",
        "description": "Intel Image Classification (landscape, buildings, etc.)",
        # Sub-folder di dalam zip yang berisi gambar training
        "source_dirs": ["seg_train/seg_train"],   # Coba path ini dulu
        "alt_source_dirs": ["seg_train"],          # Alternatif jika path beda
    },
    "violence": {
        "owner": "ahmedmoorsy",
        "dataset": "violence-vs-non-violence-11k-images",
        "description": "Violence vs Non-Violence 11K Images",
        "source_dirs": ["Violence"],               # Ambil hanya folder Violence
        "alt_source_dirs": ["violence"],
    },
}


# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def check_python_packages() -> bool:
    """Periksa apakah semua paket yang diperlukan sudah terinstal."""
    required = {
        "kaggle": "kaggle",
        "sklearn": "scikit-learn",
        "PIL": "Pillow",
        "tqdm": "tqdm",
    }
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        logger.error("Paket berikut belum terinstal: %s", ", ".join(missing))
        logger.info("Jalankan: pip install %s", " ".join(missing))
        return False
    return True


def check_kaggle_credentials() -> bool:
    """
    Periksa apakah kaggle.json tersedia dan valid.

    File kaggle.json harus ada di salah satu lokasi:
    - Windows : C:/Users/<Username>/.kaggle/kaggle.json
    - Linux   : ~/.kaggle/kaggle.json
    """
    kaggle_dir  = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    if not kaggle_json.exists():
        logger.error("=" * 60)
        logger.error("File kaggle.json TIDAK DITEMUKAN!")
        logger.error("=" * 60)
        logger.info("CARA MENDAPATKAN KAGGLE API KEY:")
        logger.info("  1. Buka https://www.kaggle.com")
        logger.info("  2. Login → klik foto profil → 'Settings'")
        logger.info("  3. Scroll ke bagian 'API' → klik 'Create New Token'")
        logger.info("  4. File kaggle.json otomatis terunduh")
        logger.info("  5. Pindahkan ke: %s", kaggle_json)
        logger.info("")
        logger.info("  Windows PowerShell:")
        logger.info("    mkdir $HOME\\.kaggle -Force")
        logger.info("    copy Downloads\\kaggle.json $HOME\\.kaggle\\kaggle.json")
        logger.info("=" * 60)
        return False

    # Validasi isi file
    try:
        with open(kaggle_json) as f:
            creds = json.load(f)
        if "username" not in creds or "key" not in creds:
            logger.error("kaggle.json tidak valid — harus berisi 'username' dan 'key'")
            return False
    except json.JSONDecodeError:
        logger.error("kaggle.json rusak / bukan format JSON yang valid")
        return False

    # Set permission (penting di Linux)
    try:
        os.chmod(kaggle_json, 0o600)
    except Exception:
        pass  # Windows tidak support chmod, abaikan

    logger.info("✅ Kaggle credentials valid (user: %s)", creds.get("username"))
    return True


def is_valid_image(path: Path) -> bool:
    """Cek apakah file adalah gambar valid yang dapat dibuka."""
    if path.suffix.lower() not in VALID_EXTENSIONS:
        return False
    try:
        from PIL import Image
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def collect_images(directory: Path, recursive: bool = True) -> List[Path]:
    """
    Kumpulkan semua path gambar valid dari sebuah direktori.

    Parameters
    ----------
    directory : Path
        Direktori yang akan di-scan.
    recursive : bool
        Jika True, scan sub-direktori juga.

    Returns
    -------
    list[Path]
        Daftar path gambar valid.
    """
    if not directory.exists():
        return []

    pattern = "**/*" if recursive else "*"
    all_files = [
        p for p in directory.glob(pattern)
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]
    return all_files


def sample_images(image_paths: List[Path], n: int, seed: int = RANDOM_SEED) -> List[Path]:
    """
    Ambil n sampel gambar secara acak.

    Parameters
    ----------
    image_paths : list[Path]
        List semua path gambar yang tersedia.
    n : int
        Jumlah gambar yang ingin diambil.
    seed : int
        Random seed untuk reproducibility.

    Returns
    -------
    list[Path]
        List path gambar yang sudah di-sample.
    """
    random.seed(seed)
    if len(image_paths) <= n:
        logger.warning(
            "Hanya tersedia %d gambar (diminta %d). Menggunakan semua.",
            len(image_paths), n
        )
        return image_paths.copy()
    return random.sample(image_paths, n)


def split_images(
    image_paths: List[Path],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float   = VAL_RATIO,
    seed: int = RANDOM_SEED,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Bagi list gambar menjadi train, validation, dan test set.

    Menggunakan train_test_split dari sklearn untuk pembagian yang benar
    secara statistik.

    Parameters
    ----------
    image_paths : list[Path]
    train_ratio : float
        Proporsi untuk training (0–1).
    val_ratio : float
        Proporsi untuk validation (0–1). Test = 1 - train - val.
    seed : int

    Returns
    -------
    tuple[list, list, list]
        (train_paths, val_paths, test_paths)
    """
    from sklearn.model_selection import train_test_split

    test_ratio = 1.0 - train_ratio - val_ratio

    # Split pertama: pisahkan train dari (val+test)
    train_paths, temp_paths = train_test_split(
        image_paths,
        test_size=(val_ratio + test_ratio),
        random_state=seed,
    )

    # Split kedua: pisahkan val dari test
    val_size_relative = val_ratio / (val_ratio + test_ratio)
    val_paths, test_paths = train_test_split(
        temp_paths,
        test_size=(1.0 - val_size_relative),
        random_state=seed,
    )

    return train_paths, val_paths, test_paths


def copy_images_to_split(
    image_paths: List[Path],
    destination_dir: Path,
    prefix: str = "",
) -> int:
    """
    Salin gambar ke direktori tujuan dengan rename yang aman.

    Parameters
    ----------
    image_paths : list[Path]
        Gambar yang akan disalin.
    destination_dir : Path
        Folder tujuan (sudah pasti ada).
    prefix : str
        Prefix untuk nama file (menghindari konflik).

    Returns
    -------
    int
        Jumlah gambar yang berhasil disalin.
    """
    from tqdm import tqdm

    destination_dir.mkdir(parents=True, exist_ok=True)
    copied = 0

    for idx, src in enumerate(tqdm(image_paths, desc=f"  Copying → {destination_dir.name}", leave=False)):
        ext = src.suffix.lower()
        dst = destination_dir / f"{prefix}_{idx:05d}{ext}"
        try:
            shutil.copy2(src, dst)
            copied += 1
        except Exception as exc:
            logger.warning("Gagal menyalin %s: %s", src.name, exc)

    return copied


# ===========================================================================
# DOWNLOAD FUNCTIONS
# ===========================================================================

def download_kaggle_dataset(owner: str, dataset_name: str, download_dir: Path) -> Path:
    """
    Download dataset dari Kaggle menggunakan kaggle CLI.

    Parameters
    ----------
    owner : str
        Username pemilik dataset di Kaggle.
    dataset_name : str
        Nama/slug dataset.
    download_dir : Path
        Direktori tempat menyimpan file yang diunduh.

    Returns
    -------
    Path
        Path ke direktori hasil ekstraksi.

    Raises
    ------
    RuntimeError
        Jika download gagal.
    """
    download_dir.mkdir(parents=True, exist_ok=True)
    full_name = f"{owner}/{dataset_name}"
    extract_dir = download_dir / dataset_name

    if extract_dir.exists() and any(extract_dir.iterdir()):
        logger.info("Dataset '%s' sudah ada, skip download.", full_name)
        return extract_dir

    logger.info("⬇️  Mengunduh dataset: %s ...", full_name)
    logger.info("    Ini mungkin membutuhkan beberapa menit...")

    zip_path = download_dir / f"{dataset_name}.zip"

    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            full_name,
            path=str(download_dir),
            unzip=False,
            quiet=False,
        )
    except Exception as exc:
        # Fallback: gunakan kaggle CLI langsung
        logger.warning("Kaggle Python API gagal (%s), mencoba kaggle CLI...", exc)
        cmd = [
            sys.executable, "-m", "kaggle",
            "datasets", "download",
            "-d", full_name,
            "-p", str(download_dir),
            "--quiet",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Download gagal untuk {full_name}.\n"
                f"Error: {result.stderr}\n\n"
                f"Pastikan:\n"
                f"  1. kaggle.json sudah dikonfigurasi\n"
                f"  2. Anda sudah accept terms dataset di:\n"
                f"     https://www.kaggle.com/datasets/{full_name}"
            )

    # Temukan file zip yang baru diunduh
    zip_files = list(download_dir.glob(f"{dataset_name}*.zip"))
    if not zip_files:
        zip_files = list(download_dir.glob("*.zip"))

    if not zip_files:
        raise RuntimeError(f"File zip tidak ditemukan setelah download {full_name}")

    zip_file = zip_files[0]
    logger.info("📦 Mengekstrak %s ...", zip_file.name)

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_file, "r") as zf:
        zf.extractall(extract_dir)

    # Hapus zip setelah ekstraksi untuk hemat ruang
    zip_file.unlink()
    logger.info("✅ Selesai mengekstrak ke %s", extract_dir)

    return extract_dir


def find_source_directory(base_dir: Path, candidates: List[str]) -> Optional[Path]:
    """
    Cari direktori sumber dari daftar kandidat path.

    Parameters
    ----------
    base_dir : Path
        Direktori root dari dataset yang sudah diekstrak.
    candidates : list[str]
        Daftar path relatif yang mungkin berisi gambar (dicoba berurutan).

    Returns
    -------
    Path | None
        Path yang ditemukan, atau None jika tidak ada yang cocok.
    """
    # Coba setiap kandidat
    for candidate in candidates:
        candidate_path = base_dir / candidate
        if candidate_path.exists():
            images = collect_images(candidate_path, recursive=True)
            if images:
                logger.info("  📁 Sumber ditemukan: %s (%d gambar)", candidate_path, len(images))
                return candidate_path

    # Fallback: cari direktori dengan gambar terbanyak
    logger.warning("Kandidat path tidak ditemukan, melakukan pencarian otomatis...")
    best_dir = None
    best_count = 0

    for subdir in base_dir.rglob("*"):
        if subdir.is_dir():
            count = len([
                f for f in subdir.iterdir()
                if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
            ])
            if count > best_count:
                best_count = count
                best_dir = subdir

    if best_dir:
        logger.info("  📁 Sumber otomatis: %s (%d gambar)", best_dir, best_count)
        return best_dir

    return None


def download_nsfw_huggingface(download_dir: Path, n_images: int = 2000) -> Path:
    """
    Download gambar NSFW dari HuggingFace datasets.

    Menggunakan dataset 'deepghs/nsfw_detect' yang berisi gambar
    berlabel 'explicit', 'questionable', dan 'safe'.
    Kita ambil gambar berlabel 'explicit' untuk kelas NSFW.

    Parameters
    ----------
    download_dir : Path
        Direktori tempat menyimpan gambar NSFW.
    n_images : int
        Jumlah gambar NSFW yang ingin diunduh.

    Returns
    -------
    Path
        Path ke direktori yang berisi gambar NSFW.
    """
    nsfw_dir = download_dir / "nsfw_images"
    nsfw_dir.mkdir(parents=True, exist_ok=True)

    existing = list(nsfw_dir.glob("*.jpg")) + list(nsfw_dir.glob("*.png"))
    if len(existing) >= n_images:
        logger.info("Dataset NSFW sudah ada (%d gambar), skip download.", len(existing))
        return nsfw_dir

    logger.info("⬇️  Mengunduh dataset NSFW dari HuggingFace...")
    logger.info("    Dataset: deepghs/nsfw_detect")

    try:
        from datasets import load_dataset
        from tqdm import tqdm
        from PIL import Image as PILImage

        logger.info("    Memuat metadata dataset...")
        # Dataset ini cukup besar, streaming untuk efisiensi
        ds = load_dataset(
            "deepghs/nsfw_detect",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )

        saved = 0
        checked = 0
        max_check = n_images * 5  # Cek maksimal 5x target untuk dapat yang NSFW

        logger.info("    Menyaring gambar berlabel 'explicit'/'nsfw'...")
        for item in tqdm(ds, total=max_check, desc="  Scanning NSFW dataset"):
            if saved >= n_images:
                break
            if checked >= max_check:
                break
            checked += 1

            # Label yang termasuk NSFW
            label = str(item.get("label", item.get("labels", ""))).lower()
            if any(nsfw_kw in label for nsfw_kw in ["explicit", "nsfw", "pornographic", "adult"]):
                try:
                    img = item.get("image") or item.get("img")
                    if img is None:
                        continue

                    # Pastikan format PIL
                    if not isinstance(img, PILImage.Image):
                        continue

                    # Simpan ke disk
                    out_path = nsfw_dir / f"nsfw_{saved:05d}.jpg"
                    img = img.convert("RGB")
                    img.save(out_path, "JPEG", quality=85)
                    saved += 1
                except Exception:
                    continue

        logger.info("✅ NSFW: %d gambar disimpan ke %s", saved, nsfw_dir)

        if saved < n_images // 2:
            logger.warning(
                "Hanya mendapat %d gambar NSFW dari deepghs/nsfw_detect. "
                "Mencoba dataset alternatif...", saved
            )
            return _download_nsfw_alternative(nsfw_dir, n_images, saved)

    except ImportError:
        logger.warning("Library 'datasets' belum terinstal. Mencoba instalasi otomatis...")
        subprocess.run([sys.executable, "-m", "pip", "install", "datasets", "-q"], check=False)
        return download_nsfw_huggingface(download_dir, n_images)
    except Exception as exc:
        logger.warning("Download NSFW dari HuggingFace gagal: %s", exc)
        logger.info("Mencoba dataset NSFW alternatif...")
        return _download_nsfw_alternative(nsfw_dir, n_images, 0)

    return nsfw_dir


def _download_nsfw_alternative(nsfw_dir: Path, n_images: int, already_have: int) -> Path:
    """
    Dataset NSFW alternatif: AnimeFace + explicit content dataset.
    Menggunakan 'notai/nsfw-images' dari HuggingFace jika tersedia.
    """
    remaining = n_images - already_have
    if remaining <= 0:
        return nsfw_dir

    try:
        from datasets import load_dataset
        from tqdm import tqdm
        from PIL import Image as PILImage

        logger.info("  Mencoba dataset alternatif: 'laiyer-ai/nsfw-text-image-dataset'...")
        ds = load_dataset(
            "laiyer-ai/nsfw-text-image-dataset",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )

        existing_count = len(list(nsfw_dir.glob("*.jpg")))
        saved = 0

        for item in tqdm(ds, total=remaining * 3, desc="  Scanning alt NSFW"):
            if saved >= remaining:
                break

            try:
                img = item.get("image") or item.get("img")
                if img is None or not isinstance(img, PILImage.Image):
                    continue
                out_path = nsfw_dir / f"nsfw_{existing_count + saved:05d}.jpg"
                img.convert("RGB").save(out_path, "JPEG", quality=85)
                saved += 1
            except Exception:
                continue

        logger.info("  NSFW alternatif: %d gambar tambahan", saved)

    except Exception as exc:
        logger.error("Semua sumber NSFW gagal: %s", exc)
        logger.info("=" * 60)
        logger.info("INSTRUKSI MANUAL untuk kelas NSFW:")
        logger.info("  1. Tempatkan gambar NSFW Anda di folder:")
        logger.info("     %s", nsfw_dir)
        logger.info("  2. Format yang didukung: .jpg .jpeg .png .webp")
        logger.info("  3. Minimal %d gambar diperlukan", n_images)
        logger.info("  4. Setelah menambahkan gambar, jalankan ulang script")
        logger.info("     dengan flag --skip_download")
        logger.info("=" * 60)
        logger.info("Sumber NSFW yang bisa digunakan secara manual:")
        logger.info("  - github.com/alex000kim/nsfw_data_scraper")
        logger.info("  - huggingface.co/datasets (cari 'nsfw images')")

    return nsfw_dir


# ===========================================================================
# DATASET ORGANIZER
# ===========================================================================

class DatasetOrganizer:
    """
    Mengorganisir gambar dari sumber yang berbeda ke dalam struktur
    dataset yang siap digunakan untuk training.

    Struktur output:
        dataset/
            train/
                Safe/
                Violence/
                NSFW/
            validation/
                Safe/
                Violence/
                NSFW/
            test/
                Safe/
                Violence/
                NSFW/
    """

    def __init__(
        self,
        dataset_dir: Path = DATASET_DIR,
        safe_count: int = DEFAULT_SAFE_COUNT,
        violence_count: int = DEFAULT_VIOLENCE_COUNT,
        nsfw_count: int = DEFAULT_NSFW_COUNT,
        train_ratio: float = TRAIN_RATIO,
        val_ratio: float = VAL_RATIO,
        seed: int = RANDOM_SEED,
    ):
        self.dataset_dir = dataset_dir
        self.safe_count = safe_count
        self.violence_count = violence_count
        self.nsfw_count = nsfw_count
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed

        self._stats: Dict[str, Dict[str, int]] = {}

    def organize_class(
        self,
        class_name: str,
        source_dir: Path,
        n_images: int,
        recursive: bool = True,
    ) -> None:
        """
        Proses satu kelas: sample → split → copy ke folder.

        Parameters
        ----------
        class_name : str
            Nama kelas ('Safe', 'Violence', 'NSFW').
        source_dir : Path
            Direktori berisi gambar sumber untuk kelas ini.
        n_images : int
            Jumlah gambar target untuk kelas ini.
        recursive : bool
            Apakah scan sub-direktori.
        """
        logger.info("")
        logger.info("━" * 55)
        logger.info("  Memproses kelas: %s", class_name)
        logger.info("━" * 55)

        # 1. Kumpulkan semua gambar yang tersedia
        all_images = collect_images(source_dir, recursive=recursive)
        logger.info("  Total gambar ditemukan : %d", len(all_images))

        if len(all_images) == 0:
            logger.error("  ❌ Tidak ada gambar di: %s", source_dir)
            return

        # 2. Acak dan ambil sampel
        sampled = sample_images(all_images, n_images, seed=self.seed)
        logger.info("  Gambar di-sample       : %d", len(sampled))

        # 3. Split menjadi train / val / test
        train_imgs, val_imgs, test_imgs = split_images(
            sampled,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            seed=self.seed,
        )
        logger.info(
            "  Split hasil            : train=%d | val=%d | test=%d",
            len(train_imgs), len(val_imgs), len(test_imgs)
        )

        # 4. Copy ke folder tujuan
        prefix = class_name.lower()

        n_train = copy_images_to_split(
            train_imgs,
            self.dataset_dir / "train" / class_name,
            prefix=f"{prefix}_train",
        )
        n_val = copy_images_to_split(
            val_imgs,
            self.dataset_dir / "validation" / class_name,
            prefix=f"{prefix}_val",
        )
        n_test = copy_images_to_split(
            test_imgs,
            self.dataset_dir / "test" / class_name,
            prefix=f"{prefix}_test",
        )

        self._stats[class_name] = {
            "available": len(all_images),
            "sampled":   len(sampled),
            "train":     n_train,
            "validation": n_val,
            "test":      n_test,
        }

        logger.info(
            "  ✅ Selesai: train=%d | val=%d | test=%d",
            n_train, n_val, n_test,
        )

    def print_summary(self) -> None:
        """Cetak ringkasan statistik dataset yang dihasilkan."""
        logger.info("")
        logger.info("=" * 65)
        logger.info("  RINGKASAN DATASET")
        logger.info("=" * 65)
        logger.info(
            "  %-12s | %-10s | %-8s | %-8s | %-8s | %-8s",
            "Kelas", "Tersedia", "Diambil", "Train", "Val", "Test"
        )
        logger.info("  " + "-" * 63)

        total_train = total_val = total_test = total_sampled = 0
        for class_name, stats in self._stats.items():
            logger.info(
                "  %-12s | %-10d | %-8d | %-8d | %-8d | %-8d",
                class_name,
                stats["available"],
                stats["sampled"],
                stats["train"],
                stats["validation"],
                stats["test"],
            )
            total_sampled += stats["sampled"]
            total_train   += stats["train"]
            total_val     += stats["validation"]
            total_test    += stats["test"]

        logger.info("  " + "-" * 63)
        logger.info(
            "  %-12s | %-10s | %-8d | %-8d | %-8d | %-8d",
            "TOTAL", "", total_sampled, total_train, total_val, total_test,
        )
        logger.info("=" * 65)
        logger.info("")
        logger.info("  📁 Lokasi dataset: %s", self.dataset_dir)
        logger.info("")
        logger.info("  Langkah selanjutnya:")
        logger.info("    python src/train.py")
        logger.info("=" * 65)

    def verify_dataset(self) -> None:
        """Verifikasi struktur folder dan jumlah gambar di dataset final."""
        logger.info("")
        logger.info("🔍 Memverifikasi struktur dataset...")

        expected_splits = ["train", "validation", "test"]
        expected_classes = ["Safe", "Violence", "NSFW"]
        all_ok = True

        for split in expected_splits:
            for cls in expected_classes:
                dir_path = self.dataset_dir / split / cls
                if not dir_path.exists():
                    logger.warning("  ⚠️  Folder tidak ada: %s", dir_path)
                    all_ok = False
                    continue

                count = len([
                    f for f in dir_path.iterdir()
                    if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
                ])

                if count == 0:
                    logger.warning("  ⚠️  Folder kosong: %s", dir_path)
                    all_ok = False
                else:
                    logger.info("  ✅ %s/%s: %d gambar", split, cls, count)

        if all_ok:
            logger.info("  ✅ Dataset siap digunakan!")
        else:
            logger.warning("  ⚠️  Ada beberapa folder yang bermasalah.")


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def main(
    dataset_dir: str,
    safe_count: int,
    violence_count: int,
    nsfw_count: int,
    skip_download: bool,
    skip_nsfw: bool,
    manual_nsfw_dir: Optional[str] = None,
) -> None:
    """
    Pipeline utama: download → sample → split → copy.

    Parameters
    ----------
    safe_count : int
        Jumlah gambar untuk kelas Safe.
    violence_count : int
        Jumlah gambar untuk kelas Violence.
    nsfw_count : int
        Jumlah gambar untuk kelas NSFW.
    skip_download : bool
        Jika True, skip download dan gunakan raw_data/ yang sudah ada.
    skip_nsfw : bool
        Jika True, skip kelas NSFW (untuk test awal).
    manual_nsfw_dir : str | None
        Path manual ke folder berisi gambar NSFW (optional).
    """
    logger.info("=" * 65)
    logger.info("  🛡️  Dataset Setup — Image Content Moderation")
    logger.info("=" * 65)
    logger.info("  Target per kelas:")
    logger.info("    Safe     : %d gambar", safe_count)
    logger.info("    Violence : %d gambar", violence_count)
    logger.info("    NSFW     : %d gambar", nsfw_count)
    logger.info("  Split     : %.0f%% train | %.0f%% val | %.0f%% test",
                TRAIN_RATIO * 100, VAL_RATIO * 100, TEST_RATIO * 100)
    logger.info("=" * 65)

    # ── Cek prasyarat ─────────────────────────────────────────────────
    if not check_python_packages():
        sys.exit(1)

    if not skip_download and not check_kaggle_credentials():
        sys.exit(1)

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset_dir_path = Path(dataset_dir)
    dataset_dir_path.mkdir(parents=True, exist_ok=True)

    # ── Download datasets ──────────────────────────────────────────────
    safe_raw_dir     = RAW_DATA_DIR / "safe_raw"
    violence_raw_dir = RAW_DATA_DIR / "violence_raw"
    nsfw_raw_dir     = RAW_DATA_DIR / "nsfw_raw"

    if not skip_download:
        # ── Safe: Intel Image Classification ──────────────────────────
        logger.info("")
        logger.info("⬇️  [1/3] Mengunduh dataset SAFE...")
        try:
            intel_dir = download_kaggle_dataset(
                owner="puneet6060",
                dataset_name="intel-image-classification",
                download_dir=RAW_DATA_DIR / "kaggle_downloads",
            )

            # Cari direktori dengan gambar
            safe_source = find_source_directory(
                intel_dir,
                candidates=[
                    "seg_train/seg_train",
                    "seg_train",
                    "train",
                    "seg_test/seg_test",
                    "seg_test",
                ]
            )

            if safe_source:
                # Salin semua gambar ke safe_raw_dir (dari berbagai sub-kelas)
                safe_raw_dir.mkdir(parents=True, exist_ok=True)
                all_intel = collect_images(safe_source, recursive=True)
                logger.info("  Intel dataset: %d gambar ditemukan", len(all_intel))
                # Intel sudah terorganisir per sub-kelas (buildings, forest, dll)
                # Langsung gunakan direktori ini
                safe_raw_dir = safe_source
            else:
                logger.warning("  Tidak bisa menemukan gambar di Intel dataset.")
                safe_raw_dir.mkdir(parents=True, exist_ok=True)

        except RuntimeError as e:
            logger.error("Download Safe dataset gagal: %s", e)
            logger.info("Melewati download Safe. Pastikan folder ada di: %s", safe_raw_dir)
            safe_raw_dir.mkdir(parents=True, exist_ok=True)

        # ── Violence: Violence vs Non-Violence 11K ─────────────────────
        logger.info("")
        logger.info("⬇️  [2/3] Mengunduh dataset VIOLENCE...")
        try:
            violence_dir = download_kaggle_dataset(
                owner="ahmedmoorsy",
                dataset_name="violence-vs-non-violence-11k-images",
                download_dir=RAW_DATA_DIR / "kaggle_downloads",
            )

            violence_source = find_source_directory(
                violence_dir,
                candidates=[
                    "Violence",
                    "violence",
                    "train/Violence",
                    "data/Violence",
                    "Images/Violence",
                ]
            )

            if violence_source:
                violence_raw_dir = violence_source
                logger.info("  Violence source: %s", violence_raw_dir)
            else:
                # Coba cari folder apapun namanya yang ada "violen" di dalamnya
                for d in violence_dir.rglob("*"):
                    if d.is_dir() and "violen" in d.name.lower():
                        imgs = collect_images(d, recursive=False)
                        if imgs:
                            violence_raw_dir = d
                            logger.info("  Violence source: %s (%d gambar)", d, len(imgs))
                            break
                else:
                    violence_raw_dir.mkdir(parents=True, exist_ok=True)

        except RuntimeError as e:
            logger.error("Download Violence dataset gagal: %s", e)
            logger.info("Melewati download Violence.")
            violence_raw_dir.mkdir(parents=True, exist_ok=True)

        # ── NSFW: HuggingFace ──────────────────────────────────────────
        if not skip_nsfw:
            logger.info("")
            logger.info("⬇️  [3/3] Mengunduh dataset NSFW...")

            if manual_nsfw_dir:
                nsfw_raw_dir = Path(manual_nsfw_dir)
                logger.info("  Menggunakan folder NSFW manual: %s", nsfw_raw_dir)
            else:
                nsfw_raw_dir = download_nsfw_huggingface(
                    RAW_DATA_DIR / "hf_downloads",
                    n_images=nsfw_count,
                )
        else:
            logger.info("  ⏭️  Kelas NSFW dilewati (--skip_nsfw aktif)")
            nsfw_raw_dir.mkdir(parents=True, exist_ok=True)

    else:
        logger.info("⏭️  Skip download (--skip_download aktif)")
        logger.info("   Menggunakan raw_data/ yang sudah ada")
        safe_raw_dir     = RAW_DATA_DIR / "safe_raw"
        violence_raw_dir = RAW_DATA_DIR / "violence_raw"
        nsfw_raw_dir     = RAW_DATA_DIR / "nsfw_raw"

    # ── Bersihkan dataset lama jika ada ───────────────────────────────
    if dataset_dir_path.exists():
        existing_count = sum(1 for _ in dataset_dir_path.rglob("*") if _.is_file())
        if existing_count > 0:
            logger.info("")
            logger.info("⚠️  Dataset lama ditemukan (%d file). Membersihkan...", existing_count)
            shutil.rmtree(dataset_dir_path)
    dataset_dir_path.mkdir(parents=True, exist_ok=True)

    # ── Organize semua kelas ───────────────────────────────────────────
    organizer = DatasetOrganizer(
        dataset_dir=dataset_dir_path,
        safe_count=safe_count,
        violence_count=violence_count,
        nsfw_count=nsfw_count,
    )

    # Safe
    if safe_raw_dir.exists():
        organizer.organize_class("Safe", safe_raw_dir, safe_count)
    else:
        logger.warning("Folder Safe tidak ada: %s", safe_raw_dir)

    # Violence
    if violence_raw_dir.exists():
        organizer.organize_class("Violence", violence_raw_dir, violence_count)
    else:
        logger.warning("Folder Violence tidak ada: %s", violence_raw_dir)

    # NSFW
    if not skip_nsfw and nsfw_raw_dir.exists():
        organizer.organize_class("NSFW", nsfw_raw_dir, nsfw_count)
    elif not skip_nsfw:
        logger.warning("Folder NSFW tidak ada: %s", nsfw_raw_dir)

    # ── Verifikasi & ringkasan ─────────────────────────────────────────
    organizer.verify_dataset()
    organizer.print_summary()


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download & organisir dataset untuk Image Content Moderation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  # Download semua (default: 2000 per kelas)
  python setup_dataset.py

  # Custom jumlah
  python setup_dataset.py --safe_count 2000 --violence_count 2000 --nsfw_count 2000

  # Skip download (gunakan raw_data/ yang sudah ada)
  python setup_dataset.py --skip_download

  # Gunakan folder NSFW manual
  python setup_dataset.py --manual_nsfw_dir D:/mydata/nsfw_images

  # Skip kelas NSFW
  python setup_dataset.py --skip_nsfw

Prasyarat:
  pip install kaggle scikit-learn Pillow tqdm datasets
  Konfigurasi ~/.kaggle/kaggle.json (API key dari kaggle.com/settings)
        """,
    )

    parser.add_argument(
        "--dataset_dir", type=str, default="dataset",
        help="Direktori target untuk menaruh dataset (default: dataset)"
    )
    parser.add_argument(
        "--safe_count", type=int, default=DEFAULT_SAFE_COUNT,
        help=f"Jumlah gambar kelas Safe (default: {DEFAULT_SAFE_COUNT})"
    )
    parser.add_argument(
        "--violence_count", type=int, default=DEFAULT_VIOLENCE_COUNT,
        help=f"Jumlah gambar kelas Violence (default: {DEFAULT_VIOLENCE_COUNT})"
    )
    parser.add_argument(
        "--nsfw_count", type=int, default=DEFAULT_NSFW_COUNT,
        help=f"Jumlah gambar kelas NSFW (default: {DEFAULT_NSFW_COUNT})"
    )
    parser.add_argument(
        "--skip_download", action="store_true",
        help="Skip download, langsung organisir dari raw_data/ yang sudah ada"
    )
    parser.add_argument(
        "--skip_nsfw", action="store_true",
        help="Skip kelas NSFW (berguna untuk test awal 2-kelas)"
    )
    parser.add_argument(
        "--manual_nsfw_dir", type=str, default=None,
        help="Path manual ke folder berisi gambar NSFW"
    )

    args = parser.parse_args()

    main(
        dataset_dir=args.dataset_dir,
        safe_count=args.safe_count,
        violence_count=args.violence_count,
        nsfw_count=args.nsfw_count,
        skip_download=args.skip_download,
        skip_nsfw=args.skip_nsfw,
        manual_nsfw_dir=args.manual_nsfw_dir,
    )
