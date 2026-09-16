"""
install_dataset_deps.py
=======================
Script instalasi semua dependensi yang diperlukan untuk setup_dataset.py
Jalankan SEKALI sebelum menjalankan setup_dataset.py

Cara pakai:
    python install_dataset_deps.py
"""
import subprocess
import sys

packages = [
    "kaggle",
    "scikit-learn",
    "Pillow",
    "tqdm",
    "datasets",           # HuggingFace datasets
    "huggingface_hub",    # HuggingFace hub
    "requests",
]

print("=" * 50)
print("  Installing dataset dependencies...")
print("=" * 50)

for pkg in packages:
    print(f"\n📦 Installing {pkg}...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", pkg, "-q"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"  ✅ {pkg} installed successfully")
    else:
        print(f"  ⚠️  {pkg} might have issues: {result.stderr[:100]}")

print("\n" + "=" * 50)
print("  ✅ All dependencies installed!")
print("=" * 50)
print("\nLangkah selanjutnya:")
print("  1. Dapatkan Kaggle API key dari https://www.kaggle.com/settings")
print("  2. Simpan kaggle.json ke C:/Users/<nama_kamu>/.kaggle/kaggle.json")
print("  3. Jalankan: python setup_dataset.py")
