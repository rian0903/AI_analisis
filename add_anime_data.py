import os
import sys
import json
import urllib.request
import urllib.parse
from pathlib import Path
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "raw_data"
DATASET_DIR = PROJECT_ROOT / "dataset_large"

ANIME_SAFE_RAW = RAW_DATA_DIR / "anime_safe_raw"
ANIME_VIOLENCE_RAW = RAW_DATA_DIR / "anime_violence_raw"
ANIME_NSFW_RAW = RAW_DATA_DIR / "anime_nsfw_raw"
LOCAL_CIVITAI_DIR = RAW_DATA_DIR / "hf_downloads" / "civitai_nsfw" / "images"

# Target Count
TARGET_COUNT = 2000
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

def fetch_safebooru_urls(tags, target_count):
    """Fetch image URLs and metadata from Safebooru API matching tags."""
    urls = []
    pid = 0
    limit = 1000
    seen_ids = set()
    
    encoded_tags = urllib.parse.quote(tags)
    
    logger.info(f"Fetching Safebooru URLs for tags: '{tags}'...")
    
    while len(urls) < target_count:
        url = f"https://safebooru.org/index.php?page=dapi&s=post&q=index&json=1&tags={encoded_tags}&limit={limit}&pid={pid}"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                res_content = response.read().decode('utf-8')
                if not res_content.strip():
                    logger.info("No more results from Safebooru API.")
                    break
                data = json.loads(res_content)
                if not data:
                    break
                
                added_in_page = 0
                for item in data:
                    item_id = item.get('id')
                    if item_id and item_id not in seen_ids:
                        seen_ids.add(item_id)
                        
                        # Use compressed sample_url if available to speed up downloads
                        img_url = item.get('sample_url')
                        if not img_url or not item.get('sample'):
                            img_url = item.get('file_url') or f"https://safebooru.org/images/{item.get('directory')}/{item.get('image')}"
                        
                        # Determine file extension from URL
                        url_path = urllib.parse.urlparse(img_url).path
                        ext = os.path.splitext(url_path)[1] or '.jpg'
                        clean_name = f"{item_id}{ext}"
                        
                        urls.append((item_id, img_url, clean_name))
                        added_in_page += 1
                        if len(urls) >= target_count:
                            break
                
                logger.info(f"Page {pid}: Found {len(data)} items, added {added_in_page} new URLs. Total collected: {len(urls)}/{target_count}")
                if added_in_page == 0:
                    # Avoid infinite loop if no new images are found
                    logger.info("No new unique images found on this page. Stopping search.")
                    break
                pid += 1
        except Exception as e:
            logger.error(f"Error fetching Safebooru page {pid}: {e}")
            break
            
    return urls[:target_count]

def download_image(item_id, url, filename, output_dir):
    """Download a single image and save to disk."""
    out_path = output_dir / filename
    if out_path.exists() and out_path.stat().st_size > 0:
        return True
        
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as response:
            with open(out_path, 'wb') as f:
                f.write(response.read())
        return True
    except Exception as e:
        logger.debug(f"Failed to download {url}: {e}")
        return False

def download_dataset_class(tags, target_count, output_dir):
    """Query Safebooru API and download target_count images to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    urls = fetch_safebooru_urls(tags, target_count)
    if not urls:
        logger.error(f"Could not retrieve any URLs for tags '{tags}'")
        return []
        
    logger.info(f"Starting download of {len(urls)} images using ThreadPoolExecutor...")
    downloaded_paths = []
    
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(download_image, item_id, url, name, output_dir): (item_id, url, name) for item_id, url, name in urls}
        
        success_count = 0
        for future in as_completed(futures):
            item_id, url, name = futures[future]
            if future.result():
                success_count += 1
                out_path = output_dir / name
                downloaded_paths.append(out_path)
                if success_count % 100 == 0:
                    logger.info(f"Progress: Downloaded {success_count}/{len(urls)} images successfully.")
                    
    logger.info(f"Class '{tags}' complete: successfully downloaded {success_count}/{len(urls)} images to {output_dir}")
    return downloaded_paths

def collect_nsfw_civitai(target_count, output_dir):
    """Copy 2000 anime NSFW images from the local civitai folder to raw_data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not LOCAL_CIVITAI_DIR.exists():
        logger.error(f"Civitai images folder not found at {LOCAL_CIVITAI_DIR}")
        return []
        
    all_images = [p for p in LOCAL_CIVITAI_DIR.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}]
    logger.info(f"Found {len(all_images)} local Civitai images.")
    
    if len(all_images) == 0:
        return []
        
    # Sample target count
    random.seed(RANDOM_SEED)
    sampled = random.sample(all_images, min(len(all_images), target_count))
    
    copied_paths = []
    for idx, src_path in enumerate(sampled):
        dst_path = output_dir / f"nsfw_{idx:05d}{src_path.suffix}"
        try:
            shutil.copy2(src_path, dst_path)
            copied_paths.append(dst_path)
        except Exception as e:
            logger.error(f"Failed to copy {src_path.name}: {e}")
            
    logger.info(f"NSFW class: copied {len(copied_paths)} images to {output_dir}")
    return copied_paths

def split_and_copy_to_dataset(image_paths, class_name):
    """Split image paths into train/val/test and copy to dataset_large folder."""
    random.seed(RANDOM_SEED)
    random.shuffle(image_paths)
    
    total = len(image_paths)
    n_train = int(total * TRAIN_RATIO)
    n_val = int(total * VAL_RATIO)
    
    train_paths = image_paths[:n_train]
    val_paths = image_paths[n_train:n_train + n_val]
    test_paths = image_paths[n_train + n_val:]
    
    splits = {
        "train": train_paths,
        "validation": val_paths,
        "test": test_paths
    }
    
    logger.info(f"Distributing {class_name} images: train={len(train_paths)}, validation={len(val_paths)}, test={len(test_paths)}")
    
    copied_counts = {"train": 0, "validation": 0, "test": 0}
    
    for split_name, paths in splits.items():
        dst_dir = DATASET_DIR / split_name / class_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, src in enumerate(paths):
            ext = src.suffix.lower()
            dst = dst_dir / f"anime_{class_name.lower()}_{idx:05d}{ext}"
            try:
                shutil.copy2(src, dst)
                copied_counts[split_name] += 1
            except Exception as e:
                logger.error(f"Failed to copy {src.name} to {dst}: {e}")
                
    logger.info(f"Finished copying {class_name} to dataset splits: {copied_counts}")
    return copied_counts

def main():
    logger.info("====================================================")
    logger.info("   Adding Anime/Illustration Data to dataset_large  ")
    logger.info("====================================================")
    
    # 1. Download Safe Anime
    # We mix scenery (1000) and general characters (1000) to get general safe anime art
    safe_images = []
    logger.info("--- Processing Safe Anime Class ---")
    safe_images += download_dataset_class("scenery", 1000, ANIME_SAFE_RAW)
    safe_images += download_dataset_class("1girl solo", 1000, ANIME_SAFE_RAW)
    
    # 2. Download Violence Anime
    # We mix fighting (1000) and sword battle scenes (1000)
    violence_images = []
    logger.info("--- Processing Violence Anime Class ---")
    violence_images += download_dataset_class("fighting", 1000, ANIME_VIOLENCE_RAW)
    violence_images += download_dataset_class("sword", 1000, ANIME_VIOLENCE_RAW)
    
    # 3. Collect NSFW Anime from Civitai
    logger.info("--- Processing NSFW Anime Class ---")
    nsfw_images = collect_nsfw_civitai(2000, ANIME_NSFW_RAW)
    
    # 4. Split and Distribute into dataset_large
    logger.info("--- Splitting and Distributing to dataset_large ---")
    
    stats = {}
    if len(safe_images) > 0:
        stats["Safe"] = split_and_copy_to_dataset(safe_images, "Safe")
    if len(violence_images) > 0:
        stats["Violence"] = split_and_copy_to_dataset(violence_images, "Violence")
    if len(nsfw_images) > 0:
        stats["NSFW"] = split_and_copy_to_dataset(nsfw_images, "NSFW")
        
    logger.info("====================================================")
    logger.info("Summary of added images:")
    for cls, split_stats in stats.items():
        logger.info(f"  {cls:10s} -> train={split_stats['train']}, val={split_stats['validation']}, test={split_stats['test']}")
    logger.info("====================================================")
    logger.info("Done! Now you can run training using: python src/train.py")

if __name__ == "__main__":
    main()
