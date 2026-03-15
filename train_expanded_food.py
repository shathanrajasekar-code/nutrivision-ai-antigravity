"""
NutriVision AI — Expanded Food Training Pipeline
=================================================
Sources (in priority order):
  1. Kaggle datasets  (requires KAGGLE_USERNAME + KAGGLE_KEY env vars or ~/.kaggle/kaggle.json)
  2. Roboflow datasets (requires ROBOFLOW_API_KEY env var or passed via CLI)
  3. Bing Image Downloader (no auth, web scraping — fallback for remaining classes)

Usage
-----
  # Full pipeline (download + train):
  python train_expanded_food.py

  # Skip download if data already exists:
  python train_expanded_food.py --skip-download

  # Use only bing downloader (no API keys needed):
  python train_expanded_food.py --source bing

  # Set Roboflow API key inline:
  python train_expanded_food.py --roboflow-key YOUR_KEY_HERE

  # Custom epochs / image size / batch:
  python train_expanded_food.py --epochs 50 --imgsz 224 --batch 32
"""

import os
import sys
import shutil
import random
import argparse
import zipfile
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─────────────────────────────────────────────────────────────────────────────
# 60-CLASS FOOD TAXONOMY
# Added: parota, uttapam, poori, pesarattu, appam, puttu, halwa, khichdi,
#        momo, kebab, spring_roll, oats, smoothie_bowl, avocado_toast, etc.
# ─────────────────────────────────────────────────────────────────────────────
CLASSES = [
    # ── Indian Breads & Flatbreads ──────────────────────────────────────────
    'parota', 'chapati', 'roti', 'naan', 'puri', 'bhatura',
    'appam', 'uttapam', 'pesarattu', 'puttu',

    # ── South Indian ────────────────────────────────────────────────────────
    'dosa', 'idli', 'vada', 'upma', 'poha', 'pongal', 'rasam',

    # ── North Indian Mains ──────────────────────────────────────────────────
    'biryani', 'dal', 'dal_makhani', 'paneer', 'butter_chicken',
    'chole_bhature', 'palak_paneer', 'rajma', 'aloo_gobi', 'khichdi',

    # ── Snacks & Street Food ────────────────────────────────────────────────
    'samosa', 'kofta', 'vada_pav', 'pav_bhaji', 'bhel_puri', 'halwa',

    # ── Indian Thali / Combos ───────────────────────────────────────────────
    'indian_thali', 'thali',

    # ── International Fast Food ─────────────────────────────────────────────
    'pizza', 'burger', 'hot_dog', 'french_fries', 'sandwich',
    'tacos', 'shawarma', 'falafel',

    # ── Asian ───────────────────────────────────────────────────────────────
    'sushi', 'ramen', 'pad_thai', 'fried_rice', 'dim_sum',
    'spring_roll', 'momo', 'kebab',

    # ── Western Mains ───────────────────────────────────────────────────────
    'pasta', 'steak', 'fried_chicken', 'omelette',

    # ── Fruits ──────────────────────────────────────────────────────────────
    'apple', 'banana', 'mango', 'orange', 'watermelon',

    # ── Healthy / Bowl formats ──────────────────────────────────────────────
    'salad', 'soup', 'smoothie_bowl', 'avocado_toast', 'oats',

    # ── Desserts / Sweets ───────────────────────────────────────────────────
    'ice_cream', 'waffle', 'pancake', 'donut', 'gulab_jamun',
]

IMAGES_PER_CLASS = 120      # Images per class target
TRAIN_SPLIT      = 0.80     # 80 % train / 20 % val
DATASET_DIR      = Path('datasets/expanded_food')
RAW_DIR          = DATASET_DIR / 'raw'
TRAIN_DIR        = DATASET_DIR / 'train'
VAL_DIR          = DATASET_DIR / 'val'

# ─────────────────────────────────────────────────────────────────────────────
# KAGGLE DATASET REGISTRY
# (dataset_slug, subfolder_with_class_dirs_inside_zip)
# ─────────────────────────────────────────────────────────────────────────────
KAGGLE_DATASETS = [
    # 20-class Indian food dataset (classification format, ready for YOLOv8)
    ("l33tc0d3r/indian-food-images",               "Indian Food Images"),
    # 35-class Indian + Western (Kaggle)
    ("harishkumardatalab/food-image-classification","food_dataset"),
    # 20-class Indian food detection (already split train/val)
    ("iamsouravbanerjee/indian-food-images-dataset","Indian Food"),
]

# ─────────────────────────────────────────────────────────────────────────────
# ROBOFLOW DATASET REGISTRY
# (workspace, project, version)
# ─────────────────────────────────────────────────────────────────────────────
ROBOFLOW_DATASETS = [
    # South Indian Food Detection — 31 classes
    ("south-indian-food", "south-indian-food-detection", 1),
    # General Indian food (~4k images, 20 classes)
    ("indian-food-detection", "indian-food-detection", 1),
]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
VALID_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

def is_image(p: Path) -> bool:
    return p.suffix.lower() in VALID_EXTS

def safe_copy(src: Path, dst: Path):
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    except Exception as e:
        pass  # silently skip broken images

def split_and_move(src_dir: Path, class_name: str):
    """Take images from src_dir, split 80/20, copy to TRAIN_DIR/VAL_DIR."""
    images = [p for p in src_dir.rglob('*') if is_image(p)]
    if not images:
        return 0
    random.shuffle(images)
    split = int(len(images) * TRAIN_SPLIT)
    for i, img in enumerate(images[:split]):
        safe_copy(img, TRAIN_DIR / class_name / f"{class_name}_{i:04d}{img.suffix}")
    for i, img in enumerate(images[split:]):
        safe_copy(img, VAL_DIR / class_name / f"{class_name}_v{i:04d}{img.suffix}")
    return len(images)

def existing_count(class_name: str) -> int:
    """Return how many images already exist in train for this class."""
    d = TRAIN_DIR / class_name
    if not d.exists():
        return 0
    return len([p for p in d.iterdir() if is_image(p)])

def _fuzzy_match(class_name: str, available: list) -> str | None:
    """Find the best matching folder name for a class."""
    cn = class_name.lower().replace('_', ' ')
    for a in available:
        if cn in a.lower() or a.lower().replace('_','') == cn.replace(' ',''):
            return a
    for a in available:
        if any(word in a.lower() for word in cn.split()):
            return a
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 1: KAGGLE
# ─────────────────────────────────────────────────────────────────────────────
def download_kaggle(classes_needed: set) -> set:
    """Download Kaggle datasets and copy matching class folders. Returns classes filled."""
    try:
        import kaggle  # pip install kaggle
    except ImportError:
        print("[Kaggle] kaggle package not installed. Run: pip install kaggle")
        return set()

    filled = set()
    kaggle_raw = RAW_DIR / 'kaggle'
    kaggle_raw.mkdir(parents=True, exist_ok=True)

    for slug, subfolder in KAGGLE_DATASETS:
        dest = kaggle_raw / slug.replace('/', '_')
        if dest.exists():
            print(f"[Kaggle] {slug} already downloaded, using cached.")
        else:
            print(f"[Kaggle] Downloading {slug} ...")
            try:
                import kaggle as kg
                kg.api.authenticate()
                kg.api.dataset_download_files(slug, path=str(dest), unzip=True, quiet=False)
            except Exception as e:
                print(f"[Kaggle] ✗ {slug}: {e}")
                continue

        # Walk the extracted directory and match folders to our class list
        print(f"[Kaggle] Scanning {dest} for class folders ...")
        for root, dirs, files in os.walk(dest):
            for d in dirs:
                cls = _fuzzy_match(d, [c for c in classes_needed])
                if cls:
                    src = Path(root) / d
                    n = split_and_move(src, cls)
                    if n > 0:
                        print(f"  ✓ {cls}: {n} images from Kaggle/{slug}")
                        filled.add(cls)

    return filled


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 2: ROBOFLOW
# ─────────────────────────────────────────────────────────────────────────────
def download_roboflow(classes_needed: set, rf_api_key: str) -> set:
    """Download Roboflow datasets and organise by class. Returns classes filled."""
    if not rf_api_key:
        print("[Roboflow] No API key. Skipping. (Pass --roboflow-key or set ROBOFLOW_API_KEY env var)")
        return set()
    try:
        from roboflow import Roboflow  # pip install roboflow
    except ImportError:
        print("[Roboflow] roboflow package not installed. Run: pip install roboflow")
        return set()

    filled = set()
    rf_raw = RAW_DIR / 'roboflow'
    rf_raw.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=rf_api_key)

    for workspace, project_name, version in ROBOFLOW_DATASETS:
        dest = rf_raw / f"{workspace}_{project_name}_v{version}"
        if dest.exists():
            print(f"[Roboflow] {project_name} already downloaded.")
        else:
            print(f"[Roboflow] Downloading {workspace}/{project_name} v{version} ...")
            try:
                proj = rf.workspace(workspace).project(project_name)
                v    = proj.version(version)
                ds   = v.download("folder", location=str(dest))
            except Exception as e:
                print(f"[Roboflow] ✗ {project_name}: {e}")
                continue

        # Scan for class folders (Roboflow folder format: train/<class>/*.jpg)
        for split_name in ['train', 'valid', 'test']:
            split_path = dest / split_name
            if not split_path.exists():
                continue
            for class_dir in split_path.iterdir():
                if not class_dir.is_dir():
                    continue
                cls = _fuzzy_match(class_dir.name, list(classes_needed))
                if cls:
                    n = split_and_move(class_dir, cls)
                    if n > 0:
                        print(f"  ✓ {cls}: {n} images from Roboflow/{project_name}")
                        filled.add(cls)

    return filled


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 3: BING IMAGE DOWNLOADER (fallback)
# ─────────────────────────────────────────────────────────────────────────────
def _bing_download_class(cls: str, need: int):
    """Download `need` more images for `cls` via Bing."""
    try:
        from bing_image_downloader import downloader  # pip install bing-image-downloader
    except ImportError:
        print(f"[Bing] bing-image-downloader not installed. Run: pip install bing-image-downloader")
        return 0

    bing_raw = RAW_DIR / 'bing'
    # Build a food-specific, high-quality query
    query_map = {
        'parota':        'kerala parota layered flaky bread food photo',
        'uttapam':       'uttapam south Indian pancake toppings photo',
        'pesarattu':     'pesarattu green moong dosa Andhra food photo',
        'appam':         'appam rice hopper South Indian food photo',
        'puttu':         'puttu steamed rice cake Kerala food photo',
        'pongal':        'ven pongal South Indian rice dish photo',
        'rasam':         'rasam South Indian soup bowl food photo',
        'dal_makhani':   'dal makhani creamy lentil Indian food photo',
        'palak_paneer':  'palak paneer spinach curry Indian food photo',
        'rajma':         'rajma red kidney bean curry food photo',
        'aloo_gobi':     'aloo gobi potato cauliflower curry food photo',
        'khichdi':       'khichdi rice lentil comfort food Indian photo',
        'vada_pav':      'vada pav Mumbai street food photo',
        'pav_bhaji':     'pav bhaji street food Mumbai plate photo',
        'bhel_puri':     'bhel puri Indian chaat street food photo',
        'halwa':         'halwa Indian dessert sweets food photo',
        'gulab_jamun':   'gulab jamun Indian sweet dessert syrup photo',
        'smoothie_bowl': 'smoothie bowl acai fruit breakfast food photo',
        'avocado_toast': 'avocado toast brunch food photo',
        'momo':          'momo steamed dumplings Tibetan Nepal food photo',
        'kebab':         'seekh kebab grilled skewer food photo',
        'spring_roll':   'spring roll crispy Chinese food photo',
        'oats':          'oatmeal bowl breakfast healthy food photo',
    }
    query = query_map.get(cls, f'delicious {cls.replace("_", " ")} food plate photo')
    raw_cls_dir = bing_raw / query

    try:
        downloader.download(
            query,
            limit=need + 20,
            output_dir=str(bing_raw),
            adult_filter_off=True,
            force_replace=False,
            timeout=15,
            verbose=False
        )
    except Exception as e:
        print(f"[Bing] ✗ {cls}: {e}")
        return 0

    # Move to train/val
    if raw_cls_dir.exists():
        n = split_and_move(raw_cls_dir, cls)
        print(f"  ✓ {cls}: {n} images from Bing")
        return n
    return 0


def download_bing(classes_needed: set):
    """Parallel Bing download for all classes that still need images."""
    print(f"\n[Bing] Downloading {len(classes_needed)} classes via Bing Image Downloader ...")
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {}
        for cls in classes_needed:
            have  = existing_count(cls)
            need  = max(0, IMAGES_PER_CLASS - have)
            if need > 0:
                futures[ex.submit(_bing_download_class, cls, need)] = cls
            else:
                print(f"  ✓ {cls}: already has {have} images, skipping Bing.")
        for f in as_completed(futures):
            pass  # results already printed inside _bing_download_class


# ─────────────────────────────────────────────────────────────────────────────
# DATASET REPORT
# ─────────────────────────────────────────────────────────────────────────────
def print_dataset_report():
    print("\n" + "═"*60)
    print("  DATASET REPORT")
    print("═"*60)
    total_train = total_val = 0
    low_classes = []
    for cls in sorted(CLASSES):
        t = len([p for p in (TRAIN_DIR/cls).rglob('*') if is_image(p)]) if (TRAIN_DIR/cls).exists() else 0
        v = len([p for p in (VAL_DIR/cls).rglob('*')   if is_image(p)]) if (VAL_DIR/cls).exists()   else 0
        total_train += t; total_val += v
        status = "✓" if t >= 30 else "⚠ LOW"
        print(f"  {status:6}  {cls:<25} train={t:4d}  val={v:3d}")
        if t < 30:
            low_classes.append(cls)
    print("─"*60)
    print(f"  TOTAL  {'':25} train={total_train}  val={total_val}")
    if low_classes:
        print(f"\n  ⚠ Classes with <30 training images (may underfit): {low_classes}")
    print("═"*60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# YAML CONFIG (YOLOv8 classification doesn't need yaml, but generate for reference)
# ─────────────────────────────────────────────────────────────────────────────
def write_yaml():
    yaml_path = DATASET_DIR / 'dataset.yaml'
    yaml_path.write_text(
        f"# NutriVision AI — {len(CLASSES)}-class food classification dataset\n"
        f"path: {DATASET_DIR.resolve()}\n"
        f"train: train\n"
        f"val: val\n"
        f"nc: {len(CLASSES)}\n"
        f"names:\n" + "\n".join(f"  - {c}" for c in CLASSES) + "\n"
    )
    print(f"[Config] dataset.yaml written to {yaml_path}")


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train_model(epochs: int, imgsz: int, batch: int, resume: bool):
    from ultralytics import YOLO

    print("\n" + "═"*60)
    print(f"  TRAINING  — {len(CLASSES)} classes | {epochs} epochs | imgsz={imgsz} | batch={batch}")
    print("═"*60)

    # Use the previously trained model if available (transfer learning / fine-tune)
    best_prev = Path('runs/classify/train/weights/best.pt')
    if best_prev.exists() and resume:
        base_model = str(best_prev)
        print(f"[Train] Resuming / fine-tuning from previous best: {base_model}")
    else:
        base_model = 'yolov8n-cls.pt'
        print(f"[Train] Starting fresh from pretrained: {base_model}")

    model = YOLO(base_model)

    results = model.train(
        data=str(DATASET_DIR.resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=0,
        project='runs/classify',
        name='nutrivision_expanded',
        exist_ok=True,
        patience=15,          # early stopping
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        augment=True,
        degrees=10.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )

    best_path = Path('runs/classify/nutrivision_expanded/weights/best.pt')
    print(f"\n[Train] ✓ Complete. Best model saved to: {best_path}")
    print(f"[Train] Update cv_service.py to load: '{best_path}'")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="NutriVision AI — Expanded Food Training Pipeline")
    parser.add_argument('--skip-download',  action='store_true', help='Skip data download, train on existing dataset')
    parser.add_argument('--source',         choices=['all','bing','kaggle','roboflow'], default='all',
                        help='Data source to use (default: all)')
    parser.add_argument('--roboflow-key',   default=os.environ.get('ROBOFLOW_API_KEY',''),
                        help='Roboflow API key (or set ROBOFLOW_API_KEY env var)')
    parser.add_argument('--epochs',         type=int, default=40,  help='Training epochs (default: 40)')
    parser.add_argument('--imgsz',          type=int, default=160, help='Image size (default: 160)')
    parser.add_argument('--batch',          type=int, default=16,  help='Batch size (default: 16)')
    parser.add_argument('--resume',         action='store_true',    help='Resume/fine-tune from previous best.pt')
    parser.add_argument('--report-only',    action='store_true',    help='Print dataset report and exit')
    args = parser.parse_args()

    # Create all class dirs upfront
    for cls in CLASSES:
        (TRAIN_DIR / cls).mkdir(parents=True, exist_ok=True)
        (VAL_DIR   / cls).mkdir(parents=True, exist_ok=True)

    if args.report_only:
        print_dataset_report()
        return

    if not args.skip_download:
        classes_needed = set(CLASSES)

        if args.source in ('all', 'kaggle'):
            filled = download_kaggle(classes_needed)
            classes_needed -= filled

        if args.source in ('all', 'roboflow'):
            filled = download_roboflow(classes_needed, args.roboflow_key)
            classes_needed -= filled

        if args.source in ('all', 'bing'):
            # For bing, top up ALL classes (not just unfilled) to reach IMAGES_PER_CLASS
            download_bing(set(CLASSES))  # always top-up regardless of prior sources

    print_dataset_report()
    write_yaml()
    train_model(args.epochs, args.imgsz, args.batch, args.resume)


if __name__ == '__main__':
    main()
