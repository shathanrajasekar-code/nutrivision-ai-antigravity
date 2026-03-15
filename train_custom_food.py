import os
import shutil
from ultralytics import YOLO
from bing_image_downloader import downloader
import time
from concurrent.futures import ThreadPoolExecutor
import random

CLASSES = [
    'indian_thali', 'pizza', 'burger', 'salad', 'apple', 'banana',
    'sushi', 'pasta', 'taco', 'ice_cream', 'steak', 'french_fries',
    'hot_dog', 'donut', 'sandwich', 'fried_chicken', 'pancake',
    'waffle', 'ramen', 'biryani', 'curry', 'dosa', 'idli', 
    'samosa', 'paneer_tikka', 'chole_bhature', 'naan', 'butter_chicken', 
    'dim_sum', 'pho', 'croissant', 'tiramisu', 'falafel', 'shawarma', 
    'pad_thai', 'macarons'
]
IMAGES_PER_CLASS = 100 # Increased amount for robust training
dataset_dir = 'datasets/custom_food'

def download_class_images(cls):
    print(f"Downloading images for {cls}...")
    try:
        query = f'delicious {cls.replace("_", " ")} plate photo'
        downloader.download(query, 
                            limit=IMAGES_PER_CLASS,  
                            output_dir=os.path.join(dataset_dir, 'raw'), 
                            adult_filter_off=True, 
                            force_replace=False, 
                            timeout=10,
                            verbose=False)
    except Exception as e:
         print(f"Dataset gathering warning for {cls}:", e)

def download_images():
    print("Gathering training data from the web using bing-image-downloader...")
    raw_dir = os.path.join(dataset_dir, 'raw')
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    
    # Download concurrently
    with ThreadPoolExecutor(max_workers=5) as executor:
        for _ in executor.map(download_class_images, CLASSES):
            pass
            
    print("Splitting images into train and val...")
    for cls in CLASSES:
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
        
        query = f'delicious {cls.replace("_", " ")} plate photo'
        cls_raw_dir = os.path.join(raw_dir, query)
        
        if not os.path.exists(cls_raw_dir):
            continue
            
        images = os.listdir(cls_raw_dir)
        random.shuffle(images)
        
        split_idx = int(len(images) * 0.8)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]
        
        for idx, img in enumerate(train_imgs):
            src = os.path.join(cls_raw_dir, img)
            dst = os.path.join(train_dir, cls, f"{cls}_{idx}.jpg")
            try: shutil.move(src, dst)
            except: pass
            
        for idx, img in enumerate(val_imgs):
            src = os.path.join(cls_raw_dir, img)
            dst = os.path.join(val_dir, cls, f"{cls}_{idx}.jpg")
            try: shutil.move(src, dst)
            except: pass

def train_model():
    print("Starting custom YOLO model training on your new dataset...")
    # Load a pretrained classification model
    model = YOLO('yolov8n-cls.pt')
    
    # Train the model - 30 epochs for better learning on the expanded dataset
    results = model.train(data=os.path.abspath(dataset_dir), epochs=30, imgsz=128, batch=16, workers=0)
    print("\nTraining Complete! Custom model saved to:", model.ckpt_path)
    return model

if __name__ == '__main__':
    download_images()
    train_model()
