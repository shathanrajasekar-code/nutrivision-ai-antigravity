"""
cv_service.py — NutriVision AI Computer Vision Service
=======================================================
Model priority (auto-detected at startup):
  1. runs/classify/nutrivision_expanded/weights/best.pt  (60-class expanded, trained locally)
  2. runs/classify/train/weights/best.pt                 (previous 36-class custom model)
  3. yolov8n-cls.pt                                      (pretrained classification fallback)
  4. yolov8n.pt                                          (object detection fallback + ViT ensemble)
"""

from pathlib import Path
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# MODEL PRIORITY AUTO-DETECTION
# ─────────────────────────────────────────────────────────────────────────────
_MODEL_CANDIDATES = [
    Path('runs/classify/nutrivision_expanded/weights/best.pt'),  # 60-class expanded
    Path('runs/classify/train/weights/best.pt'),                  # 36-class previous
    Path('yolov8n-cls.pt'),                                       # pretrained cls
]

_DETECTION_MODEL_PATH = Path('yolov8n.pt')  # fallback object-detection model

def _load_best_model():
    for p in _MODEL_CANDIDATES:
        if p.exists():
            print(f"[cv_service] Loading classification model: {p}")
            return YOLO(str(p)), 'classify'
    print(f"[cv_service] No custom model found. Falling back to object-detection: {_DETECTION_MODEL_PATH}")
    return YOLO(str(_DETECTION_MODEL_PATH)), 'detect'

# ─────────────────────────────────────────────────────────────────────────────
# VISION ENSEMBLE: YOLO + SPECIALIZED INDIAN FOOD VIT
# ─────────────────────────────────────────────────────────────────────────────
_model, _mode = _load_best_model()

# Primary Indian Food Classifier (93% Accuracy on Indian dishes)
_indian_classifier = None
try:
    from transformers import pipeline as hf_pipeline
    _indian_classifier = hf_pipeline('image-classification', model='DrishtiSharma/finetuned-ViT-Indian-Food-Classification-v3')
    print("[cv_service] Specialized Indian Food ViT loaded.")
except Exception as e:
    print(f"[cv_service] Indian ViT not available: {e}")

# General ViT fallback
_gen_classifier = None
try:
    from transformers import pipeline as hf_pipeline
    if not _indian_classifier:
        _gen_classifier = hf_pipeline('image-classification', model='google/vit-base-patch16-224')
        print("[cv_service] General ViT fallback loaded.")
except Exception:
    pass

# COCO food classes that the base yolov8n.pt understands
_COCO_FOOD = {
    'apple', 'banana', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake',
}

def detect_food(image_path: str) -> list[dict]:
    """
    Run food detection/classification on an image.
    Uses an ensemble of YOLO (custom or pretrained) and specialized ViT.
    """
    results = _model(image_path, verbose=False)
    detections = []

    # 1. Run Specialized Indian Classification
    indian_label = None
    indian_conf = 0
    if _indian_classifier:
        try:
            res = _indian_classifier(image_path)
            # Find the top prediction
            top = res[0]
            indian_label = top['label'].lower().replace(' ', '_')
            indian_conf = top['score']
        except Exception:
            pass

    # 2. Process YOLO Detections
    if _mode == 'classify':
        for result in results:
            if hasattr(result, 'probs') and result.probs is not None:
                # Top-1 from classification YOLO
                name = result.names[int(result.probs.top1)]
                detections.append({'name': name, 'confidence': float(result.probs.top1conf)})
    else:
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                name   = _model.names[cls_id]
                if name.replace(' ', '_').lower() in _COCO_FOOD or name in _COCO_FOOD:
                    detections.append({'name': name.replace(' ', '_'), 'confidence': conf})

    # 3. Ensemble Logic
    # If Indian ViT is confident (>45%), it takes priority for Indian dishes
    # This specifically addresses Parota (often mis-called sandwich/salad)
    if indian_label and indian_conf > 0.45:
        # Check if YOLO gave a conflicting generic label
        generic_hits = [d for d in detections if d['name'] in ['salad', 'sandwich', 'pizza', 'donut']]
        
        if generic_hits or not detections:
            # Prioritize Indian label
            detections.insert(0, {'name': indian_label, 'confidence': indian_conf})
        elif indian_conf > 0.85:
            # Extreme confidence, override anything
            detections = [{'name': indian_label, 'confidence': indian_conf}]

    # 4. Final Fallback (General ViT)
    if not detections and _gen_classifier:
        try:
            res = _gen_classifier(image_path)
            detections.append({'name': res[0]['label'].lower(), 'confidence': res[0]['score']})
        except:
            pass

    if not detections:
        detections.append({'name': 'unknown_food_item', 'confidence': 0.5})

    # Sort by confidence + Deduplicate
    seen = set()
    final_dets = []
    # Re-sort to ensure best is top
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    for d in detections:
        if d['name'] not in seen:
            final_dets.append(d)
            seen.add(d['name'])
            
    return final_dets
