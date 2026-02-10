
from ultralytics import YOLO
import cv2
import os
import numpy as np

# Load model
model_path = 'models/best.pt'
print(f"Loading model from {model_path}...")
model = YOLO(model_path)
print(f"Model classes: {model.names}")

# Test image path (using one from uploads)
img_path = 'uploads/20260210_181404_00000002_000.png'

if not os.path.exists(img_path):
    print(f"Image not found at {img_path}")
    exit(1)

print(f"\nTesting on image: {img_path}")

# 1. Raw prediction (no preprocessing)
print("\n--- Test 1: Raw Image ---")
results = model(img_path, conf=0.1) # low conf to see anything
for r in results:
    print(f"Detections: {len(r.boxes)}")
    for box in r.boxes:
        print(f"  Class: {int(box.cls[0])} ({model.names[int(box.cls[0])]}), Conf: {float(box.conf[0]):.4f}")

# 2. With CLAHE (as done in app.py)
print("\n--- Test 2: With CLAHE (app.py logic) ---")
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)
enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

results = model(enhanced_bgr, conf=0.1)
for r in results:
    print(f"Detections: {len(r.boxes)}")
    for box in r.boxes:
        print(f"  Class: {int(box.cls[0])} ({model.names[int(box.cls[0])]}), Conf: {float(box.conf[0]):.4f}")

# 3. Check image stats
print("\n--- Image Stats ---")
print(f"Original shape: {img.shape}")
print(f"Original min/max: {img.min()}/{img.max()}")
print(f"Enhanced min/max: {enhanced_bgr.min()}/{enhanced_bgr.max()}")
