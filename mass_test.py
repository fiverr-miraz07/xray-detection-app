
from ultralytics import YOLO
import os
import glob

# Load model
model = YOLO('models/best.pt')

# Get all images
images = glob.glob('uploads/*.png') + glob.glob('uploads/*.jpg')
print(f"Found {len(images)} images in uploads.")

print("\n--- Running Inference (conf=0.01) ---")
for img_path in images:
    results = model(img_path, conf=0.01, verbose=False)
    for r in results:
        if len(r.boxes) > 0:
            print(f"\n{os.path.basename(img_path)}: {len(r.boxes)} detections")
            for box in r.boxes:
                 print(f"  - {model.names[int(box.cls[0])]}: {float(box.conf[0]):.4f}")
