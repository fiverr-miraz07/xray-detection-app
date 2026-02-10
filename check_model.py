
from ultralytics import YOLO

try:
    model = YOLO('yolov8m.pt')
    print("Model classes:")
    print(model.names)
except Exception as e:
    print(f"Error loading model: {e}")
