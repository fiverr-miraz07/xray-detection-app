
import requests
import os

url = 'http://127.0.0.1:5000/predict'
img_path = 'uploads/20260210_043919_00016052_000.png'

if not os.path.exists(img_path):
    print(f"Image not found: {img_path}")
    exit(1)

files = {'file': open(img_path, 'rb')}
response = requests.post(url, files=files)

if response.status_code == 200:
    data = response.json()
    print(f"Success! Detections: {len(data['detections'])}")
    for d in data['detections']:
        print(f"  - {d['class']}: {d['confidence']}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
