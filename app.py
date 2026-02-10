"""
Medical X-Ray Detection Web Application
Using YOLOv8 for real-time disease detection in chest X-rays
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import base64
from io import BytesIO
import json
from datetime import datetime

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Disease classes
DISEASE_CLASSES = {
    0: 'Atelectasis',
    1: 'Cardiomegaly',
    2: 'Effusion',
    3: 'Infiltrate',
    4: 'Mass',
    5: 'Nodule',
    6: 'Pneumonia',
    7: 'Pneumothorax'
}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Global model variable
model = None
MODEL_PATH = 'models/best.pt'  # Path to your trained YOLOv8 model


def load_model():
    """Load the YOLOv8 model"""
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            print(f"Loading model from {MODEL_PATH}...")
            model = YOLO(MODEL_PATH)
            print("✓ Custom trained model loaded successfully!")
        else:
            print(f"⚠ Warning: Trained model not found at {MODEL_PATH}")
            print("⚠ Please train your model and export it to webapp/models/best.pt")
            print("⚠ For now, no detections will be shown (model not suitable for X-ray disease detection)")
            # Load a model but it won't detect our disease classes
            model = YOLO('yolov8n.pt')  # Use nano for faster inference as placeholder
    return model


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def apply_clahe(image):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    for contrast enhancement
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Convert back to BGR
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    return enhanced_bgr


def preprocess_image(image_path, apply_enhancement=True):
    """
    Preprocess the uploaded X-ray image
    """
    try:
        # Read image using PIL first (more robust)
        pil_img = Image.open(image_path)
        # Convert to RGB if necessary
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        # Convert PIL to numpy array for OpenCV
        img = np.array(pil_img)
        # Convert RGB to BGR for OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise ValueError(f"Unable to read image: {str(e)}")
    
    if img is None or img.size == 0:
        raise ValueError("Image is empty or corrupted")
    
    # Apply CLAHE if requested
    if apply_enhancement:
        img = apply_clahe(img)
    
    return img


def image_to_base64(image):
    """Convert OpenCV image to base64 string for web display"""
    _, buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"


def draw_detections(image, results):
    """
    Draw bounding boxes and labels on the image
    """
    annotated_image = image.copy()
    
    # Define colors for different classes (BGR format)
    colors = [
        (255, 0, 0),      # Blue for Infiltrate
        (0, 255, 0),      # Green for Atelectasis
        (0, 0, 255),      # Red for Pneumonia
        (255, 255, 0),    # Cyan for Cardiomegaly
        (255, 0, 255),    # Magenta for Effusion
        (0, 255, 255),    # Yellow for Pneumothorax
        (128, 0, 128),    # Purple for Mass
        (255, 165, 0)     # Orange for Nodule
    ]
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get class and confidence
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Skip if class ID is not in our disease classes
            if cls not in DISEASE_CLASSES:
                continue
            
            # Get color for this class
            color = colors[cls % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label = f"{DISEASE_CLASSES[cls]}: {conf:.2f}"
            
            # Get label size
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw label background
            cv2.rectangle(
                annotated_image,
                (x1, y1 - label_height - baseline - 5),
                (x1 + label_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated_image,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
    
    return annotated_image


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle image upload and perform prediction
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, bmp, tiff'}), 400
        
        # Save uploaded file first (more reliable approach)
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the file
        file.save(filepath)
        
        # Verify file exists
        if not os.path.exists(filepath):
            return jsonify({'error': 'Failed to save uploaded file'}), 500
        
        # Read image using PIL
        try:
            pil_img = Image.open(filepath)
            
            # Convert to RGB if necessary
            if pil_img.mode not in ['RGB', 'L']:
                pil_img = pil_img.convert('RGB')
            elif pil_img.mode == 'L':
                pil_img = pil_img.convert('RGB')
            
            # Convert PIL to numpy array
            img = np.array(pil_img)
            
            # Convert RGB to BGR for OpenCV
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
        except Exception as e:
            # Clean up the file if it failed
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Unable to read image: {str(e)}. Please upload a valid image file.'}), 400
        
        if img is None or img.size == 0:
            return jsonify({'error': 'Image is empty or corrupted.'}), 400
        
        # Save uploaded file for record keeping
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(filepath, img)
        
        # Get enhancement option
        apply_enhancement = request.form.get('enhance', 'true').lower() == 'true'
        
        # Apply CLAHE if requested
        if apply_enhancement:
            preprocessed_img = apply_clahe(img)
        else:
            preprocessed_img = img.copy()
        
        # Load model
        model = load_model()
        
        # Perform inference
        results = model(img, conf=0.01)  # Run on raw image, not enhanced. Low conf for sensitivity.
        
        # Draw detections
        annotated_img = draw_detections(preprocessed_img, results)
        
        # Save result image
        result_filename = f"result_{filename}"
        result_filepath = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        cv2.imwrite(result_filepath, annotated_img)
        
        # Prepare detection results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Check if class ID is valid
                if cls not in DISEASE_CLASSES:
                    print(f"Warning: Unknown class ID {cls}, skipping detection")
                    continue
                
                detections.append({
                    'class': DISEASE_CLASSES[cls],
                    'class_id': cls,
                    'confidence': round(conf, 4),
                    'bbox': {
                        'x1': round(x1, 2),
                        'y1': round(y1, 2),
                        'x2': round(x2, 2),
                        'y2': round(y2, 2)
                    }
                })
        
        # Convert images to base64 for display
        original_base64 = image_to_base64(img)
        preprocessed_base64 = image_to_base64(preprocessed_img)
        result_base64 = image_to_base64(annotated_img)
        
        # Summary statistics
        summary = {
            'total_detections': len(detections),
            'diseases_found': list(set([d['class'] for d in detections])),
            'max_confidence': max([d['confidence'] for d in detections]) if detections else 0,
            'avg_confidence': sum([d['confidence'] for d in detections]) / len(detections) if detections else 0
        }
        
        # Check if using custom model
        model_warning = None
        if not os.path.exists(MODEL_PATH):
            model_warning = "⚠ Using fallback model. Please train and export your custom model to get X-ray disease detections. See README.md for instructions."
        
        response_data = {
            'success': True,
            'detections': detections,
            'summary': summary,
            'images': {
                'original': original_base64,
                'preprocessed': preprocessed_base64,
                'result': result_base64
            },
            'timestamp': timestamp
        }
        
        if model_warning:
            response_data['warning'] = model_warning
        
        return jsonify(response_data)
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error during prediction: {str(e)}")
        print(f"Full traceback:\n{error_trace}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/results/<filename>')
def result_file(filename):
    """Serve result files"""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)


@app.route('/health')
def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not loaded"
    model_exists = os.path.exists(MODEL_PATH)
    
    return jsonify({
        'status': 'healthy',
        'model_status': model_status,
        'model_exists': model_exists,
        'model_path': MODEL_PATH
    })


if __name__ == '__main__':
    # Load model at startup
    print("Starting Medical X-Ray Detection Web Application...")
    load_model()
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
