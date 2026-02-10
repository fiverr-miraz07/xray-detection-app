# Medical X-Ray AI Detection Web Application

A scalable web application for real-time disease detection in chest X-ray images using YOLOv8 deep learning model.

## 🎯 Features

- **Real-time Detection**: Upload X-ray images and get instant AI-powered disease detection
- **8 Disease Classes**: Detects Infiltrate, Atelectasis, Pneumonia, Cardiomegaly, Effusion, Pneumothorax, Mass, and Nodule
- **CLAHE Enhancement**: Optional contrast enhancement for better image quality
- **Interactive UI**: Modern, responsive web interface with drag-and-drop upload
- **Detailed Results**: View bounding boxes, confidence scores, and detection statistics
- **Image Comparison**: Compare original, enhanced, and annotated images side-by-side

## 📁 Project Structure

```
webapp/
├── app.py                  # Flask backend application
├── requirements.txt        # Python dependencies
├── static/
│   ├── styles.css         # Styling
│   └── script.js          # Frontend JavaScript
├── templates/
│   └── index.html         # Main HTML template
├── models/
│   └── best.pt           # Your trained YOLOv8 model (place here)
├── uploads/              # Uploaded images (auto-created)
└── results/              # Detection results (auto-created)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Trained YOLOv8 model (from your notebook training)

### Installation

1. **Navigate to the webapp directory**:
   ```powershell
   cd "c:\Users\Kamrul\OneDrive\Desktop\Thesis_final\webapp"
   ```

2. **Create a virtual environment** (recommended):
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

4. **Export your trained model**:
   
   Run the last cell in your Jupyter notebook to export the trained model:
   ```python
   # This will copy your best.pt model to webapp/models/
   ```
   
   Or manually copy your trained model:
   ```powershell
   mkdir models
   copy "path\to\your\runs\detect\train\weights\best.pt" "models\best.pt"
   ```

### Running the Application

1. **Start the Flask server**:
   ```powershell
   python app.py
   ```

2. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

3. **Upload an X-ray image** and get instant detections!

## 🔧 Configuration

### Model Path
Edit `app.py` to change the model path:
```python
MODEL_PATH = 'models/best.pt'  # Update this path
```

### Server Settings
Modify the server configuration at the bottom of `app.py`:
```python
app.run(
    debug=True,      # Set to False in production
    host='0.0.0.0',  # Allow external connections
    port=5000        # Change port if needed
)
```

### Upload Limits
Adjust file size limits in `app.py`:
```python
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
```

## 📊 API Endpoints

### `POST /predict`
Upload an image for detection.

**Request:**
- Form data with `file` (image file) and optional `enhance` (boolean)

**Response:**
```json
{
  "success": true,
  "detections": [
    {
      "class": "Pneumonia",
      "class_id": 2,
      "confidence": 0.8534,
      "bbox": {"x1": 150.2, "y1": 200.5, "x2": 350.8, "y2": 450.3}
    }
  ],
  "summary": {
    "total_detections": 2,
    "diseases_found": ["Pneumonia", "Infiltrate"],
    "max_confidence": 0.8534,
    "avg_confidence": 0.7821
  },
  "images": {
    "original": "data:image/jpeg;base64,...",
    "preprocessed": "data:image/jpeg;base64,...",
    "result": "data:image/jpeg;base64,..."
  }
}
```

### `GET /health`
Check server and model status.

## 🎨 Disease Classes & Colors

| Disease | Color | Class ID |
|---------|-------|----------|
| Infiltrate | Blue (#3B82F6) | 0 |
| Atelectasis | Green (#10B981) | 1 |
| Pneumonia | Red (#EF4444) | 2 |
| Cardiomegaly | Orange (#F59E0B) | 3 |
| Effusion | Purple (#8B5CF6) | 4 |
| Pneumothorax | Pink (#EC4899) | 5 |
| Mass | Indigo (#6366F1) | 6 |
| Nodule | Orange (#F97316) | 7 |

## 🔬 Image Preprocessing

The application applies CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement:

- **Clip Limit**: 2.0
- **Tile Grid Size**: 8x8
- Converts to grayscale → Applies CLAHE → Converts back to BGR

This can be toggled on/off in the UI.

## 🌐 Deployment

### Local Network Access
The app runs on `0.0.0.0` by default, making it accessible from other devices on your network:
```
http://YOUR_IP_ADDRESS:5000
```

### Production Deployment

For production deployment, consider:

1. **Use a production WSGI server** (Gunicorn/uWSGI):
   ```powershell
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

2. **Set up NGINX as reverse proxy**

3. **Use HTTPS** with SSL certificates

4. **Set `debug=False`** in `app.py`

5. **Environment variables** for configuration:
   ```python
   import os
   MODEL_PATH = os.getenv('MODEL_PATH', 'models/best.pt')
   ```

### Docker Deployment (Optional)

Create a `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "app.py"]
```

Build and run:
```powershell
docker build -t xray-detection .
docker run -p 5000:5000 xray-detection
```

## 🧪 Testing

Test the health endpoint:
```powershell
curl http://localhost:5000/health
```

Test prediction with an image:
```powershell
curl -X POST -F "file=@path\to\xray.jpg" http://localhost:5000/predict
```

## 📝 Troubleshooting

### Model Not Found
```
Warning: Model not found at models/best.pt
Using pre-trained YOLOv8m as fallback...
```
**Solution**: Export your trained model using the notebook cell or copy it manually to `webapp/models/best.pt`

### Port Already in Use
```
Address already in use
```
**Solution**: Change the port in `app.py` or kill the process using port 5000:
```powershell
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

### CUDA/GPU Issues
The app automatically uses CPU. For GPU support, install PyTorch with CUDA:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Import Errors
```
ModuleNotFoundError: No module named 'ultralytics'
```
**Solution**: Ensure all dependencies are installed:
```powershell
pip install -r requirements.txt
```

## 🔐 Security Considerations

- **File Upload Validation**: Only allows image files (png, jpg, jpeg, bmp, tiff)
- **File Size Limit**: Maximum 16MB per upload
- **Secure Filenames**: Uses `secure_filename()` to prevent path traversal
- **No Authentication**: Add authentication for production use
- **CORS**: Not enabled by default, add if needed for API access

## 📈 Performance Optimization

1. **Model Caching**: Model is loaded once and cached globally
2. **Image Compression**: Results are JPEG-compressed for web display
3. **Async Processing**: Consider using Celery for heavy workloads
4. **GPU Acceleration**: Use CUDA if available for faster inference

## 🤝 Contributing

To extend the application:

1. **Add more disease classes**: Update `DISEASE_CLASSES` dict in `app.py`
2. **Modify preprocessing**: Edit `apply_clahe()` function
3. **Change UI theme**: Modify `static/styles.css`
4. **Add analytics**: Integrate tracking in `static/script.js`

## 📄 License

This project is part of a thesis/research work. Please cite appropriately if used.

## 🙏 Acknowledgments

- **YOLOv8** by Ultralytics
- **Flask** web framework
- **OpenCV** for image processing
- **NIH Chest X-ray Dataset** (if applicable)

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure the model file exists and is valid
4. Check the console/terminal for error messages

---

**Made with ❤️ for Medical AI Research**
