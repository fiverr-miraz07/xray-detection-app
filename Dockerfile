# Use an official lightweight Python image
FROM python:3.9-slim

# Install system dependencies required for OpenCV and generalized file handling
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install Python dependencies
# Adding gunicorn ensuring it is installed even if verified missing in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install gunicorn

# Copy the content of the local src directory to the working directory
COPY . .

# Create necessary directories for the app if they are missing
RUN mkdir -p uploads results models

# Expose port 10000 (Render's default)
EXPOSE 10000

# Command to run the application using Gunicorn
# app:app refers to looking for the 'app' object in 'app.py'
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app", "--timeout", "120"]
