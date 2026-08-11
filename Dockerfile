FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by OpenCV and EasyOCR
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install core dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install API and Reconciliation dependencies
RUN pip install --no-cache-dir fastapi uvicorn python-multipart easyocr

# Copy the entire project
COPY . .

# Expose port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
