FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    YOLO_CONFIG_DIR=/tmp/ultralytics

WORKDIR /app

# System libraries needed by OpenCV headless & pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt

# Install CPU-only PyTorch wheels first to minimize image size (~180 MB vs ~2.5 GB CUDA)
RUN pip install --upgrade pip && \
    pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY . .

# Create runtime directories for uploads, SQLite fallback, and feedback archive
RUN mkdir -p /app/uploads /app/models /app/data/feedback /app/.cache/huggingface

# Pre-cache CLIP backbone & COCO YOLOv8s during image build to avoid cold-start timeout
RUN python -c "from transformers import CLIPModel, CLIPProcessor; CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')" || true
RUN python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')" || true

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8000}/health" || exit 1

# Single worker recommended when using local SQLite or in-memory ML model registry
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
