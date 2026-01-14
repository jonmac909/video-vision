# RunPod Serverless Worker - LLaVA-NeXT v1.6 Frame Descriptions
# Base image with CUDA support
# Build trigger: 2026-01-13 22:05
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler script
COPY handler.py .

# Model weights will download on first worker startup (~14GB, ~2-3 minutes)
# RunPod caches downloaded models across worker restarts

# Set RunPod handler
CMD ["python", "-u", "handler.py"]
