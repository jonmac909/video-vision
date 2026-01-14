# RunPod Serverless Worker - LLaVA-NeXT-Video Frame Descriptions
# Base image with CUDA support
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

# Download model weights on build (caches in image)
# This prevents cold start delays
RUN python -c "from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration; \
    MODEL_ID='llava-hf/llava-v1.6-mistral-7b-hf'; \
    print('Downloading model weights...'); \
    LlavaNextProcessor.from_pretrained(MODEL_ID); \
    LlavaNextForConditionalGeneration.from_pretrained(MODEL_ID, low_cpu_mem_usage=True); \
    print('Model weights cached successfully')"

# Set RunPod handler
CMD ["python", "-u", "handler.py"]
