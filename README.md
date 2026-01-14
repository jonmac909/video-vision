# LLaVA-NeXT RunPod Worker

RunPod serverless worker for generating production-focused visual descriptions of video frames using [LLaVA-NeXT v1.6](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf).

## Overview

This worker analyzes video frames and generates detailed descriptions of visual production elements:
- Camera angles and framing
- Visual effects (smoke, fire, particles, etc.)
- Color grading and lighting
- Composition and layout
- Text overlays
- Background elements

**Use Case**: VideoRAG system for Auto Poster video recreation

## Model

- **Model**: `llava-hf/llava-v1.6-mistral-7b-hf` (LLaVA-NeXT v1.6)
- **Base**: Mistral-7B with vision encoder
- **Architecture**: Multimodal vision-language model
- **Size**: 7B parameters
- **Hardware**: RTX 4090 (24GB VRAM) or RTX 3090 (24GB VRAM)
- **Inference Speed**: ~1-2 seconds per frame

## API Interface

### Input Format

```json
{
  "input": {
    "frame_urls": [
      "https://example.com/frame1.jpg",
      "https://example.com/frame2.jpg"
    ]
  }
}
```

### Output Format

```json
{
  "descriptions": [
    "Wide shot with dark gradient background. White sans-serif text centered...",
    "Close-up shot with dramatic lighting. Warm color grading with high contrast..."
  ],
  "failed_indices": [],
  "count": 2
}
```

## Deployment to RunPod

### Prerequisites

1. RunPod account with API access
2. GitHub repository linked to RunPod

### Step 1: Push to GitHub

```bash
cd /Users/jonmac/Documents/video-vision-runpod
git init
git add .
git commit -m "Initial commit: LLaVA-NeXT-Video RunPod worker"
git remote add origin https://github.com/jonmac909/video-vision-runpod.git
git push -u origin main
```

### Step 2: Create RunPod Endpoint

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Click "New Endpoint"
3. Configure:
   - **Name**: `llava-next-video-vision`
   - **GPU**: RTX 4090 (8GB VRAM)
   - **Workers**: Start with 4, scale up to 10 as needed
   - **Source**: GitHub repository
   - **Repository**: `jonmac909/video-vision-runpod`
   - **Branch**: `main`
   - **Docker Build Method**: Use Dockerfile
4. Click "Deploy"

### Step 3: Configure Environment Variables

In Railway API (`render-api`), add:

```bash
RUNPOD_VISION_ENDPOINT_ID=<endpoint-id-from-runpod>
```

## Local Testing (Optional)

### Test handler locally with mock input:

```bash
python3 -c "
from handler import handler

# Mock event
event = {
    'input': {
        'frame_urls': [
            'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg'
        ]
    }
}

result = handler(event)
print(result)
"
```

## Cost Analysis

**RunPod Pricing** (RTX 4090):
- $0.00049/second
- ~1.5 seconds per frame
- 10 frames batch = ~15 seconds = $0.0074
- **100 frames** = $0.074 (vs Claude Vision $0.65)

**Comparison**:
- **Claude Vision**: $0.65 per 100 frames
- **LLaVA-NeXT-Video**: $0.074 per 100 frames
- **Savings**: 88.6% reduction per batch

**Full Video** (2 hours @ 1 FPS = 7,200 frames):
- **Claude Vision**: $47
- **LLaVA-NeXT-Video**: $0.53 (10 workers parallel)
- **Savings**: 98.9%

**Scene Keyframes Only** (~400 frames):
- **Claude Vision**: $2.60
- **LLaVA-NeXT-Video**: $0.12
- **Savings**: 95.4%

## Updating the Worker

After making changes, push to GitHub to trigger rebuild:

```bash
cd /Users/jonmac/Documents/video-vision-runpod
git add .
git commit -m "Update: description"
git push origin main
```

RunPod will automatically rebuild the Docker image (takes 2-5 minutes).

## Troubleshooting

### Worker Crashes on Startup

- Check GPU memory: LLaVA-NeXT-Video requires 8GB minimum
- Verify model download completed in Docker build logs
- Check RunPod worker logs for Python errors

### Slow Inference

- Ensure `torch_dtype=torch.float16` is set (reduces memory, increases speed)
- Consider using `bitsandbytes` for 8-bit quantization (further speed improvement)
- Increase worker allocation for parallel processing

### Out of Memory

- Reduce batch size (process fewer frames per call)
- Use LLaVA-NeXT-Video-7B instead of larger variants
- Enable 8-bit quantization in handler.py

## Model Alternatives

If quality is insufficient, try these alternatives:

1. **Molmo 2** (better cinematography understanding)
   - Model ID: `allenai/molmo`
   - Requires more VRAM (12GB+)

2. **Qwen3-VL** (longer context for video sequences)
   - Model ID: `QwenLM/Qwen3-VL`
   - Overkill for single-frame analysis

## License

This worker uses LLaVA-NeXT-Video under Apache 2.0 license.
