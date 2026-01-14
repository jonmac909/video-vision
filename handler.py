"""
RunPod Serverless Handler for LLaVA-NeXT v1.6 Frame Descriptions
Generates production-focused visual descriptions for video frames
Updated: 2026-01-13 22:05 - transformers 4.45.0
"""

import runpod
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load LLaVA-NeXT model on cold start
logger.info("Loading LLaVA-NeXT model...")
# Using LLaVA-NeXT v1.6 (Mistral-7B base) - proven model with strong vision capabilities
MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"

processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
model = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)
logger.info("Model loaded successfully")

# Production-focused prompt for cinematography analysis
PRODUCTION_PROMPT = """Analyze this video frame for production recreation. Focus on VISUAL PRODUCTION DETAILS that can be replicated:

**Must Include:**
- Camera angle/framing (wide, close-up, overhead, etc.)
- Visual effects (smoke, fire, particles, glow, blur)
- Color grading (warm/cool tones, contrast, saturation)
- Composition (rule of thirds, symmetry, depth)
- Text overlays (font style, size, position, animation)
- Background elements (solid color, gradient, image, video)
- Lighting (dramatic, soft, high-key, low-key)

Provide 2-3 sentences covering camera, effects, colors, and composition."""


def download_image(url: str) -> Image.Image:
    """Download image from URL and return PIL Image"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image.convert('RGB')
    except Exception as e:
        logger.error(f"Failed to download image from {url}: {e}")
        raise


def describe_frame(image: Image.Image) -> str:
    """Generate production description for a single frame"""
    try:
        # LLaVA-NeXT v1.6 uses chat format with specific structure
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": PRODUCTION_PROMPT},
                ],
            },
        ]

        # Apply chat template and process
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

        # Generate description with reasonable limits
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,  # Deterministic
            temperature=None,  # Not used when do_sample=False
        )

        # Decode and extract response
        description = processor.decode(output[0], skip_special_tokens=True)

        # Extract assistant response (LLaVA-NeXT format)
        # Try multiple split patterns
        for pattern in ["[/INST]", "ASSISTANT:", "Assistant:"]:
            if pattern in description:
                description = description.split(pattern)[-1].strip()
                break

        # Remove the prompt if it leaked through
        if PRODUCTION_PROMPT in description:
            description = description.replace(PRODUCTION_PROMPT, "").strip()

        return description

    except Exception as e:
        logger.error(f"Failed to generate description: {e}")
        return f"Error generating description: {str(e)}"


def handler(event):
    """
    RunPod handler function

    Input format:
    {
        "input": {
            "frame_urls": ["https://...", "https://..."]
        }
    }

    Output format:
    {
        "descriptions": ["description1", "description2", ...],
        "failed_indices": [0, 3, ...]  // Indices that failed
    }
    """
    try:
        job_input = event["input"]
        frame_urls = job_input.get("frame_urls", [])

        if not frame_urls:
            return {"error": "No frame_urls provided"}

        logger.info(f"Processing {len(frame_urls)} frames")

        descriptions = []
        failed_indices = []

        for idx, url in enumerate(frame_urls):
            try:
                # Download image
                image = download_image(url)

                # Generate description
                description = describe_frame(image)
                descriptions.append(description)

                logger.info(f"Frame {idx + 1}/{len(frame_urls)}: Success")

            except Exception as e:
                logger.error(f"Frame {idx + 1}/{len(frame_urls)}: Failed - {e}")
                descriptions.append(f"Failed to process frame")
                failed_indices.append(idx)

        return {
            "descriptions": descriptions,
            "failed_indices": failed_indices,
            "count": len(descriptions),
        }

    except Exception as e:
        logger.error(f"Handler error: {e}")
        return {"error": str(e)}


# Start RunPod serverless handler
if __name__ == "__main__":
    logger.info("Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})
