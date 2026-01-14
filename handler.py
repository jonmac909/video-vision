"""
RunPod Serverless Handler for LLaVA-NeXT v1.6 Frame Descriptions
Generates production-focused visual descriptions for video frames
Updated: 2026-01-14 - Added base64 support to eliminate network I/O
"""

import runpod
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO
import base64
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


def decode_base64_image(base64_str: str) -> Image.Image:
    """Decode base64 image data to PIL Image (eliminates network I/O)"""
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data))
        return image.convert('RGB')
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}")
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


def describe_frames_batch(images: list[Image.Image]) -> list[str]:
    """Generate production descriptions for multiple frames in a single batch"""
    try:
        # Prepare batch inputs
        conversations = []
        for _ in images:
            conversations.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": PRODUCTION_PROMPT},
                    ],
                },
            ])

        # Process all images in batch
        prompts = [processor.apply_chat_template(conv, add_generation_prompt=True) for conv in conversations]
        inputs = processor(images=images, text=prompts, padding=True, return_tensors="pt").to(model.device)

        # Generate descriptions for all frames at once
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

        # Decode all outputs
        descriptions = []
        for output in outputs:
            description = processor.decode(output, skip_special_tokens=True)

            # Extract assistant response
            for pattern in ["[/INST]", "ASSISTANT:", "Assistant:"]:
                if pattern in description:
                    description = description.split(pattern)[-1].strip()
                    break

            # Remove prompt if leaked
            if PRODUCTION_PROMPT in description:
                description = description.replace(PRODUCTION_PROMPT, "").strip()

            descriptions.append(description)

        return descriptions

    except Exception as e:
        logger.error(f"Batch description failed: {e}")
        return [f"Error: {str(e)}" for _ in images]


def handler(event):
    """
    RunPod handler function

    Input format (URL mode):
    {
        "input": {
            "frame_urls": ["https://...", "https://..."]
        }
    }

    Input format (base64 mode - faster, no network I/O):
    {
        "input": {
            "frame_data": ["base64_string1", "base64_string2", ...],
            "format": "base64"
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
        frame_data = job_input.get("frame_data", [])
        format_type = job_input.get("format", "url")

        # Validate input
        if format_type == "base64":
            if not frame_data:
                return {"error": "No frame_data provided for base64 mode"}
            logger.info(f"Processing {len(frame_data)} frames in base64 mode (no network I/O)")
        else:
            if not frame_urls:
                return {"error": "No frame_urls provided"}
            logger.info(f"Processing {len(frame_urls)} frames in URL mode")

        # Load images based on format
        images = []
        failed_indices = []

        if format_type == "base64":
            # Decode base64 data directly (NO network I/O)
            for idx, data in enumerate(frame_data):
                try:
                    image = decode_base64_image(data)
                    images.append(image)
                except Exception as e:
                    logger.error(f"Frame {idx + 1}/{len(frame_data)}: Base64 decode failed - {e}")
                    images.append(None)
                    failed_indices.append(idx)
        else:
            # Download from URLs (existing behavior)
            for idx, url in enumerate(frame_urls):
                try:
                    image = download_image(url)
                    images.append(image)
                except Exception as e:
                    logger.error(f"Frame {idx + 1}/{len(frame_urls)}: Download failed - {e}")
                    images.append(None)
                    failed_indices.append(idx)

        # Process all valid images in a single batch
        valid_images = [img for img in images if img is not None]
        if valid_images:
            logger.info(f"Running batch inference on {len(valid_images)} frames")
            batch_descriptions = describe_frames_batch(valid_images)
        else:
            batch_descriptions = []

        # Merge results with failed frames
        descriptions = []
        valid_idx = 0
        for idx, img in enumerate(images):
            if img is None:
                descriptions.append("Failed to download frame")
            else:
                descriptions.append(batch_descriptions[valid_idx])
                valid_idx += 1

        logger.info(f"Batch complete: {len(descriptions)} descriptions, {len(failed_indices)} failed")

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
