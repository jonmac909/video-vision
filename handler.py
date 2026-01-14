"""
RunPod Serverless Handler for LLaVA-NeXT-Video Frame Descriptions
Generates production-focused visual descriptions for video frames
"""

import runpod
import torch
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load LLaVA-NeXT-Video model on cold start
logger.info("Loading LLaVA-NeXT-Video model...")
MODEL_ID = "llava-hf/LLaVA-NeXT-Video-7B-hf"

processor = LlavaNextVideoProcessor.from_pretrained(MODEL_ID)
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
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
        # Prepare conversation format (LLaVA expects chat format)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PRODUCTION_PROMPT},
                    {"type": "image"},
                ],
            },
        ]

        # Apply chat template
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        # Process inputs
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

        # Generate description
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,  # Deterministic for consistency
        )

        # Decode output
        description = processor.decode(output[0], skip_special_tokens=True)

        # Extract only the assistant's response (after the prompt)
        if "ASSISTANT:" in description:
            description = description.split("ASSISTANT:")[-1].strip()

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
