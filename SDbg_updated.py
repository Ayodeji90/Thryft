import os
import requests
import base64
from PIL import Image, ImageEnhance, ImageFilter, ImageTransform
from io import BytesIO
from dotenv import load_dotenv
import numpy as np
import torch
from transformers import pipeline

# Load API Key from .env file
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("StableDiffusion_APIKey")

# API Endpoint for Stable Diffusion
HF_API_URL = "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"

# Headers with API Key
HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

# Load ZoeDepth model for depth estimation
depth_estimator = pipeline("depth-estimation", model="Intel/zoedepth-nyu-kitti")

def generate_background(prompt, negative_prompt=None, steps=30, cfg_scale=8.0):
    """
    Generate a background image using Stable Diffusion API and return it.
    
    Args:
        prompt (str): Description of the background (e.g., "smooth marble surface, soft light").
        negative_prompt (str, optional): Elements to exclude (e.g., "clutter, people").
        steps (int): Number of inference steps (higher = better quality, slower).
        cfg_scale (float): How closely the image follows the prompt (7-10 is typical).
    
    Returns:
        PIL.Image: Generated background image.
    """
    # Prepare payload
    payload = {
        "inputs": prompt,
        "parameters": {
            "num_inference_steps": steps,
            "guidance_scale": cfg_scale,
        }
    }
    if negative_prompt:
        payload["parameters"]["negative_prompt"] = negative_prompt

    # Send request to Hugging Face API
    response = requests.post(HF_API_URL, headers=HEADERS, json=payload)

    # Check response
    if response.status_code == 200:
        # Convert response bytes to PIL Image
        image = Image.open(BytesIO(response.content))
        return image
    else:
        raise Exception(f"API Error: {response.status_code} - {response.text}")

def generate_multiple_backgrounds(prompts, output_dir="backgrounds", negative_prompt=None):
    """
    Generate multiple backgrounds from a list of prompts and save them.
    
    Args:
        prompts (list): List of background descriptions.
        output_dir (str): Directory to save the backgrounds.
        negative_prompt (str, optional): Elements to exclude from all generations.
    """
    # Create output directory if it doesn’t exist
    os.makedirs(output_dir, exist_ok=True)

    for i, prompt in enumerate(prompts):
        output_file = os.path.join(output_dir, f"background_{i}.png")
        # Generate the background image
        image = generate_background(
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=30,
            cfg_scale=7.5
        )
        # Save the image
        image.save(output_file)
        print(f"Background saved as {output_file}")

def estimate_background_depth(background):
    """Estimate depth map of the background using ZoeDepth."""
    bg_rgb = background.convert("RGB")
    depth_map = depth_estimator(bg_rgb)["depth"]
    return depth_map

def infer_light_direction(depth_map):
    """Infer light direction based on depth map (simplified heuristic)."""
    depth_array = np.array(depth_map)
    gradient_x, gradient_y = np.gradient(depth_array)
    light_x, light_y = np.mean(gradient_x), np.mean(gradient_y)
    return light_x, light_y

def apply_perspective(image, tilt=0.1):
    """Apply a simple perspective tilt to the product image."""
    width, height = image.size
    coeffs = [1 - tilt, tilt, 0, tilt, 1 - tilt, 0, 0, 0]
    return image.transform((width, height), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC)

def adjust_lighting(product, background):
    """Adjust product lighting to match background."""
    bg_rgb = background.convert("RGB")
    bg_array = np.array(bg_rgb)
    bg_brightness = np.mean(bg_array)
    enhancer = ImageEnhance.Brightness(product.convert("RGB"))
    product_adjusted = enhancer.enhance(bg_brightness / 128)
    return product_adjusted.convert("RGBA")

def create_realistic_shadow(product, bg_size, position, light_x, light_y):
    """Create shadow with direction based on inferred light."""
    shadow = product.split()[3].convert("L").filter(ImageFilter.GaussianBlur(radius=5))
    shadow = ImageEnhance.Brightness(shadow).enhance(0.3)
    shadow_pos = (int(position[0] - light_x * 20), int(position[1] - light_y * 20))
    shadow_layer = Image.new("RGBA", bg_size, (0, 0, 0, 0))
    shadow_layer.paste(shadow, shadow_pos, shadow)
    return shadow_layer

def composite_product_with_background(background_path, product_path, output_path, position=(50, 50), shadow=True):
    """
    Composite a segmented product image onto a generated background with realistic adjustments.
    
    Args:
        background_path (str): Path to the generated background image.
        product_path (str): Path to the segmented product image (PNG with transparency).
        output_path (str): Path to save the final composite image.
        position (tuple): (x, y) coordinates to place the product (top-left corner).
        shadow (bool): Whether to add a drop shadow.
    """
    # Open images
    background = Image.open(background_path).convert("RGBA")
    product = Image.open(product_path).convert("RGBA")

    # AI-driven depth estimation with ZoeDepth
    depth_map = estimate_background_depth(background)
    light_x, light_y = infer_light_direction(depth_map)

    # Resize product dynamically based on background size
    bg_width, bg_height = background.size
    product_width, product_height = product.size
    scale_factor = min(bg_width * 0.3 / product_width, bg_height * 0.6 / product_height)
    new_size = (int(product_width * scale_factor), int(product_height * scale_factor))
    product = product.resize(new_size, Image.Resampling.LANCZOS)

    # Apply perspective based on depth and light direction
    tilt_x = light_x * 0.2  # Adjust tilt based on light direction
    product = apply_perspective(product, tilt=tilt_x)

    # Adjust lighting based on background
    product = adjust_lighting(product, background)

    # Create composite image
    composite = background.copy()
    composite.paste(product, position, product)

    # Add realistic shadow
    if shadow:
        shadow_layer = create_realistic_shadow(product, composite.size, position, light_x, light_y)
        composite = Image.alpha_composite(composite, shadow_layer)

    # Convert to RGB and save
    composite = composite.convert("RGB")
    composite.save(output_path, "PNG")
    print(f"Composite saved as {output_path}")
    composite.show()

# Example usage
if __name__ == "__main__":
    # Multiple backgrounds example
    background_prompts = [
        "A textured beige background, graffiti background with bold strip-line black and red lines, yellow splashes, blue accents, scattered red dots, vibrant colors, smooth lighting",
        "A soft pastel gradient background, light whitish and mint green tones, gentle bokeh dots, dreamy and playful atmosphere, smooth texture, bright even lighting",
        "A plain white studio background, smooth matte texture, bright even lighting, clean and minimalistic"
    ]
    try:
        # Uncomment the following line if you need to generate new backgrounds
        # generate_multiple_backgrounds(
        #     prompts=background_prompts,
        #     output_dir="ecommerce_backgrounds",
        #     negative_prompt="people, objects, text, overly busy"
        # )
        pass
    except Exception as e:
        print(f"Error generating multiple backgrounds: {e}")

    # Use the existing backgrounds from ecommerce_backgrounds directory
    output_dir = "ecommerce_backgrounds"
    product_path = "segmented_output.png"  # Segmented product with transparency
    if not os.path.exists(product_path):
        print(f"Error: Product file {product_path} not found.")
        exit()

    # Iterate over the generated backgrounds
    for i in range(len(background_prompts)):
        background_path = os.path.join(output_dir, f"background_{i}.png")
        if os.path.exists(background_path):
            output_path = os.path.join(output_dir, f"final_composite_{i}.png")
            composite_product_with_background(
                background_path=background_path,
                product_path=product_path,
                output_path=output_path,
                position=(100, 200),  # Adjust x, y to position the product
                shadow=True
            )
        else:
            print(f"Error: Background file {background_path} not found.")