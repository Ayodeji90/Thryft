import os
import requests
import base64
from PIL import Image, ImageFilter
from io import BytesIO
from dotenv import load_dotenv

# Load API Key from .env file
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("StableDiffusion_APIKey")

# API Endpoint for Stable Diffusion
HF_API_URL = "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"

# Headers with API Key
HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

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

def composite_product_with_background(background_path, product_path, output_path, position=(50, 50), shadow=True):
    """
    Composite a segmented product image onto a generated background.
    
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

    # Resize product if needed (optional, adjust based on background)
    # product = product.resize((300, 300), Image.Resampling.LANCZOS)

    # Create a new image with the background size
    composite = background.copy()

    # Paste the product onto the background
    composite.paste(product, position, product)

    # Add a drop shadow (optional)
    if shadow:
        shadow_layer = Image.new("RGBA", composite.size, (0, 0, 0, 0))
        shadow = product.copy()
        # Create a blurred shadow
        shadow_blur = shadow.filter(ImageFilter.BLUR)
        # Reduce opacity and offset
        shadow_blur.putalpha(50)  # Adjust opacity (0-255)
        shadow_pos = (position[0] + 10, position[1] + 10)  # Offset for shadow
        shadow_layer.paste(shadow_blur, shadow_pos, shadow_blur)
        composite = Image.alpha_composite(composite, shadow_layer)

    # Convert to RGB for saving as PNG/JPEG
    composite = composite.convert("RGB")
    composite.save(output_path, "PNG")
    print(f"Composite saved as {output_path}")

    # Optional: Display the result
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
                position=(100, 200),  # Adjust x, y to position the shoe
                shadow=True
            )
        else:
            print(f"Error: Background file {background_path} not found.")