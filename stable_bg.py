import os
import requests
from io import BytesIO
from PIL import Image, ImageFilter
from dotenv import load_dotenv
import torch
from transformers import CLIPProcessor, CLIPModel
import pytesseract

class EcommerceImageProcessor:
    def __init__(self, api_key=None):
        """Initialize the EcommerceImageProcessor with API key and models."""
        load_dotenv()
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("API Key not found in .env file. Please set 'TOGETHER_API_KEY'.")
        
        # Together AI Stable Diffusion endpoint
        self.together_api_url = "https://api.together.xyz/inference"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Initialize CLIP model and processor
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Set device (GPU if available)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = self.clip_model.to(self.device)

    def generate_background(self, prompt, negative_prompt=None, steps=30, cfg_scale=8.0):
        """Generate a background image using Together AI's Stable Diffusion API."""
        payload = {
            "model": "stabilityai/stable-diffusion-2-1",  # Example model, check Together AI docs for available models
            "prompt": prompt,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "width": 512,  # Adjust as needed
            "height": 512,  # Adjust as needed
            "output_type": "pil"  # Returns image directly
        }
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

        response = requests.post(self.together_api_url, headers=self.headers, json=payload)
        if response.status_code == 200:
            # Together AI typically returns JSON with base64 image data; adjust based on actual response
            data = response.json()
            if "output" in data and "choices" in data["output"]:
                image_data = data["output"]["choices"][0]["image_base64"]
                image = Image.open(BytesIO(base64.b64decode(image_data)))
                return image
            else:
                raise Exception("Unexpected response format from Together AI API")
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")

    def generate_multiple_backgrounds(self, prompts, output_dir="backgrounds", negative_prompt=None):
        """Generate multiple backgrounds from a list of prompts and save them."""
        os.makedirs(output_dir, exist_ok=True)
        for i, prompt in enumerate(prompts):
            output_file = os.path.join(output_dir, f"background_{i}.png")
            image = self.generate_background(
                prompt=prompt,
                negative_prompt=negative_prompt,
                steps=30,
                cfg_scale=7.5
            )
            image.save(output_file)
            print(f"Background saved as {output_file}")

    def composite_product_with_background(self, background_path, product_path, output_path, position=(50, 50), shadow=True):
        """Composite a segmented product image onto a generated background."""
        background = Image.open(background_path).convert("RGBA")
        product = Image.open(product_path).convert("RGBA")

        composite = background.copy()
        composite.paste(product, position, product)

        if shadow:
            shadow_layer = Image.new("RGBA", composite.size, (0, 0, 0, 0))
            shadow = product.copy()
            shadow_blur = shadow.filter(ImageFilter.BLUR)
            shadow_blur.putalpha(50)
            shadow_pos = (position[0] + 10, position[1] + 10)
            shadow_layer.paste(shadow_blur, shadow_pos, shadow_blur)
            composite = Image.alpha_composite(composite, shadow_layer)

        composite = composite.convert("RGB")
        composite.save(output_path, "PNG")
        print(f"Composite saved as {output_path}")
        composite.show()

    def extract_product_features(self, product_path):
        """Extract product features using CLIP for type detection and OCR+CLIP for attributes."""
        # Load and preprocess the product image
        product_image = Image.open(product_path).convert("RGB")
        inputs = self.clip_processor(images=product_image, return_tensors="pt").to(self.device)

        # CLIP for image type detection
        labels = ["shoe", "bag", "shirt", "gown", "sneaker"]
        texts = self.clip_processor(text=labels, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            text_features = self.clip_model.get_text_features(**texts)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).softmax(dim=-1)
        
        # Get the most likely product type
        product_type_idx = similarity.argmax().item()
        product_type = labels[product_type_idx]

        # OCR for text extraction (e.g., size, color, material)
        ocr_text = pytesseract.image_to_string(product_image).lower().strip()
        attributes = {"type": product_type}

        # Infer attributes using OCR and CLIP context
        if "size" in ocr_text or any(str(i) in ocr_text for i in range(1, 20)):
            attributes["size"] = next((word for word in ocr_text.split() if any(str(i) in word for i in range(1, 20))), "unknown")
        if "color" in ocr_text or any(color in ocr_text for color in ["black", "white", "red", "blue", "green", "yellow"]):
            attributes["color"] = next((color for color in ["black", "white", "red", "blue", "green", "yellow"] if color in ocr_text), "unknown")
        if "material" in ocr_text or any(mat in ocr_text for mat in ["leather", "cotton", "polyester", "canvas"]):
            attributes["material"] = next((mat for mat in ["leather", "cotton", "polyester", "canvas"] if mat in ocr_text), "unknown")

        # Fallback to CLIP for visual inference if OCR fails
        if attributes["color"] == "unknown":
            color_texts = ["black", "white", "red", "blue", "green", "yellow"]
            color_texts = self.clip_processor(text=color_texts, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                color_features = self.clip_model.get_text_features(**color_texts)
                color_similarity = (image_features @ color_features.T).softmax(dim=-1)
                color_idx = color_similarity.argmax().item()
                attributes["color"] = color_texts[color_idx]
        if attributes["material"] == "unknown":
            material_texts = ["leather", "cotton", "polyester", "canvas"]
            material_texts = self.clip_processor(text=material_texts, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                material_features = self.clip_model.get_text_features(**material_texts)
                material_similarity = (image_features @ material_features.T).softmax(dim=-1)
                material_idx = material_similarity.argmax().item()
                attributes["material"] = material_texts[material_idx]

        return attributes

# Example usage
if __name__ == "__main__":
    # Initialize the processor
    processor = EcommerceImageProcessor()

    # Generate multiple backgrounds
    background_prompts = [
        "A textured beige background, graffiti background with bold strip-line black and red lines, yellow splashes, blue accents, scattered red dots, vibrant colors, smooth lighting",
        "A soft pastel gradient background, light whitish and mint green tones, gentle bokeh dots, dreamy and playful atmosphere, smooth texture, bright even lighting",
        "A plain white studio background, smooth matte texture, bright even lighting, clean and minimalistic"
    ]
    try:
        processor.generate_multiple_backgrounds(
            prompts=background_prompts,
            output_dir="ecommerce_backgrounds",
            negative_prompt="people, objects, text, overly busy"
        )
    except Exception as e:
        print(f"Error generating multiple backgrounds: {e}")

    # Extract product features
    product_path = "segmented_output.png"
    if not os.path.exists(product_path):
        print(f"Error: Product file {product_path} not found.")
        exit()
    features = processor.extract_product_features(product_path)
    print("Extracted Product Features:", features)

    # Composite product onto each background
    output_dir = "ecommerce_backgrounds"
    for i in range(len(background_prompts)):
        background_path = os.path.join(output_dir, f"background_{i}.png")
        if os.path.exists(background_path):
            output_path = os.path.join(output_dir, f"final_composite_{i}.png")
            processor.composite_product_with_background(
                background_path=background_path,
                product_path=product_path,
                output_path=output_path,
                position=(100, 200),
                shadow=True
            )
        else:
            print(f"Error: Background file {background_path} not found.")