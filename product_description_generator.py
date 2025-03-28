import requests
from PIL import Image
import base64
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class ProductDescriptionGenerator:
    """
    A module to generate detailed product descriptions from images using DeepSeek v3 via Together AI.
    """

    def __init__(self):
        """
        Initialize the generator with Together AI API configuration.
        """
        self.api_key = os.getenv("TOGETHERAI_API_KEY")
        if not self.api_key:
            raise ValueError("TOGETHERAI_API_KEY not found in .env file")
        self.api_url = "https://api.together.xyz/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _image_to_base64(self, image_path):
        """
        Convert image to base64 string.
        
        Args:
            image_path (str): Path to the image file.
            
        Returns:
            str: Base64 encoded image string.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def generate_description(self, image_path, max_length=150, temperature=0.7):
        """
        Generate a detailed product description from an image using DeepSeek v3.

        Args:
            image_path (str): Path to the image file.
            max_length (int): Maximum length of the generated description.
            temperature (float): Sampling temperature. Lower values make results more predictable.

        Returns:
            str: Generated product description.
        """
        # Convert image to base64
        image_base64 = self._image_to_base64(image_path)

        # Prepare the prompt with image data in the message format expected by chat/completions
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this product for an e-commerce listing based on the provided image(sneakerS). Include details about features, materials, and benefits."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]

        # Prepare the payload for Together AI API
        payload = {
            "model": "deepseek-ai/DeepSeek-V3",  # Updated to specify the exact model
            "messages": messages,
            "max_tokens": max_length,
            "temperature": temperature,
            "top_p": 0.9,
            "stop": ["\n\n"]
        }

        try:
            # Make API request
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            
            # Extract the generated description
            result = response.json()
            description = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            
            return description

        except requests.exceptions.RequestException as e:
            return f"Error generating description: {str(e)}"

# Example usage
if __name__ == "__main__":
    # Initialize the product description generator
    description_generator = ProductDescriptionGenerator()

    # Path to the image
    image_path = r"C:\Users\olami\Desktop\Thryft\AI\final_shoe_image.png"

    # Generate the product description
    description = description_generator.generate_description(image_path)

    # Print the generated description
    print("Generated Product Description:")
    print(description)