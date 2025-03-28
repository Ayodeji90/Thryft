import os
import requests

# Define model URLs
model_urls = {
    "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    #"RealESRGAN_x4plus_anime": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus_anime_6B.pth"
}

# Define save directory
save_dir = "models"
os.makedirs(save_dir, exist_ok=True)

# Download models
for name, url in model_urls.items():
    response = requests.get(url, stream=True)
    file_path = os.path.join(save_dir, f"{name}.pth")

    with open(file_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)

    print(f"Downloaded {name} to {file_path}")
