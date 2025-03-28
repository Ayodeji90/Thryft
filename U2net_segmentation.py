import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image
from model import U2NET
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="model.u2net")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

class U2NetSegmenter:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
    
    def load_model(self, model_path):
        model = U2NET(3, 1)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    
    def preprocess_image(self, image_path, target_size=(320, 320)):
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        transform = T.Compose([
            T.Resize(target_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0), original_size, image
    
    def postprocess_mask(self, mask, original_size):
        mask = mask.squeeze().cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_LINEAR)
        _, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        binary_mask = cv2.GaussianBlur(binary_mask, (5, 5), 0)
        return binary_mask
    
    def apply_mask(self, image, mask):
        image = np.array(image)
        alpha_channel = mask.astype(np.uint8)
        result = np.dstack((image, alpha_channel))
        return Image.fromarray(result, "RGBA")
    
    def segment_product(self, image_path, output_path="segmented_output.png"):
        image_tensor, original_size, image = self.preprocess_image(image_path)
        
        with torch.no_grad():
            mask = self.model(image_tensor)[0]
            mask = F.interpolate(mask, size=original_size[::-1], mode='bilinear', align_corners=False)
            mask = torch.sigmoid(mask)
        
        processed_mask = self.postprocess_mask(mask, original_size)
        final_image = self.apply_mask(image, processed_mask)
        final_image.save(output_path)
        print(f"Segmented image saved as {output_path}")

# Usage example
if __name__ == "__main__":
    segmenter = U2NetSegmenter("C:\\Users\\olami\\Downloads\\u2net.pth")
    segmenter.segment_product("input_sneaker.jpeg")
