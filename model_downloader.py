import torch
from model.u2net import U2NET  # Adjust path to your U2NET code

model = U2NET(3, 1)
model_path = r"C:\Users\olami\Downloads\u2net.pth"  # Your 168 MB file
try:
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    print("Model loaded successfully!")
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 320, 320)
    output = model(dummy_input)
    print("Output shape:", output.shape)
except Exception as e:
    print(f"Loading failed: {e}")