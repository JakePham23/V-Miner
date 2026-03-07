import os
import mlx_vlm
from PIL import Image

model_id = "mlx-community/LightOnOCR-2-1B-bf16"
print("Loading model...")
model, processor = mlx_vlm.load(model_id)
print("Model loaded.")

# Create a small dummy image
image = Image.new("RGB", (100, 100), color="white")

prompt = "Extract all text from this image accurately."
print("Generating...")
res = mlx_vlm.generate(model, processor, prompt, image=image)
print("Generation result type:", type(res))
if hasattr(res, "text"):
    print("res.text:", res.text)
else:
    print("res:", res)
