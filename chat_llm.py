import torch
from diffusers import FluxPipeline
from huggingface_hub import login
import os

# Login to Hugging Face using your token
login("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")

# Set the path to save and load the model
path = "./saved_models/flux_model.pt"

# Step 1: Download the model (and use GPU if available)
if not os.path.exists(path):
    # Download the model and load it directly to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
    ).to(device)
    
    # Optionally enable CPU offloading to save VRAM
    pipe.enable_model_cpu_offload()  # Only if necessary for VRAM

    # Save the model to a file for future use
    torch.save(pipe, path)
    print(f"Model downloaded and saved to {path}")
else:
    # Step 2: Load the model from the saved file and move it to GPU
    pipe = torch.load(path).to("cuda" if torch.cuda.is_available() else "cpu")
    print("Model loaded from saved file")

# Step 3: Generate an image using the loaded model
prompt = "A cat holding a sign that says hello world"

image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)  # Random seed
).images[0]

# Save the generated image
image_path = "./images/flux_dev_image.png"
os.makedirs(os.path.dirname(image_path), exist_ok=True)
image.save(image_path)

print(f"Image generated and saved to {image_path}")
