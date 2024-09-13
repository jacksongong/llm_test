import torch
from diffusers import FluxPipeline
from huggingface_hub import login
login("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")

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
path="./saved_models"




torch.save(pipe, path)

pipeline_loaded = torch.load(path).to("cuda")
print("Model loaded from saved file")

prompt = "A cat holding a sign that says hello world"
image = pipeline_loaded(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]

image_path="./images/flux-dev.png"
image.save(image_path)