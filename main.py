from diffusers import DiffusionPipeline
import torch
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")


prompt = "Kitten with sword fighting big dragon, realistic,  detailed, 4k"
image = pipe(prompt).images[0]
print(image)
output_path = "output/cats5.png"
image.save(output_path)
print(f"Image saved at {output_path}")
