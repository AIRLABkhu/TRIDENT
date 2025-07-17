import json
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

# JSON 파일 경로
json_path = "best_captions.json"  # 여기에 JSON 파일 경로 입력
output_dir = "/data3/local_datasets/Diffusion_text_only"
os.makedirs(output_dir, exist_ok=True)

# Stable Diffusion 로드
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    torch_dtype=torch.float16
).to("cuda")

# JSON 파일 불러오기
with open(json_path, "r") as f:
    data = json.load(f)

# 각 항목 처리
for item in data:
    original_caption = item["best_caption"]
    prompt = original_caption.replace("an art painting style", "a cartoon style")
    
    # 이미지 생성
    image = pipe(prompt).images[0]
    
    # 이미지 저장
    filename = os.path.basename(item["image_file"])
    save_path = os.path.join(output_dir, filename)
    image.save(save_path)

    print(f"Saved: {save_path}")
