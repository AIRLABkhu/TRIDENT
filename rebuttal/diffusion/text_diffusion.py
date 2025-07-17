import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import os
import argparse

def main(args):
    # 모델 로드
    

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
    ).to("cuda")

    # 입력 이미지 경로 설정
    img_dir_path = os.path.join(args.data_root, args.source_domain, args.class_name)
    ref_dir_path = os.path.join(args.ref_root, f"{args.source_domain}2{args.target_domain}", args.class_name)
    ref_paths = sorted(os.listdir(ref_dir_path))
    img_paths = sorted(os.listdir(img_dir_path))

    # 출력 디렉토리 생성
    output_dir = os.path.join(args.save_path, f"{args.source_domain}2{args.target_domain}", args.class_name)
    os.makedirs(output_dir, exist_ok=True)

    for i, ref_name in enumerate(ref_paths):
        img_name = ref_name.split("_0_")[-1]
        input_path = os.path.join(img_dir_path, img_name)
        input_image = Image.open(input_path).convert("RGB").resize((512, 512))

        # 텍스트 프롬프트 구성
        prompt = f"{args.target_domain} of {args.class_name}"

        # 이미지 생성
        result = pipe(
            prompt=prompt,
            image=input_image,
            strength=args.strength,
            guidance_scale=args.guidance_scale
        ).images[0]

        # 저장
        # result.save(os.path.join(output_dir, f"{args.source_domain}2{args.target_domain}", args.class_name, img_name))
        save_path = os.path.join(output_dir, f"{args.source_domain}2{args.target_domain}", args.class_name, img_name)
        # 디렉토리 없으면 생성
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # 저장
        result.save(save_path)
        print(f"[{i+1}/{len(img_paths)}] Saved to: {os.path.join(output_dir, img_name)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stable Diffusion img2img domain transfer")
    parser.add_argument("--source_domain", type=str, required=True, help="Source domain (e.g., cartoon)")
    parser.add_argument("--target_domain", type=str, required=True, help="Target domain (e.g., photo)")
    parser.add_argument("--class_name", type=str, required=True, help="Class name (e.g., person, dog)")
    parser.add_argument("--save_path", type=str, default="Diffusion", help="Directory to save outputs")
    parser.add_argument("--data_root", type=str, default="/data2/local_datasets/PACS/", help="Root directory of input data")
    parser.add_argument("--ref_root", type=str, default="/data2/local_datasets/TRI_20250302_PACS_CLEANED/ACP", help="Root directory of input data")
    parser.add_argument("--strength", type=float, default=0.75, help="Strength for img2img (0.0 ~ 1.0)")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale (higher = more prompt-following)")

    args = parser.parse_args()
    main(args)
