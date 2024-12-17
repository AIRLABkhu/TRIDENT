import argparse
import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import time
from typing import Tuple
import random

from diffusers import StableUnCLIPImg2ImgPipeline, ControlNetModel
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)
from src.models.pipeline_stable_unclip_controlnet_img2img import (
    StableUnCLIPControlNetImg2ImgPipeline,
)

from src.models.trident import TRIDENTModule

STABLE_DIFFUSION_MODEL = "stabilityai/stable-diffusion-2-1-unclip"
OPENAI_CLIP_MODEL = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"


class DomainGenerator:
    def __init__(
        self,
        save_dir,
        src_dir,
        device="cuda",
        seed=42,
    ) -> None:
        self.out_size = 768
        self.clip_size = 224
        if save_dir is not None:
            self.data_save_dir = os.path.join(save_dir, "data")
            self.grid_save_dir = os.path.join(save_dir, "grid")
        self.src_dir = src_dir
        self.device = device
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.processor = CLIPImageProcessor.from_pretrained(
            STABLE_DIFFUSION_MODEL,
            subfolder="feature_extractor",
            torch_dtype=torch.float16,
        )
        self.src_show_transform = transforms.Compose(
            [
                transforms.Resize(self.out_size),
                transforms.CenterCrop((self.out_size, self.out_size)),
            ]
        )
        self.control_img_transform = transforms.Compose(
            [
                transforms.Resize(self.out_size),
                transforms.CenterCrop((self.out_size, self.out_size)),
            ]
        )
        self.init_visual_clip_models()
        
        self.pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
                STABLE_DIFFUSION_MODEL, torch_dtype=torch.float16
            ).to(self.device)

        self.pipe = self.pipe.to(self.device)


        self.pipe.enable_xformers_memory_efficient_attention()
        torch.cuda.empty_cache()

    def get_src_images(self, n):
        return self.get_genric_images(n)

    def get_genric_images(self, n):
        image_file_list = []
        while True:
            #if len(image_file_list) < n:
            image_file_list = os.listdir(self.src_dir)
            image_file_list.sort()
            #np.random.shuffle(image_file_list)
            for _ in range(len(os.listdir(self.src_dir))):
                image_file = image_file_list.pop()
                with Image.open(os.path.join(self.src_dir, image_file)) as img:
                    selection = np.array(img.convert("RGB"))
                proc_images = (
                    self.processor(images=selection, return_tensors="pt")
                    .pixel_values.to(self.device)
                    .to(torch.float16)
                )
                yield image_file, proc_images, selection

    def init_visual_clip_models(self):
        self.visual_encoder = CLIPVisionModelWithProjection.from_pretrained(
            STABLE_DIFFUSION_MODEL, subfolder="image_encoder", torch_dtype=torch.float16
        ).to(self.device)

    def generate(
        self,
        neg_prompt=None,
        num_inference_steps=20,
        guidance_scale=10.0,
        noise_level=0,
        n_per_prompt=4,
        n_batch=4,
        save_grid=False,
        split_model: torch.nn.Module=None
    ):
        os.makedirs(self.data_save_dir, exist_ok=True)
        os.makedirs(self.grid_save_dir, exist_ok=True)
        if neg_prompt:
            print(f"Negative Prompt: {neg_prompt}")
        if save_grid:
            print(f"Saving grid images to {self.grid_save_dir}")
        grid_images = []
        data_store = self.get_src_images(n_per_prompt * n_batch)

        for batch in range(len(os.listdir(self.src_dir))):
            src_filename, src_data, src_img_raw = next(data_store)
            label = None
            if isinstance(src_data, dict):
                src_img = src_data["img"]
            else:
                src_img = src_data
 
            src_emb = self.visual_encoder(src_img).image_embeds
            src_emb = src_emb.type(torch.float32)

            src_dom_emb, src_cls_emb, src_dom_other_emb, src_cls_other_emb = split_model(src_emb)
            src_other_emb = src_dom_other_emb + src_cls_other_emb
            src_other_emb = (2*random.random()-1) * src_other_emb

            src_emb = src_dom_emb + src_cls_emb + src_other_emb
            
            guide_emb = src_emb.type(torch.float16)
            if save_grid:
                grid_offset = 1
                grid_images.append(
                    self.src_show_transform(Image.fromarray(src_img_raw))
                )

            imgs = self.pipe(
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=neg_prompt,
                image_embeds=guide_emb,
                num_images_per_prompt=n_per_prompt,
                cross_attention_kwargs=(
                    None
                ),
                noise_level=noise_level,
            ).images  # type: ignore

            for i, img in enumerate(imgs):
                img.save(
                    os.path.join(self.data_save_dir, f"{batch}_{i}_{src_filename}")
                )
                if save_grid:
                    grid_images.append(img)
                    if len(grid_images) == min(
                        (n_per_prompt + grid_offset) * n_batch,
                        (n_per_prompt + grid_offset) * n_per_prompt,
                    ):
                        width, height = grid_images[0].size
                        grid = Image.new(
                            "RGB",
                            (
                                width * (n_per_prompt + grid_offset),
                                height * min(n_per_prompt, n_batch),
                            ),
                        )
                        for i, img in enumerate(grid_images):
                            grid.paste(
                                img,
                                (
                                    width * (i % (n_per_prompt + grid_offset)),
                                    height * (i // (n_per_prompt + grid_offset)),
                                ),
                            )
                        grid.save(
                            os.path.join(
                                self.grid_save_dir,
                                f"{batch//n_per_prompt}_{src_filename}",
                            )
                        )
                        grid_images = []



def get_args():
    parser = argparse.ArgumentParser(
        description="Generation with domain embedding guidance"
    )
    parser.add_argument(
        "--src_dir", type=str, help="Path to the source domain directory"
    )
    parser.add_argument(
        "--tgt_dir", type=str, help="Path to the target domain directory"
    )
    parser.add_argument(
        "--pre_trained_dir", type=str, help="Path to the target domain directory"
    )
    parser.add_argument(
        "--src_limit", type=int, default=-1, help="Number of source images to use"
    )
    parser.add_argument(
        "--tgt_limit", type=int, default=-1, help="Number of target images to use"
    )

    parser.add_argument(
        "--gen_src_dir", type=str, help="Path to the generation source directory"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="output/",
        help="Path to save the generation results",
    )

    parser.add_argument(
        "--n_batch", type=int, default=3, help="Number of batches to generate"
    )
    parser.add_argument(
        "--n_per_prompt", type=int, default=4, help="Number of images per prompt"
    )
    parser.add_argument(
        "--neg_prompt",
        type=str,
        default=None,
        help="Prompt to further aid the generation",
    )
    parser.add_argument(
        "--noise_level",
        type=int,
        default=0,
        help="Noise level to add to the generation pipeline",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Number of inference steps to add to the generation pipeline",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=10,
        help="guidance_scale to add to the generation pipeline",
    )

    parser.add_argument(
        "--save_grid", action="store_true", help="Whether to save the grid images"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run the model on"
    )
    return parser.parse_args()


def main():
    args = get_args()

    print("domain gap embedding loaded")

    generator = DomainGenerator(
        args.save_dir,
        args.gen_src_dir,
        device=args.device,
        seed=args.seed,
    )

    trident_module = TRIDENTModule().to(args.device)
    weight_path = args.pre_trained_dir
    trident_module.load_state_dict(torch.load(weight_path))
    trident_module.eval()

    generator.generate(
        neg_prompt=args.neg_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        noise_level=args.noise_level,
        n_per_prompt=args.n_per_prompt,
        n_batch=args.n_batch,
        save_grid=args.save_grid,
        split_model=trident_module
    )


if __name__ == "__main__":
    main()
