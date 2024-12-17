import argparse
import os
from typing import List, Dict, Any

import clip
import numpy as np
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from transformers import (
    CLIPProcessor,
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)

def is_folder_of_folders(dir: str) -> bool:
    return any(map(lambda x: os.path.isdir(os.path.join(dir, x)), os.listdir(dir)))

def data_loader_clipProc(dir: str, preprocess: CLIPProcessor, limit: int = 128) -> List[torch.Tensor]:
    src_set = []
    print(f"Loading images from {dir}... ", end="")
    
    if is_folder_of_folders(dir):
        classes = [os.path.join(dir, class_name) for class_name in os.listdir(dir) if os.path.isdir(os.path.join(dir, class_name))]
        # np.random.shuffle(classes)
        
        for class_dir in classes:
            files = os.listdir(class_dir)
            files = sorted(files)
            # np.random.shuffle(files)
            for f in files[:limit]:
                with Image.open(os.path.join(class_dir, f)) as img:
                    src_set.append(img.copy())
        assert len(src_set) > 0, f"\nSource directory {dir} is empty"
        print(f"Loaded {len(src_set)} images")
    else:
        files = os.listdir(dir)
        np.random.shuffle(files)
        for f in files[:limit]:
            with Image.open(os.path.join(dir, f)) as img:
                src_set.append(img.copy())
        assert len(src_set) > 0, f"Source directory {dir} is empty"
    
    src_blob = preprocess(images=src_set, return_tensors="pt").pixel_values
    return src_blob

@torch.no_grad()
def extract_mean_representations(
    root: str,
    domain: str,
    clip_model_name: str = "ViT-B/32",
    device: str = "cuda",
    save_loc: str = "output/domain_reps/"
) -> None:
    clip_processor = CLIPImageProcessor.from_pretrained(
        clip_model_name, subfolder="feature_extractor"
    )
    clip_vision_encoder = CLIPVisionModelWithProjection.from_pretrained(
        clip_model_name, subfolder="image_encoder"
    ).to(device)

    if not os.path.exists(os.path.join(save_loc, domain)):
        os.makedirs(os.path.join(save_loc, domain))

    dir = os.path.join(root, domain)

    if is_folder_of_folders(dir):
        classes = [os.path.join(dir, class_name) for class_name in os.listdir(dir) if os.path.isdir(os.path.join(dir, class_name))]
        class_len = [len(os.listdir(c)) for c in classes]
        min_class = np.array(class_len).min()
        print(f'min nums of class is : {min_class}')
        
        class_total = []
        for class_dir in classes:
            class_name = os.path.basename(class_dir)
            blob = data_loader_clipProc(class_dir, clip_processor, limit=min_class).to(device)

            image_reps = clip_vision_encoder(blob).image_embeds
            mean_rep = torch.mean(image_reps, dim=0)

            torch.save(mean_rep.cpu(), os.path.join(save_loc, domain, f"{class_name}_mean_rep.pt"))
            torch.save(image_reps.cpu(), os.path.join(save_loc, domain, f"{class_name}_image_reps.pt"))
            class_total.append(mean_rep.cpu())

            print(f"Saved {class_name}: mean_rep and image_reps.")

        class_total = torch.stack(class_total)
        domain_mean = class_total.mean(dim=0)
        torch.save(domain_mean, os.path.join(save_loc, domain, f"{domain}_mean.pt"))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=str, help="Path to the directory containing domain folders"
    )
    parser.add_argument(
        "--domain", type=str, help="Name of folder containitng images"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        help="Path to save the mean representations and image representations",
    )
    parser.add_argument(
        "--clip_model_name", type=str, default="stabilityai/stable-diffusion-2-1-unclip", help="CLIP model name"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for CLIP"
    )
    return parser.parse_args()
    

def main():
    args = get_args()

    extract_mean_representations(
        args.root,
        args.domain,
        clip_model_name=args.clip_model_name,
        device=args.device,
        save_loc=args.save_dir
    )


if __name__ == "__main__":
    main()
