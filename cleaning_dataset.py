import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import argparse
import os
import shutil
from PIL import Image

from transformers import (
    CLIPProcessor,
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTokenizer, CLIPTextModel,
)

from src.models.trident import TRIDENTModule

def get_args():
    parser = argparse.ArgumentParser(
        description="Cleaning data"
    )
    parser.add_argument("--reps-root", type=str, default='PACS_reps/')
    parser.add_argument("--domain", type=str)
    
    parser.add_argument("--ckpt", type=str)

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--save-dir", type=str)

    return parser.parse_args()

def main():
    args = get_args()
    device = f"cuda:{args.device}"
    
    # get domain reps 
    reps_path = f'{args.reps_root}/{args.domain}'
    reps_files = sorted(os.listdir(reps_path))

    dom_feat = torch.load(f'{reps_path}/{args.domain}_mean.pt', map_location='cpu')
        
    cls_reps = []
    for f in reps_files:
        if 'mean_rep' in f:
            if args.domain not in f:
                cls_rep = torch.load(os.path.join(reps_path, f))
                cls_reps.append(cls_rep - dom_feat)
    cls_reps = torch.stack(cls_reps)

    # Load pretrianed domain module
    model = TRIDENTModule() 
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    # Load CLIP Model
    clip_processor = CLIPImageProcessor.from_pretrained(
            "stabilityai/stable-diffusion-2-1-unclip", subfolder="feature_extractor"
        )
    clip_vision_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "stabilityai/stable-diffusion-2-1-unclip", subfolder="image_encoder"
    ).to(device)

    # check directory
    dom_root = os.path.join(args.data_dir, args.domain)
    save_root = os.path.join(args.save_dir, args.domain)
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    clss = sorted(os.listdir(dom_root))
    for idx, c in enumerate(clss):
        if not os.path.exists(os.path.join(save_root, c)):
            os.makedirs(os.path.join(save_root, c))
        print(f'{c} class cleaning Start!')
        files = sorted(os.listdir(os.path.join(dom_root, c, 'data')))
        # breakpo?int()
        img_list = []
        count = 0
        length = len(files)
        min_loss = 100000
        for f in tqdm(files):
            with Image.open(os.path.join(dom_root, c, 'data', f)) as img:
                processed = img.copy()
            img_blob = clip_processor(images=processed, return_tensors="pt").pixel_values
            image_features = clip_vision_encoder(img_blob.to(device)).image_embeds
            domain_vector, class_vector, _, _ = model(image_features)
            loss = nn.functional.mse_loss(class_vector, cls_reps.to(device), reduction='none')
            if loss.mean(dim=-1)[idx] < min_loss:
                min_path = os.path.join(dom_root, c, 'data', f)
                min_loss = loss.mean(dim=-1)[idx]
            if idx == loss.mean(dim=-1).argmin():
                shutil.copy(os.path.join(dom_root, c, 'data', f), os.path.join(save_root, c))
                count += 1

        if count == 0:
            shutil.copy(min_path, os.path.join(save_root, c))
            count += 1
        print(f'{c} class cleaning Finished {count} / {length}')
        print('#######################################################')

if __name__ == "__main__":
    main()