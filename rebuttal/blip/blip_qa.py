import os
import re
import torch
import argparse
from PIL import Image
import random
from tqdm import tqdm
import pandas as pd
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util


parser = argparse.ArgumentParser()
parser.add_argument("--gen_root", type=str, required=True, help="Root dir of generated images (e.g., /PACS/ACS)")
parser.add_argument("--origin_root", type=str, default="/data2/local_datasets/PACS", help="Original dataset root")
parser.add_argument("--output_csv", type=str, default="semantic_similarity_log.csv")
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
blip2_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl", device_map="auto", torch_dtype=torch.float16
)
bert_model = SentenceTransformer("all-mpnet-base-v2").to(device)

# ------------------------ #
# Caption / Similarity Fn  #
# ------------------------ #
def generate_blip2_caption(image, prompt="Describe the image in detail."):
    inputs = blip2_processor(image, prompt, return_tensors="pt").to(device)
    output = blip2_model.generate(**inputs, max_new_tokens=75)
    return blip2_processor.tokenizer.decode(output[0], skip_special_tokens=True).strip()

def compare_semantic_similarity(img1, img2):
    cap1 = remove_domain_terms(generate_blip2_caption(img1))
    cap2 = remove_domain_terms(generate_blip2_caption(img2))
    emb1 = bert_model.encode(cap1, convert_to_tensor=True, device=device)
    emb2 = bert_model.encode(cap2, convert_to_tensor=True, device=device)
    return util.cos_sim(emb1, emb2).item()

prompt = "Describe the image in as much detail as possible, including objects, background, actions, and scene context."
# prompt = "Describe the image in full detail, including the objects, background, actions, and scene layout. However, do not mention anything about the visual style, domain, or medium (e.g., do not say 'cartoon', 'painting', 'sketch', or 'photo')."

# -------- Random Sampling --------
for transfer_dir in os.listdir(args.gen_root):
    if "2" not in transfer_dir:
        continue

    transfer_path = os.path.join(args.gen_root, transfer_dir)
    class_folders = [f for f in os.listdir(transfer_path) if os.path.isdir(os.path.join(transfer_path, f))]
    if not class_folders:
        continue

    # 랜덤 class 폴더 선택
    selected_class = random.choice(class_folders)
    class_path = os.path.join(transfer_path, selected_class)

    print(f"\n[Transfer: {transfer_dir}] Class: {selected_class}")

    # 이미지 파일 중 랜덤 10장
    image_files = [f for f in os.listdir(class_path) if f.endswith(".jpg")]
    sample_images = random.sample(image_files, min(10, len(image_files)))

    for fname in sample_images:
        try:
            orig_fname = fname.split("_0_")[1]
            orig_image_path = os.path.join(args.origin_root, transfer_dir.split("2")[0], selected_class, orig_fname)
            orig_image = Image.open(orig_image_path).convert("RGB")
            gen_image = Image.open(os.path.join(class_path, fname)).convert("RGB")
            
            gen_caption = generate_blip2_caption(gen_image, prompt=prompt)
            orig_caption = generate_blip2_caption(orig_image, prompt=prompt)
            
            gen_emb = bert_model.encode(gen_caption, convert_to_tensor=True, device=device)
            orig_emb = bert_model.encode(orig_caption, convert_to_tensor=True, device=device)
            similarity = util.cos_sim(gen_emb, orig_emb).item()
            
            print(f"\n원본 파일 이름 > {orig_fname}")
            print(f"생성 파일 이름 > {fname}")
            print(f"원본 캡션 > {orig_caption}")
            print(f"생성 캡션 > {gen_caption}")
            print(f"similarity > {similarity}\n")
        except Exception as e:
            print(f"  [Error {fname}] {e}")