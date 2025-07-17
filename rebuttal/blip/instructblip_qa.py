import os
import re
import torch
import argparse
from PIL import Image
import random
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import torch
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--gen_root", type=str, required=True, help="Root dir of generated images (e.g., /PACS/ACS)")
parser.add_argument("--origin_root", type=str, default="/data2/local_datasets/PACS", help="Original dataset root")
parser.add_argument("--source_domain", type=str,required=True)
parser.add_argument("--target_domain", type=str,required=True)
parser.add_argument("--class_name", type=str,required=True)
parser.add_argument("--rename", action="store_true", help="Rename files to match original dataset format")
parser.add_argument("--output_json", type=str, default="semantic_similarity_log.csv")
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 모델 & 프로세서 불러오기
model = InstructBlipForConditionalGeneration.from_pretrained(
    "Salesforce/instructblip-flan-t5-xl", 
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
bert_model = SentenceTransformer("all-mpnet-base-v2").to(device)

def remove_domain_terms(caption):
    domain_terms = ["art painting", "art", "painting", "cartoon", "sketch", "photo", "drawing", "illustration"]
    for term in domain_terms:
        caption = re.sub(rf"\b{re.escape(term)}\b", "", caption.lower())
    return caption.strip()

instruction = "Describe the image thoroughly while strictly avoiding any mention of its visual style or domain. Focus entirely on what is shown in the image, not how it looks or how it was made."

results = []
source_domain, target_domain = args.source_domain, args.target_domain
print(f"🚩 Source Domain: {source_domain}, Target Domain: {target_domain}")
transfer_dir = f"{source_domain}2{target_domain}"
transfer_path = os.path.join(args.gen_root, transfer_dir)
cls = args.class_name

gen_cls_path = os.path.join(transfer_path, cls)
origin_cls_path = os.path.join(args.origin_root, source_domain, cls)

for fname in sorted(os.listdir(gen_cls_path)):
    if args.rename:
        orig_fname = fname.split("_0_")[1]
    else:
        orig_fname = fname
    gen_image_path = os.path.join(gen_cls_path, fname)
    orig_image_path = os.path.join(origin_cls_path, orig_fname)

    # Caption: Original
    orig_image = Image.open(orig_image_path).convert("RGB")
    orig_input = processor(images=orig_image, text=instruction, return_tensors="pt").to(device)
    orig_out = model.generate(**orig_input, max_new_tokens=100)
    orig_caption = processor.tokenizer.decode(orig_out[0], skip_special_tokens=True)
    orig_caption = remove_domain_terms(orig_caption)

    # Caption: Generated
    gen_image = Image.open(gen_image_path).convert("RGB")
    gen_input = processor(images=gen_image, text=instruction, return_tensors="pt").to(device)
    gen_out = model.generate(**gen_input, max_new_tokens=100)
    gen_caption = processor.tokenizer.decode(gen_out[0], skip_special_tokens=True)
    gen_caption = remove_domain_terms(gen_caption)

    # Similarity
    gen_emb = bert_model.encode(gen_caption, convert_to_tensor=True, device=device)
    orig_emb = bert_model.encode(orig_caption, convert_to_tensor=True, device=device)
    similarity = util.cos_sim(gen_emb, orig_emb).item()

    # 출력
    print(f"\n원본 파일 이름 > {orig_fname}")
    print(f"생성 파일 이름 > {fname}")
    print(f"원본 캡션 > {orig_caption}")
    print(f"생성 캡션 > {gen_caption}")
    print(f"similarity > {similarity:.4f}\n")

    # JSON에 저장할 샘플
    results.append({
        "source_domain": source_domain,
        "target_domain": target_domain,
        "class": cls,
        "original_image": orig_fname,
        "generated_image": fname,
        "original_caption": orig_caption,
        "generated_caption": gen_caption,
        "similarity": round(similarity, 4)
    })
        
print(f"✅ {cls} 클래스 처리 완료")


# 저장
output_json_path = args.output_json
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n🚩 결과 저장 완료: {output_json_path}")