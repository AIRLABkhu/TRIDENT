import os
import re
import torch
import argparse
from PIL import Image
from tqdm import tqdm
import pandas as pd
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util

# ---------------- #
# Argument Parsing #
# ---------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--gen_root", type=str, required=True, help="Root dir of generated images (e.g., /PACS/ACS)")
parser.add_argument("--origin_root", type=str, default="/data2/local_datasets/PACS", help="Original dataset root")
parser.add_argument("--output_csv", type=str, default="semantic_similarity_log.csv")
args = parser.parse_args()

# ---------------- #
# Device + Models  #
# ---------------- #
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
    output = blip2_model.generate(**inputs, max_new_tokens=50)
    return blip2_processor.tokenizer.decode(output[0], skip_special_tokens=True).strip()

def remove_domain_terms(caption):
    domain_terms = ["art painting", "art", "painting", "cartoon", "sketch", "photo", "drawing", "illustration"]
    for term in domain_terms:
        caption = re.sub(rf"\b{re.escape(term)}\b", "", caption.lower())
    return caption.strip()

def compare_semantic_similarity(img1, img2):
    cap1 = remove_domain_terms(generate_blip2_caption(img1))
    cap2 = remove_domain_terms(generate_blip2_caption(img2))
    emb1 = bert_model.encode(cap1, convert_to_tensor=True, device=device)
    emb2 = bert_model.encode(cap2, convert_to_tensor=True, device=device)
    return util.cos_sim(emb1, emb2).item()

# -------------------- #
# Main Traversal Logic #
# -------------------- #
results = []

for transfer_dir in os.listdir(args.gen_root):
    if "2" not in transfer_dir:
        continue
    source_domain, _ = transfer_dir.split("2")
    transfer_path = os.path.join(args.gen_root, transfer_dir)

    for cls in os.listdir(transfer_path):
        gen_cls_path = os.path.join(transfer_path, cls)
        origin_cls_path = os.path.join(args.origin_root, source_domain, cls)
        if not os.path.exists(origin_cls_path):
            continue

        similarities = []
        for fname in tqdm(os.listdir(gen_cls_path), desc=f"{transfer_dir}/{cls}"):
            if not fname.endswith(".jpg") or "_0_" not in fname:
                continue
            orig_fname = fname.split("_0_")[1]
            gen_path = os.path.join(gen_cls_path, fname)
            orig_path = os.path.join(origin_cls_path, orig_fname)
            if not os.path.exists(orig_path):
                continue
            try:
                sim = compare_semantic_similarity(
                    Image.open(orig_path).convert("RGB"),
                    Image.open(gen_path).convert("RGB")
                )
                similarities.append(sim)
            except Exception as e:
                print(f"Error processing {fname}: {e}")

        if similarities:
            avg_sim = sum(similarities) / len(similarities)
            results.append({
                "transfer": transfer_dir,
                "class": cls,
                "avg_similarity": avg_sim,
                "count": len(similarities)
            })

# -------------------- #
# Save or Print Result #
# -------------------- #
df = pd.DataFrame(results)
df.to_csv(args.output_csv, index=False)
print(df.groupby("transfer")["avg_similarity"].mean())