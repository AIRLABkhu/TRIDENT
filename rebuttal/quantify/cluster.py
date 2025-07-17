import numpy as np
from sklearn.metrics import pairwise_distances
import pandas as pd

import os
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples
import torch
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoModel, AutoImageProcessor
from scipy.spatial import ConvexHull


def get_clip_features(image_paths, model, processor, device, model_type):
    features = []
    for path in tqdm(image_paths, desc=f"Extracting {model_type} features"):
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            if model_type=="clip":
                feature = model.get_image_features(**inputs)
            elif model_type=="vit":
                feature = model(**inputs).last_hidden_state[:, 0, :]  # Use the CLS token for ViT
            # feature = feature / feature.norm(dim=-1, keepdim=True)  # normalize
        features.append(feature.cpu().numpy())
    return np.vstack(features)

def collect_image_paths(root, num_samples, domain_prefix):
    domain_list = sorted(os.listdir(root))
    all_paths, all_labels = [], []
    class_list = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
    for d, domain_name in enumerate(domain_list):
        for c, class_name in enumerate(class_list):
            class_path = os.path.join(root, domain_name, class_name)
            if not os.path.exists(class_path):
                continue
            img_files = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
            selected = random.sample(img_files, min(num_samples, len(img_files)))
            all_paths.extend(selected)
            all_labels.extend([(d, c)] * len(selected))
    return all_paths, all_labels


def mean_domain_centroid_dist(features, labels, domain_count=3):
    centroids = []
    for d in range(domain_count):
        idx = [i for i, (dom, _) in enumerate(labels) if dom == d]
        centroids.append(np.mean(features[idx], axis=0))
    dists = pairwise_distances(centroids)
    return np.sum(np.triu(dists, 1)) / (domain_count * (domain_count - 1) / 2)

def overall_variance(features):
    return np.mean(np.var(features, axis=0))

def intra_cluster_distance(features, labels, domain_count=3, class_count=2):
    total_dist = 0
    count = 0
    for d in range(domain_count):
        for c in range(class_count):
            idx = [i for i, (dom, cls) in enumerate(labels) if dom == d and cls == c]
            cluster = features[idx]
            center = np.mean(cluster, axis=0)
            dists = np.linalg.norm(cluster - center, axis=1)
            total_dist += np.sum(dists)
            count += len(dists)
    return total_dist / count


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "clip":
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    elif args.model == "vit":
        model_name = "google/vit-base-patch16-224-in21k"
        model = AutoModel.from_pretrained(model_name).to(device).eval()
        processor = AutoImageProcessor.from_pretrained(model_name)

    fds_paths, fds_labels = collect_image_paths(args.fds_root, args.num_per_domain , domain_prefix="FDS")
    tri_paths, tri_labels = collect_image_paths(args.trident_root, int(args.num_per_domain / 2), domain_prefix="TRI")
    # orig_paths, orig_labels = collect_image_paths(args.origin_root, args.num_per_domain, domain_prefix="ORIGIN")

    fds_features = get_clip_features(fds_paths, model, processor, device, model_type=args.model)
    tri_features = get_clip_features(tri_paths, model, processor, device, model_type=args.model)
    # orig_features = get_clip_features(orig_paths, model, processor, device)
    
    results = {
        "FDS": {
            "Inter-domain Distance": mean_domain_centroid_dist(fds_features, fds_labels, domain_count=3),
            "Total Variance": overall_variance(fds_features),
            "Intra-cluster Distance": intra_cluster_distance(fds_features, fds_labels, domain_count=3)
        },
        "TRIDENT": {
            "Inter-domain Distance": mean_domain_centroid_dist(tri_features, tri_labels, domain_count=6),
            "Total Variance": overall_variance(tri_features),
            "Intra-cluster Distance": intra_cluster_distance(tri_features, tri_labels, domain_count=6)
        }
    }
    print(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to FDS dataset root")
    parser.add_argument("--fds_root", type=str, required=True, help="Path to FDS dataset root")
    parser.add_argument("--trident_root", type=str, required=True, help="Path to TRIDENT dataset root")
    parser.add_argument("--num_per_domain", type=int, default=100, help="Number of samples per domain")
    args = parser.parse_args()
    main(args)
