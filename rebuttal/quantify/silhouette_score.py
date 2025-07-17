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
from scipy.spatial import ConvexHull

def get_clip_features(image_paths, model, processor, device):
    features = []
    for path in tqdm(image_paths, desc="Extracting CLIP features"):
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            feature = model.get_image_features(**inputs)
            # feature = feature / feature.norm(dim=-1, keepdim=True)  # normalize
        features.append(feature.cpu().numpy())
    return np.vstack(features)

def collect_image_paths(root, target_class, num_samples, domain_prefix):
    domain_list = sorted(os.listdir(root))
    all_paths, all_labels = [], []
    for d in domain_list:
        class_path = os.path.join(root, d, target_class)
        if not os.path.exists(class_path):
            continue
        img_files = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        selected = random.sample(img_files, min(num_samples, len(img_files)))
        all_paths.extend(selected)
        all_labels.extend([f"{domain_prefix}_{d}"] * len(selected))
    return all_paths, all_labels

def tsne_visualization(features_list, labels_list, title="t-SNE Comparison"):
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    import matplotlib.pyplot as plt

    # 1. Feature 합치기
    all_features = np.vstack(features_list)
    tsne = TSNE(n_components=2, random_state=42)
    emb_2d = tsne.fit_transform(all_features)

    # 2. 색상 팔레트 및 마커 세트
    color_map = plt.get_cmap('tab10')
    marker_styles = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', 'h', '+']  # 10개 마커 스타일

    # 3. 샘플 개수별 인덱싱
    counts = [f.shape[0] for f in features_list]
    cum_counts = np.cumsum([0] + counts)

    # # 4. 시각화
    plt.figure(figsize=(12, 10))
    for i, (start, end) in enumerate(zip(cum_counts[:-1], cum_counts[1:])):
        plt.scatter(
            emb_2d[start:end, 0],
            emb_2d[start:end, 1],
            c=[color_map(i % 10)] * (end - start),
            marker=marker_styles[i % len(marker_styles)],
            label=labels_list[i],
            alpha=0.7,
            s=40,
            edgecolors='k'
        )

    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=300)

    # 5. silhouette score 계산
    cluster_labels = []
    for i, count in enumerate(counts):
        cluster_labels += [i] * count

    sil_score = silhouette_score(all_features, cluster_labels)
    print(f"Silhouette Score: {sil_score:.4f}")

def compute_intra_class_diversity(features_list, labels_list):
    """
    features_list: list of (N_k, D) numpy arrays for each class
    labels_list: list of class names (strings), same length as features_list
    """
    assert len(features_list) == len(labels_list)
    
    intra_class_vars = []
    for feature in features_list:
        if feature.shape[0] < 2:
            continue  # skip classes with too few samples
        mu_k = feature.mean(axis=0, keepdims=True)  # (1, D)
        sq_dist = np.sum((feature - mu_k) ** 2, axis=1)  # (N_k,)
        var_k = sq_dist.mean()
        intra_class_vars.append(var_k)
    
    diversity_score = np.mean(intra_class_vars)
    print(f"Intra-class Diversity Score: {diversity_score:.4f}")
    return diversity_score
    
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    class_name = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
    
    features = []
    for cls in class_name:
        path, _ = collect_image_paths(args.data_root, cls, args.num_per_domain, domain_prefix="FDS")
        feature = get_clip_features(path, model, processor, device)
        features.append(feature)
    
    if args.mode == "intra_class_diversity":
        compute_intra_class_diversity(features, class_name)
    elif args.mode == "tsne_visualization":
        tsne_visualization(
            features_list=features,
            labels_list=class_name,
            title=args.title
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Path to FDS dataset root")
    parser.add_argument("--num_per_domain", type=int, default=100, help="Number of samples per domain")
    parser.add_argument("--title", type=str, default="t-SNE Visualization", help="Title for the t-SNE plot")
    parser.add_argument("--mode", type=str, )
    args = parser.parse_args()
    main(args)
