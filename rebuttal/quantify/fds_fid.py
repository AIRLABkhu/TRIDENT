import os
import random
import shutil
from cleanfid import fid

def sample_image_paths(root_dir, num_samples, exts={'.png', '.jpg', '.jpeg'}):
    all_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in exts:
                full_path = os.path.join(dirpath, fname)
                all_paths.append(full_path)

    if len(all_paths) < num_samples:
        print(f"[WARNING] Only {len(all_paths)} images found, returning all.")
        return all_paths
    return random.sample(all_paths, num_samples)

def copy_sampled_images(sampled_paths, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for i, src in enumerate(sampled_paths):
        dst = os.path.join(save_dir, f"{i:04d}_" + os.path.basename(src))
        shutil.copy2(src, dst)

def compute_fid_vs_all_real(fds_root: str, real_root: str, save_root: str, output_log: str = "fds_fid_all_vs_all.txt"):
    os.makedirs(os.path.dirname(output_log) or ".", exist_ok=True)
    fid_scores = {}

    real_domains = sorted([d for d in os.listdir(real_root) if os.path.isdir(os.path.join(real_root, d))])

    with open(output_log, "w") as f:
        for folder in sorted(os.listdir(fds_root)):
            folder_path = os.path.join(fds_root, folder)
            if not os.path.isdir(folder_path):
                continue

            sample_paths = sample_image_paths(folder_path, 2000)
            save_folder = os.path.join(save_root, folder)
            copy_sampled_images(sample_paths, save_folder)

            fid_scores[folder] = {}

            for real_domain in real_domains:
                real_path = os.path.join(real_root, real_domain)

                if not os.path.exists(real_path):
                    print(f"[WARNING] Real domain not found: {real_path}")
                    continue

                score = fid.compute_fid(real_path, save_folder, mode="clean")
                fid_scores[folder][real_domain] = score

                log_line = f"{folder}\tvs {real_domain}\tFID: {score:.2f}"
                print(f"[FID] {log_line}")
                f.write(log_line + "\n")

    return fid_scores

# 사용 예시
if __name__ == "__main__":
    fds_gen_root = "/data/choiyy0313/repos/FDS/save/dm/PACS/123/generation"
    real_root = "/data2/local_datasets/PACS"
    save_root = "/data/choiyy0313/repos/TRIDENT/rebuttal/fds_samples/123"
    compute_fid_vs_all_real(fds_gen_root, real_root, save_root, output_log="./fds_fid_123_all.txt")
