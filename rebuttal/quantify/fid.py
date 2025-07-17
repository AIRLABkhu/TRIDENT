import os
from cleanfid import fid

def compute_fid_to_all_domains(gen_root_dir: str, real_root_dir: str, output_log: str = "fid_all_vs_all.txt"):
    os.makedirs(os.path.dirname(output_log), exist_ok=True)
    fid_scores = {}

    real_domains = sorted([d for d in os.listdir(real_root_dir) if os.path.isdir(os.path.join(real_root_dir, d))])

    with open(output_log, "w") as f:
        for gen_folder in sorted(os.listdir(gen_root_dir)):
            gen_path = os.path.join(gen_root_dir, gen_folder)
            if not os.path.isdir(gen_path):
                continue

            fid_scores[gen_folder] = {}

            for real_domain in real_domains:
                real_path = os.path.join(real_root_dir, real_domain)

                if not os.path.exists(real_path):
                    print(f"[WARNING] Real domain path does not exist: {real_path}")
                    continue

                score = fid.compute_fid(real_path, gen_path)
                fid_scores[gen_folder][real_domain] = score

                log_line = f"{gen_folder}\tvs {real_domain}\tFID: {score:.2f}"
                print(f"[FID] {log_line}")
                f.write(log_line + "\n")

    return fid_scores

# 예시 사용
if __name__ == "__main__":
    gen_root = "/data2/local_datasets/TRI_20250302_PACS_CLEANED/ACS"
    real_root = "/data2/local_datasets/PACS"
    compute_fid_to_all_domains(gen_root, real_root, output_log="./ACS_fid_all_domains.txt")
