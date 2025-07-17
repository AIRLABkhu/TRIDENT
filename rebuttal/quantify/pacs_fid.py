import os
from cleanfid import fid

DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]

def compute_cross_fid_matrix(pacs_root: str, output_log: str = "pacs_fid_matrix.txt"):
    os.makedirs(os.path.dirname(output_log), exist_ok=True)
    fid_matrix = {}
    
    with open(output_log, "w") as f:
        f.write("FID Matrix between PACS domains\n")
        f.write("\t" + "\t".join(DOMAINS) + "\n")

        for src in DOMAINS:
            row_scores = []
            for tgt in DOMAINS:
                if src == tgt:
                    fid_value = 0.0
                else:
                    src_path = os.path.join(pacs_root, src)
                    tgt_path = os.path.join(pacs_root, tgt)
                    fid_value = fid.compute_fid(tgt_path, src_path)  # FID(target, source)

                row_scores.append(fid_value)
                fid_matrix[(src, tgt)] = fid_value

            row_str = "\t".join(f"{v:.2f}" for v in row_scores)
            f.write(f"{src}\t{row_str}\n")

    print(f"✅ FID matrix saved to: {output_log}")
    return fid_matrix

# 예시 사용
if __name__ == "__main__":
    pacs_root = "/data2/local_datasets/PACS"  # 도메인별 폴더 포함한 PACS 루트 경로
    compute_cross_fid_matrix(pacs_root, output_log="./pacs_fid_matrix.txt")
