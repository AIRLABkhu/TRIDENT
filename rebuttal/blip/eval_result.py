import json
import numpy as np

# JSON 파일 경로
json_path = "json/Diffusion_art_painting2cartoon_person.json"  # ← 경로에 맞게 수정

# JSON 로드
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# similarity 값만 추출
similarities = [entry["similarity"] for entry in data if "similarity" in entry]

# 평균과 표준편차 계산
mean_sim = np.mean(similarities)
std_sim = np.std(similarities)

# 출력
print(f"📊 평균 유사도: {mean_sim:.4f}")
print(f"📈 표준편차:   {std_sim:.4f}")