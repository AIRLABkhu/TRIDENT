#!/usr/bin/bash

#SBATCH -J instructblip_cartoon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v4
#SBATCH -t 4-0
#SBATCH -o logs/slurm-%A.out

cd /data/choiyy0313/repos/TRIDENT
source init_domain.sh
conda activate blip
cd /data/choiyy0313/repos/TRIDENT/blip
python instructblip_qa.py --gen_root /data3/local_datasets/Diffusion_text_only --source_domain art_painting --target_domain cartoon  --class_name person --output_json json/Diffusion_text_only_art_painting2cartoon_person.json

python instructblip_qa.py --gen_root /data2/local_datasets/TRI_20250302_PACS_CLEANED/ACP --source_domain cartoon --target_domain art_painting --output_json json/TRI_cartoon2art_painting.json
python instructblip_qa.py --gen_root /data2/local_datasets/TRI_20250302_PACS_CLEANED/ACP --source_domain cartoon --target_domain photo --output_json json/TRI_cartoon2photo.json