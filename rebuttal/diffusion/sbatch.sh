#!/usr/bin/bash

#SBATCH -J generate_text_diffusion
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v4
#SBATCH -t 4-0
#SBATCH -o logs/slurm-%A.out

cd /data/choiyy0313/repos/TRIDENT
source init_doge.sh
conda activate doge
cd /data/choiyy0313/repos/TRIDENT/diffusion

python text_diffusion.py \
    --source_domain art_painting \
    --target_domain photo \
    --class_name person \
    --save_path /data2/local_datasets/Diffusion_text

python text_diffusion.py \
    --source_domain photo \
    --target_domain art_painting \
    --class_name person \
    --save_path /data2/local_datasets/Diffusion_text

# declare -a source_domains=("art_painting" "cartoon" "photo" "sketch")
# declare -a target_domains=("art_painting" "cartoon" "photo" "sketch")

# for src_domain in "${source_domains[@]}"; do
#   for target_domain in "${target_domains[@]}"; do
#     if [ "$src_domain" != "$target_domain" ]; then
#       python text_diffusion.py \
#         --source_domain photo \
#         --target_domain cartoon \
#         --class_name person \
#         --save_path /local_datasets/Diffusion_text
#     fi
#   done
# done