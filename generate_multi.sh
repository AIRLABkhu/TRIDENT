# src_domain="photo"  # Change this to fix to any domain you'd like
declare -a source_domains=("art_painting" "cartoon" "photo")
declare -a target_domains=("art_painting" "cartoon" "photo")
declare -a classes=("dog" "elephant" "giraffe" "guitar" "horse" "house" "person")

for src_domain in "${source_domains[@]}"; do
  for target_domain in "${target_domains[@]}"; do
    if [ "$src_domain" != "$target_domain" ]; then
      for class in "${classes[@]}"; do
        python generate_trident_multi.py --seed=42 \
          --gen_src_dir="PACS/$src_domain/$class" --gen_src_dir2="PACS/$target_domain" \
          --pre_trained_dir="pretrained_trident/${src_domain}_trident.pt" \
          --pre_trained_dir2="pretrained_trident/${target_domain}_trident.pt" \
          --save_dir="output_trident/MDG/${src_domain}/${target_domain}/${class}" \
          --n_batch=10 --n_per_prompt=1 --num_inference_steps=20 \
          --neg_prompt "blurry, blurred, ambiguous, blending, opaque, translucent, layering, shading, mixing, ugly, tiling, poorly drawn face, out of frame, mutation, disfigured, deformed, blurry, bad art, bad anatomy, text, watermark, grainy, underexposed, unreal architecture, unreal sky, weird colors" \
          --guidance_scale=5.0 
      done
    fi
  done
done