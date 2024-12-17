src_domain="photo"  # Change this to fix to any domain you'd like
declare -a classes=("dog" "elephant" "giraffe" "guitar" "horse" "house" "person")

for class in "${classes[@]}"; do
  python generate_trident_single.py --seed=42 \
    --gen_src_dir="PACS/$src_domain/$class" \
    --pre_trained_dir="pretrained_trident/${src_domain}_trident.pt" \
    --save_dir="output_trident/SDG/${src_domain}/${class}" \
    --n_batch=10 --n_per_prompt=1 --num_inference_steps=20 --device="cuda:0" \
    --neg_prompt "blurry, blurred, ambiguous, blending, opaque, translucent, layering, shading, mixing, ugly, tiling, poorly drawn face, out of frame, mutation, disfigured, deformed, blurry, bad art, bad anatomy, text, watermark, grainy, underexposed, unreal architecture, unreal sky, weird colors" \
    --guidance_scale=5.0 
done
