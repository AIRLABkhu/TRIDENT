# TRIDENT: Text-Free Data Augmentation Using Image Embedding Decomposition for Domain Generalization

This is the official implementation of the paper:

> **TRIDENT: Text-Free Data Augmentation Using Image Embedding Decomposition for Domain Generalization**

TRIDENT is a training-free, prompt-free, and interpretable image augmentation framework designed for domain generalization. By decomposing CLIP image embeddings and recombining domain and class components, it synthesizes diverse, controllable image variations without requiring any textual supervision.

---

## 📦 Environment Setup

- Python ≥ 3.10
- PyTorch 2.0.1
- CUDA 11.7

Install dependencies via conda:

```bash
conda env create -f environment.yaml
conda activate trident
```

---

## ⚙️ Usage Instructions

### 1. Extract CLIP Features

```bash
python extract_feat.py \
    --root PACS \
    --domain $domain \
    --save-dir output_reps \
    --device "cuda:0"
```

---

### 2. Train TRIDENT Embedding Module

```bash
python train_module.py \
    --root output_reps \
    --domain photo \
    --save-dir pretrained_trident \
    --device "cuda:0"
```

---

### 3. Generate Domain-Transferred Images

```bash
python generate_trident_multi.py --seed 42 \
    --gen_src_dir "PACS/$src_domain/$class" \
    --gen_src_dir2 "PACS/$target_domain" \
    --pre_trained_dir "20250302_TRI_FIN/PACS/ACP.pt" \
    --save_dir "${gen_dir}/${src_domain}2${target_domain}/${class}" \
    --n_batch 10 \
    --n_per_prompt 1 \
    --num_inference_steps 20 \
    --neg_prompt "blurry, blurred, ambiguous, blending, opaque, translucent, layering, shading, mixing, ugly, tiling, poorly drawn face, out of frame, mutation, disfigured, deformed, blurry, bad art, bad anatomy, text, watermark, grainy, underexposed, unreal architecture, unreal sky, weird colors" \
    --guidance_scale 5.0
```

---

### 4. Filter Inconsistent or Noisy Samples

```bash
python cleaning_dataset.py \
    --domain photo \
    --ckpt pretrained_trident/photo_trident.pt \
    --data-dir output_trident \
    --save-dir output_trident_cleaned \
    --reps-root output_reps
```

---

### 5. Train Domain Generalization Model (DomainBed)

```bash
python3 -m domainbed.scripts.train \
    --data_dir /data2/local_datasets/ \
    --output_dir results/HPARAMS_CLEANED/ACP_10/seed0 \
    --algorithm ERM \
    --dataset augmented_PACS \
    --hparams '{"resnet50_augmix":"True", "data_augmentation_root": $gen_dir}' \
    --test_env 3 \
    --trial_seed 0
```

---

## 🔍 References & Acknowledgements

We build upon the following open-source repositories:

- [DoGE (CVPR 2023)](https://github.com/humansensinglab/DoGE) – using CLIP and unCLIP for Image Generation
- [DomainBed (ICLR 2021)](https://github.com/facebookresearch/DomainBed) – Benchmark framework for DG
- [Trager et al. (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/papers/Trager_Linear_Spaces_of_Meanings_Compositional_Structures_in_Vision-Language_Models_ICCV_2023_paper.pdf) – Linear Spaces of Meanings: Compositional Structures in Vision-Language Models — insights on compositional embedding structures

---

## 📄 License

This project is licensed under the **MIT License**.

---

## ✏️ Citation

If you use this codebase, please cite our paper:



<!-- ```bibtex
@article{your2025trident,
  title={TRIDENT: Text-Free Data Augmentation Using Image Embedding Decomposition for Domain Generalization},
  author={Your Name and Collaborators},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025}
}
``` -->
