## Environment setup
The code has been tested with the following setup.

- PyTorch 2.0.1
- CUDA 11.7

```bash
conda env create --name trident --file=environment.yaml
```

## Usage
The examples of PACS.

1. Extract Domain Mean Embeddings

```bash
python extract_feat.py --root PACS --domain photo --save-dir output_reps --device="cuda:0"
```

2. pre-training the TRIDENT module

```bash
python train_module.py --root output_reps --domain photo --save-dir pretrained_trident --device="cuda:0"
```

3. Generate Datsaets
We provide bash file example for generating datasets.
Multi-source 

```bash
bash generate_multi.sh
```
Single-source

```bash
bash generate_single.sh
```
4. Cleaning Datasets
When cleaning the data, you need to load pretrained moduel which is trained with target dataset.

```bash
python cleaning_dataset.py --domain photo --ckpt pretrained_trident/photo_trident.pt --data-dir output_trident/SDG --save-dir output_trident_cleaned --reps-root output_reps
```
## Acknowledgement
This work was partly supported by an Institute of Information and Communications Technology Planning and Evaluation (IITP) grant funded by the Korean government (MSIT) (No.RS-2022-00155911, Artificial Intelligence Convergence Innovation Human Resources Development (KyungHee University)).
