import argparse
import os
from typing import *

import clip
import numpy as np
import torch
import torch.nn as nn
import torchvision
import nltk
from nltk.corpus import brown
from PIL import Image
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from transformers import (
    CLIPProcessor,
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTokenizer, CLIPTextModel,
)

from src.models.trident import TRIDENTModule

import random
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

def get_reps(domain, root):
    root = os.path.join(root, domain)
    print(root)
    class_list = os.listdir(root)
    sorted_class_list = sorted(class_list)

    class_mean_list = []
    dom_mean_list = []
    class_list = []

    for cl in sorted_class_list:
        if 'rep' in cl:
            if 'mean' in cl:
                class_mean_list.append(cl)
            else:
                class_list.append(cl)
        else:
            dom_mean_list.append(cl)

    print("Class List:", class_list)
    print("Class List:", len(class_list))
    print("Class Mean List:", class_mean_list)
    print("Class Mean List:", len(class_mean_list))
    print("Domain Mean List:", dom_mean_list)

    reps = []
    targets = []
    print('load reps')
    for c, class_rep in enumerate(class_list):
        path = os.path.join(root, class_rep)
        print(path)
        class_rep = torch.load(path, map_location='cpu')
        trg = torch.full((class_rep.size(0),), c)
        targets.append(trg)
        reps.append(class_rep)

    class_mean_reps = []
    print('load mean reps')
    for c, class_rep in enumerate(class_mean_list):
        path = os.path.join(root, class_rep)
        class_rep = torch.load(path, map_location='cpu')
        class_mean_reps.append(class_rep)
        print(path)

    path = os.path.join(root, dom_mean_list[0])
    print(path)
    dom_mean_reps = torch.load(path, map_location='cpu')

    targets = torch.cat(targets)
    reps = torch.cat(reps)
    class_mean_reps = torch.stack(class_mean_reps, dim=0)

    return targets, reps, class_mean_reps, dom_mean_reps

def get_args():
    parser = argparse.ArgumentParser(
        description="Generation with domain embedding guidance"
    )
    parser.add_argument("--root", type=str, default='PACS_reps/')
    parser.add_argument("--domain", type=str)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save-dir", type=str)
    parser.add_argument("--steps", type=int, default=150)
    return parser.parse_args()

def orthogonal_loss(v1, v2):
    v1 = v1 - v1.mean(dim=1, keepdim=True)
    v2 = v2 - v2.mean(dim=1, keepdim=True)
    v1 = F.normalize(v1, dim=1)
    v2 = F.normalize(v2, dim=1)
    correlation = (v1 * v2).sum(dim=1).pow(2).mean()
    return correlation

def calculate_total_loss(domain_vector, class_vector, attribute_vector, target_domain, target_class, target_attr):
    orth_loss = (
        orthogonal_loss(attribute_vector, target_class) +
        orthogonal_loss(attribute_vector, target_domain.unsqueeze(0))
    )
    recon_loss_attr = (
        F.mse_loss(domain_vector, target_domain) + 
        F.mse_loss(class_vector, target_class)
    )

    # Total loss
    total_loss = recon_loss_attr + 0.1 * orth_loss 
    return total_loss

def main():
    args = get_args()
    os.makedirs(f'{args.save_dir}', exist_ok=True)
    save_path = f'{args.save_dir}/{args.domain}_trident.pt'
    device = args.device
    targets, reps, class_mean_reps, dom_mean_reps = get_reps(args.domain, args.root)

    reps_dataset = TensorDataset(reps, targets)
    len_dataset = len(reps_dataset)
    train_len = int(len_dataset*0.8)
    train_dataset, val_dataset = torch.utils.data.random_split(reps_dataset, [train_len, len_dataset-train_len])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)
    
    model = TRIDENTModule()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    model.to(device)

    class_mean_reps = class_mean_reps.to(device)
    dom_mean_reps = dom_mean_reps.to(device)

    total_loss_list = []
    best_loss = float('inf')
    for step in tqdm(range(args.steps)):
        total_loss, count = 0, 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            target = target.long()

            domain_vector, class_vector, dom_att_vector, cls_att_vector = model(data)
            attribute_vector = dom_att_vector + cls_att_vector
            out = domain_vector + class_vector + attribute_vector
            loss = calculate_total_loss(domain_vector, class_vector, attribute_vector, dom_mean_reps, (class_mean_reps[target] - dom_mean_reps), (data - class_mean_reps[target])) + F.mse_loss(out, data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = data.size(0)
            total_loss += loss.item() * batch_size
            count += batch_size

        total_loss_list.append(total_loss / count)
        tqdm.write(f'Epoch {step + 1}, Loss: {total_loss_list[-1]:.4f}')
        scheduler.step()

        # for Eval    
        model.eval()
        with torch.no_grad():
            total_loss, count = 0, 0
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                target = target.long()

                domain_vector, class_vector, dom_att_vector, cls_att_vector = model(data)
                attribute_vector = dom_att_vector + cls_att_vector
                out = domain_vector + class_vector + attribute_vector
                loss = calculate_total_loss(domain_vector, class_vector, attribute_vector, dom_mean_reps, (class_mean_reps[target] - dom_mean_reps), (data - class_mean_reps[target])) + F.mse_loss(out, data)
                
                total_loss += loss
                count += 1

        eval_loss = total_loss/count
        if eval_loss < best_loss:
            print(step, "save!!!!!")
            best_loss = eval_loss
            torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    main()