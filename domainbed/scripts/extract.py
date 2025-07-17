import argparse
import collections
import json
import os
import random
import sys
import time
import uuid
from tqdm import tqdm
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
import torchvision
import PIL

from domainbed import hparams_registry, datasets, algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader



class Namespace_(argparse.Namespace):
    def __init__(self, d: dict) -> None:
        self.__dict__.update(d)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--load_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--load_feats', type=str, default=None, help='load_pretrained featurizer')
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation", "cross_domain"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')

    args = parser.parse_args()
    
    start_step = 0
    algorithm_dict = None
    
    hparams = hparams_registry.default_hparams('ERM', args.dataset)
    hparams.update(json.loads(args.hparams))
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = f"cuda:{args.device}"
    else:
        device = "cpu"

    dataset = vars(datasets)[args.dataset](args.data_dir, args.test_envs, hparams)

    
    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []

        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))

        in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    
    algorithm_class = algorithms.get_algorithm_class('get_baseline')
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes, len(dataset) - len(args.test_envs), hparams)
    
    pretrained_pkl = torch.load(Path(args.load_dir, 'model.pkl'), map_location='cpu')
    algorithm_dict = pretrained_pkl['model_dict']
    print(algorithm.load_state_dict(algorithm_dict))
    algorithm.to(device)
    
    algorithm.eval()
    
    save_dict = {}
    evals = zip(eval_loader_names, eval_loaders)
    for name, loader in evals:
        if str(args.test_envs[0]) in name:
            encode_mode = 'q'
        else:
            encode_mode = 'p'
        print(f'extracting features from envs {name}_{encode_mode}')
        y_minibatches = []
        z_minibatches = []
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            with torch.no_grad():
                z = algorithm.predict(x)
            y_minibatches.append(y)
            z_minibatches.append(z.cpu())
        y_cat = torch.cat(y_minibatches)
        z_cat = torch.cat(z_minibatches)
        save_dict[f'y_{name}_{encode_mode}'] = y_cat.numpy()
        save_dict[f'z_{name}_{encode_mode}'] = z_cat.numpy()
    save_dir = Path(args.load_dir)
    np.savez(save_dir.joinpath(f'data.npz'), **save_dict)
    
    