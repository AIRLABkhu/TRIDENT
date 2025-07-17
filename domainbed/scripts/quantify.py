import json
import random
import sys
import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
import scipy
from scipy.stats import gaussian_kde



def compute_div(p: Sequence[float], q: Sequence[float], probs: Sequence[int],
                eps_div: float, legacy_mode: bool = False) -> float:
    if not len(p) == len(q) == len(probs):
        raise ValueError
    div = 0
    for i in range(len(probs)):
        if p[i] < eps_div or q[i] < eps_div:
            if legacy_mode:
                div += abs(p[i] - q[i]) / probs[i]
            else:
                div += (np.sqrt(p[i]) - np.sqrt(q[i])) ** 2 / probs[i]
    div /= len(probs) * 2
    return div

def sep_data_npz(data):
    keys = data.files

    # 각각 담을 리스트 초기화
    y_p_list, z_p_list = [], []
    y_q_list, z_q_list = [], []

    # key 이름 기반 분류
    for k in keys:
        if k.startswith('y_') and k.endswith('_p'):
            y_p_list.append(data[k])
        elif k.startswith('z_') and k.endswith('_p'):
            z_p_list.append(data[k])
        elif k.startswith('y_') and k.endswith('_q'):
            y_q_list.append(data[k])
        elif k.startswith('z_') and k.endswith('_q'):
            z_q_list.append(data[k])

    # concat
    y_p = np.concatenate(y_p_list, axis=0)
    z_p = np.concatenate(z_p_list, axis=0)
    y_q = np.concatenate(y_q_list, axis=0)
    z_q = np.concatenate(z_q_list, axis=0)
    return  y_p, z_p, y_q, z_q
if __name__ == '__main__':
    # args, _ = config.parse_argument(Path(__file__).name)
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--feature_dir', type=str)
    # parser.add_argument('--save_dir', type=str)
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    save_dir = Path(args.feature_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    args_dir = save_dir.joinpath('args')
    args_dir.mkdir(exist_ok=True)
    

    data = np.load(Path(args.feature_dir, 'data.npz'))
    y_p, z_p, y_q, z_q = sep_data_npz(data)
    print(f'features loaded: (p) {z_p.shape}, (q) {z_q.shape}')
    print(f'labels   loaded: (p) {y_p.shape}, (q) {y_q.shape}')
    if len(z_p) != len(y_p) or len(z_q) != len(y_q):
        raise RuntimeError

    z_all = np.append(z_p, z_q, 0)
    scaler = StandardScaler().fit(z_all)
    z_all, z_p, z_q = map(scaler.transform, (z_all, z_p, z_q))

    print('computing KDE for importance sampling')
    sampling_pdf = gaussian_kde(z_all.T)
    points = sampling_pdf.resample(10000, seed=args.seed)
    probs = sampling_pdf(points)

    print('computing KDE for p and q')
    p = gaussian_kde(z_p.T)(points)
    q = gaussian_kde(z_q.T)(points)

    print('computing diversity shift')
    div = compute_div(p, q, probs, 1e-12, legacy_mode=True)
    print(f'Diversity shift: {div}')

    with save_dir.joinpath('quantify.json').open('w') as f:
        json.dump({'div': div}, f)