#!/usr/bin/env python
# encoding: utf-8

import os
import re
import glob
import numpy as np
import torch
from scipy.spatial import cKDTree


def sample_index(path, prefix):
    numbers = []
    pattern = re.compile(rf"^{re.escape(prefix)}.*_(\d+)\.[^.]+$")
    # print(pattern)

    for filename in os.listdir(path):
        # print(filename)
        match = pattern.match(filename)
        if match:
            numbers.append(int(match.group(1)))

    return numbers


def sample_loader(path, prefix):
    filename = glob.glob(os.path.join(path, prefix + ".*"))[0]
    ext = os.path.splitext(filename)[1]

    if ext == ".npy":
        return torch.from_numpy(np.load(filename))
    elif ext in (".pt", ".pth"):
        return torch.load(filename, map_location="cpu")
    else:
        raise ValueError(f"Unsupported extension: {ext}")


def sample_duplicates(
        data: torch.Tensor,
        tol: float
):
    X = data.reshape(data.shape[0], -1).detach().cpu().numpy()  # [N, D]
    tree = cKDTree(X, leafsize=64)
    neighbors = tree.query_ball_tree(tree, r=tol)
    pairs = []
    for i, nbrs in enumerate(neighbors):
        for j in nbrs:
            if j > i:
                pairs.append((i, j))
    return pairs
