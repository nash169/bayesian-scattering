import torch


def average_distance(x):
    return torch.cdist(x, x, p=2.0, compute_mode='donot_use_mm_for_euclid_dist').mean()
