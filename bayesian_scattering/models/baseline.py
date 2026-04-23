#!/usr/bin/env python
# encoding: utf-8

import gpytorch
import torch
import torch.nn as nn


class Baseline(nn.Module):
    def __init__(
        self,
        labels: torch.Tensor,
        is_class: bool = False,
        num_classes: int | None = None,
    ):
        super().__init__()
        labels = labels.reshape(-1)
        self.is_class = is_class
        self.register_buffer("labels", labels)

        if is_class:
            if num_classes is None:
                num_classes = int(labels.max().item()) + 1
            class_probs = torch.bincount(labels.long(), minlength=num_classes).float()
            class_probs /= class_probs.sum().clamp(min=1.0)
            self.register_buffer("class_probs", class_probs)

    def forward(self, x):
        batch_size = x.shape[0] if x.ndim > 1 else 1
        if self.is_class:
            return self.class_probs.unsqueeze(0).expand(batch_size, -1)
        return self.labels.mean()

    def posterior(self, x):
        batch_size = x.shape[0] if x.ndim > 1 else 1
        if self.is_class:
            return self.class_probs.unsqueeze(0).expand(batch_size, -1)

        mean = self.labels.mean() * torch.ones(batch_size, device=x.device)
        covar = self.labels.var() * torch.eye(batch_size, device=x.device)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
