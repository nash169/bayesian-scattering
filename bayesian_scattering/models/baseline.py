#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
import gpytorch


class Baseline(nn.Module):
    def __init__(
            self,
            labels
    ):
        super().__init__()
        self.labels = labels

    def forward(self, x):
        return self.labels.mean()

    def posterior(self, x):
        return gpytorch.distributions.MultivariateNormal(self.labels.mean() * torch.ones(x.shape[0] if len(x.shape) > 1 else 1).to(x.device), self.labels.var() * torch.eye(x.shape[0] if len(x.shape) > 1 else 1).to(x.device))
