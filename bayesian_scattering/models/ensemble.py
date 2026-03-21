#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
import gpytorch


class Ensemble(nn.Module):
    def __init__(
            self,
            model,
            dim,
            **kwargs
    ):
        super().__init__()
        self.ensemble = nn.ModuleList([model(**kwargs) for _ in range(dim)])

    def forward(self, x):
        preds = [m(x) for m in self.ensemble]
        return torch.stack(preds).mean(0)
        # pred = []
        # for model in self.ensemble:
        #     pred.append(model(x))
        # pred = torch.stack(pred).squeeze()
        # return torch.mean(pred, axis=0)

    def posterior(self, x):
        pred = []
        for model in self.ensemble:
            pred.append(model(x))
        pred = torch.stack(pred).squeeze()
        return gpytorch.distributions.MultivariateNormal(pred.mean(axis=0), pred.var(axis=0).diag())

    def __len__(self):
        return len(self.ensemble)

    def __getitem__(self, idx):
        return self.ensemble[idx]
