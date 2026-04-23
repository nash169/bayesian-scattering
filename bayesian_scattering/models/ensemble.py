#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
import gpytorch


class Ensemble(nn.Module):
    def __init__(self, model, dim, **kwargs):
        super().__init__()
        self.is_class = kwargs.pop("is_class", False)
        self.ensemble = nn.ModuleList([model(**kwargs) for _ in range(dim)])

    def _stack_predictions(self, x):
        return torch.stack([model(x) for model in self.ensemble])

    def forward(self, x):
        preds = self._stack_predictions(x)
        if self.is_class:
            return preds.softmax(dim=-1).mean(0)
        return preds.mean(0)
        # pred = []
        # for model in self.ensemble:
        #     pred.append(model(x))
        # pred = torch.stack(pred).squeeze()
        # return torch.mean(pred, axis=0)

    def posterior(self, x):
        pred = self._stack_predictions(x).squeeze()
        if self.is_class:
            return pred.softmax(dim=-1).mean(0)
        return gpytorch.distributions.MultivariateNormal(
            pred.mean(axis=0), pred.var(axis=0).diag()
        )

    def __len__(self):
        return len(self.ensemble)

    def __getitem__(self, idx):
        return self.ensemble[idx]
