#!/usr/bin/env python
# encoding: utf-8

import gpytorch
from botorch.models.gpytorch import GPyTorchModel


class ExactGP(gpytorch.models.ExactGP, GPyTorchModel):
    def __init__(
            self,
            train_x,
            train_y,
            likelihood,
            kernel
    ):
        super(ExactGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel
        self._num_outputs = 1  # for botorch compatibility

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
