#!/usr/bin/env python
# encoding: utf-8

import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import (
    VariationalStrategy,
    IndependentMultitaskVariationalStrategy,
)
from gpytorch.distributions import MultivariateNormal


class MultitaskApproxGP(ApproximateGP):
    def __init__(
        self,
        inducing_points,
        num_classes,
        likelihood,
        kernel,
    ):
        batch_shape = torch.Size([num_classes])
        if inducing_points.ndim == 2:
            inducing_points = (
                inducing_points.unsqueeze(0).expand(num_classes, -1, -1).contiguous()
            )

        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=batch_shape
        )

        variational_strategy = IndependentMultitaskVariationalStrategy(
            VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True,
            ),
            num_tasks=num_classes,
        )

        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        self.covar_module = kernel
        self.likelihood = likelihood

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
