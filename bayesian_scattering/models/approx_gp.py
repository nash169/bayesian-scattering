#!/usr/bin/env python
# encoding: utf-8

import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.distributions import MultivariateNormal
from botorch.models.gpytorch import GPyTorchModel


class ApproxGP(ApproximateGP, GPyTorchModel):
    def __init__(
            self,
            inducing_points,
            likelihood,
            kernel
    ):
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(ApproxGP, self).__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel
        self.likelihood = likelihood

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
