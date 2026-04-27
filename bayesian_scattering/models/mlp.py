import gpytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional


class MLP(nn.Module):
    def __init__(
        self,
        layers: List,
        activation: Optional[nn.Module] = None,
        features: Optional[nn.Module] = None,
    ):
        super().__init__()

        module_list = nn.ModuleList()
        if features is not None:
            module_list.append(features)
        for i, _ in enumerate(layers[:-2]):
            module_list.append(nn.Linear(layers[i], layers[i + 1]))
            if activation is not None:
                module_list.append(activation)
        module_list.append(nn.Linear(layers[-2], layers[-1]))

        self.net = nn.Sequential(*(module_list[i] for i in range(len(module_list))))

    def forward(self, x):
        return self.net(x)


class GaussianMLP(nn.Module):
    def __init__(
        self,
        layers: List,
        activation: Optional[nn.Module] = None,
        features: Optional[nn.Module] = None,
        min_variance: float = 1e-6,
    ):
        super().__init__()
        self.model = MLP(layers=layers, activation=activation, features=features)
        self.min_variance = min_variance

    def moments(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.model(x)
        if output.ndim == 1:
            output = output.unsqueeze(0)
        if output.shape[-1] != 2:
            raise ValueError("GaussianMLP expects exactly two outputs per sample.")

        mean = output[..., 0]
        variance = F.softplus(output[..., 1]) + self.min_variance
        return mean, variance

    def posterior(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean, variance = self.moments(x)
        return gpytorch.distributions.MultivariateNormal(
            mean, torch.diag_embed(variance)
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return self.posterior(x)
