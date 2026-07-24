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


class ConformalMLP(nn.Module):
    """Deterministic MLP with a constant predictive variance fit on the
    training residuals. Prediction intervals are meaningful only after
    post-hoc conformal calibration; with a constant variance the normalized
    residual score reduces to the absolute residual, so the calibrated
    intervals have constant width."""

    def __init__(
        self,
        layers: List,
        activation: Optional[nn.Module] = None,
        features: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.model = MLP(layers=layers, activation=activation, features=features)
        self.register_buffer("noise_var", torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @torch.no_grad()
    def fit_noise_var(self, loader):
        device = next(self.parameters()).device
        sq_err, count = 0.0, 0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            residual = y_batch.squeeze() - self.model(x_batch).squeeze(-1)
            sq_err += residual.pow(2).sum().item()
            count += residual.numel()
        self.noise_var.fill_(max(sq_err / count, 1e-6))

    def posterior(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean = self.model(x).squeeze(-1)
        variance = self.noise_var.expand(mean.shape)
        return gpytorch.distributions.MultivariateNormal(
            mean, torch.diag_embed(variance)
        )


class CQRMLP(nn.Module):
    """MLP with quantile heads for conformalized quantile regression (CQR):
    a median head plus a (lower, upper) quantile pair per coverage level,
    trained jointly with the pinball loss. Intervals are calibrated post hoc
    with the CQR score max(lo - y, y - hi). The Gaussian posterior is an
    approximation (matched to the raw band of the first coverage level) used
    only for the generic Gaussian metrics."""

    def __init__(
        self,
        layers: List,
        activation: Optional[nn.Module] = None,
        features: Optional[nn.Module] = None,
        coverages: tuple = (90.0, 95.0, 99.0),
    ):
        super().__init__()
        self.coverages = tuple(coverages)
        self.quantiles = [0.5]
        self._bounds_idx = {}
        for coverage in self.coverages:
            alpha = 1.0 - coverage / 100.0
            self._bounds_idx[coverage] = (len(self.quantiles), len(self.quantiles) + 1)
            self.quantiles += [alpha / 2.0, 1.0 - alpha / 2.0]
        self.model = MLP(
            layers=list(layers) + [len(self.quantiles)],
            activation=activation,
            features=features,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        return output.unsqueeze(0) if output.ndim == 1 else output

    def bounds(
        self, prediction: torch.Tensor, coverage: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        idx_lo, idx_hi = self._bounds_idx[coverage]
        return prediction[..., idx_lo], prediction[..., idx_hi]

    def posterior(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        prediction = self(x)
        mean = prediction[..., 0]
        lo, hi = self.bounds(prediction, self.coverages[0])
        deviation = torch.distributions.Normal(0.0, 1.0).icdf(
            torch.tensor(0.5 + 0.5 * self.coverages[0] / 100.0)
        )
        variance = ((hi - lo) / (2.0 * deviation)).pow(2).clamp_min(1e-6)
        return gpytorch.distributions.MultivariateNormal(
            mean, torch.diag_embed(variance)
        )


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
