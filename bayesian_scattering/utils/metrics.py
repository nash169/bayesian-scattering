import torch
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal


def quantile_coverage_error_signed(
    pred_dist: MultivariateNormal,
    test_y: torch.Tensor,
    quantile: float = 95.0,
):
    if quantile <= 0 or quantile >= 100:
        raise NotImplementedError("Quantile must be between 0 and 100")
    combine_dim = -2 if isinstance(pred_dist, MultitaskMultivariateNormal) else -1
    standard_normal = torch.distributions.Normal(loc=0.0, scale=1.0)
    deviation = standard_normal.icdf(torch.as_tensor(0.5 + 0.5 * (quantile / 100)))
    lower = pred_dist.mean - deviation * pred_dist.stddev
    upper = pred_dist.mean + deviation * pred_dist.stddev
    n_samples_within_bounds = ((test_y > lower) * (test_y < upper)).sum(combine_dim)
    fraction = n_samples_within_bounds / test_y.shape[combine_dim]
    return fraction - quantile / 100


def prediction_interval_length(
    pred_dist: MultivariateNormal,
    test_y: torch.Tensor,
    quantile: float = 95.0,
):
    if quantile <= 0 or quantile >= 100:
        raise NotImplementedError("Quantile must be between 0 and 100")
    combine_dim = -2 if isinstance(pred_dist, MultitaskMultivariateNormal) else -1
    standard_normal = torch.distributions.Normal(loc=0.0, scale=1.0)
    deviation = standard_normal.icdf(torch.as_tensor(0.5 + 0.5 * (quantile / 100)))
    lower = pred_dist.mean - deviation * pred_dist.stddev
    upper = pred_dist.mean + deviation * pred_dist.stddev
    return (upper - lower).mean(), (upper - lower).std()
