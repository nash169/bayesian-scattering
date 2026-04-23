import torch
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal


def _classification_probabilities(prediction, eps: float = 1e-12) -> torch.Tensor:
    probs = prediction.probs if hasattr(prediction, "probs") else prediction
    probs = torch.as_tensor(probs)

    if probs.ndim == 0:
        raise ValueError("Classification predictions must have at least one dimension.")

    if probs.ndim >= 3:
        probs = probs.mean(dim=0)

    if probs.ndim == 1:
        if probs.numel() > 1 and torch.isclose(
            probs.sum(),
            torch.tensor(1.0, device=probs.device, dtype=probs.dtype),
            atol=1e-4,
            rtol=1e-4,
        ):
            probs = probs.unsqueeze(0)
        else:
            probs = torch.stack((1.0 - probs, probs), dim=-1)
    elif probs.ndim == 2:
        row_sums = probs.sum(dim=-1)
        if not torch.allclose(
            row_sums,
            torch.ones_like(row_sums),
            atol=1e-4,
            rtol=1e-4,
        ):
            probs = probs.mean(dim=0)
            if probs.ndim == 0:
                probs = probs.reshape(1)
            probs = torch.stack((1.0 - probs, probs), dim=-1)
        elif probs.shape[-1] == 1:
            probs = torch.cat((1.0 - probs, probs), dim=-1)
    else:
        raise ValueError("Unsupported classification probability shape.")

    probs = probs.clamp(min=eps)
    return probs / probs.sum(dim=-1, keepdim=True).clamp(min=eps)


def _classification_targets(target: torch.Tensor) -> torch.Tensor:
    target = torch.as_tensor(target).long().reshape(-1)
    if target.ndim != 1:
        raise ValueError(
            "Classification targets must be one-dimensional after reshape."
        )
    return target


def accuracy(prediction, target: torch.Tensor) -> torch.Tensor:
    probs = _classification_probabilities(prediction)
    target = _classification_targets(target)
    return (probs.argmax(dim=-1) == target).float().mean()


def f1(prediction, target: torch.Tensor, average: str = "macro") -> torch.Tensor:
    if average != "macro":
        raise ValueError("Only macro F1 is implemented.")

    probs = _classification_probabilities(prediction)
    target = _classification_targets(target)
    pred = probs.argmax(dim=-1)
    num_classes = probs.shape[-1]

    scores = []
    for class_idx in range(num_classes):
        pred_mask = pred == class_idx
        target_mask = target == class_idx
        support = target_mask.sum()
        if support == 0:
            continue

        true_positive = (pred_mask & target_mask).sum().float()
        false_positive = (pred_mask & ~target_mask).sum().float()
        false_negative = (~pred_mask & target_mask).sum().float()

        precision = true_positive / (true_positive + false_positive).clamp(min=1.0)
        recall = true_positive / (true_positive + false_negative).clamp(min=1.0)
        scores.append(2.0 * precision * recall / (precision + recall).clamp(min=1e-12))

    if not scores:
        return torch.zeros((), dtype=probs.dtype, device=probs.device)
    return torch.stack(scores).mean()


def nll(prediction, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    probs = _classification_probabilities(prediction, eps=eps)
    target = _classification_targets(target)
    target_probs = probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    return -torch.log(target_probs.clamp(min=eps)).mean()


def ece(prediction, target: torch.Tensor, n_bins: int = 15) -> torch.Tensor:
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")

    probs = _classification_probabilities(prediction)
    target = _classification_targets(target)
    confidence, pred = probs.max(dim=-1)
    correctness = (pred == target).float()
    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)
    calibration_error = torch.zeros((), dtype=probs.dtype, device=probs.device)

    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if lower == 0:
            mask = (confidence >= lower) & (confidence <= upper)
        if not mask.any():
            continue

        bin_fraction = mask.float().mean()
        avg_confidence = confidence[mask].mean()
        avg_accuracy = correctness[mask].mean()
        calibration_error += bin_fraction * (avg_accuracy - avg_confidence).abs()

    return calibration_error


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
