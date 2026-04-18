import torch
import gpytorch

from bayesian_scattering.utils.metrics import quantile_coverage_error_signed, prediction_interval_length
from bayesian_scattering.models import Baseline, Ensemble


def test_regression(
    model,
    loader,
    labels_norm=None,
):
    eval_log = {
        'rmse': 0.0,
        'mae': 0.0,
        'nll': 0.0,
        'qce_90': 0.0,
        'qce_95': 0.0,
        'qce_99': 0.0,
        'pi_90': [0.0, 0.0],
        'pi_95': [0.0, 0.0],
        'pi_99': [0.0, 0.0],
    }
    if isinstance(model, Baseline):
        device = model.labels.device
    elif isinstance(model, Ensemble):
        device = next(model.ensemble[0].parameters()).device
        model.eval()
    else:
        device = next(model.parameters()).device
        model.likelihood.eval()
        model.eval()

    with torch.no_grad():
        for x_batch, y_batch in loader:
            y_batch = y_batch.squeeze()
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            if labels_norm is not None:
                mu_y, std_y = labels_norm
                y_batch.sub_(mu_y).div_(std_y)
            if isinstance(model, (Baseline, Ensemble, LaplaceApproximation)):
                posterior = model.posterior(x_batch)
            else:
                posterior = model.likelihood(model(x_batch))
            eval_log['rmse'] += gpytorch.metrics.mean_squared_error(posterior, y_batch, squared=False).item() * y_batch.numel()
            eval_log['mae'] += gpytorch.metrics.mean_absolute_error(posterior, y_batch).item() * y_batch.numel()
            eval_log['nll'] += gpytorch.metrics.negative_log_predictive_density(posterior, y_batch).item() * y_batch.numel()
            eval_log['qce_90'] += quantile_coverage_error_signed(posterior, y_batch, quantile=90.0).item() * y_batch.numel()
            eval_log['qce_95'] += quantile_coverage_error_signed(posterior, y_batch, quantile=95.0).item() * y_batch.numel()
            eval_log['qce_99'] += quantile_coverage_error_signed(posterior, y_batch, quantile=99.0).item() * y_batch.numel()
            pi_mu, pi_std = prediction_interval_length(posterior, y_batch, quantile=90.0)
            eval_log['pi_90'][0] += pi_mu.item() * y_batch.numel()
            eval_log['pi_90'][1] += pi_std.item() * y_batch.numel()
            pi_mu, pi_std = prediction_interval_length(posterior, y_batch, quantile=95.0)
            eval_log['pi_95'][0] += pi_mu.item() * y_batch.numel()
            eval_log['pi_95'][1] += pi_std.item() * y_batch.numel()
            pi_mu, pi_std = prediction_interval_length(posterior, y_batch, quantile=99.0)
            eval_log['pi_99'][0] += pi_mu.item() * y_batch.numel()
            eval_log['pi_99'][1] += pi_std.item() * y_batch.numel()
        for key, val in eval_log.items():
            if key.startswith("pi_"):
                eval_log[key][0] /= len(loader.dataset)
                eval_log[key][1] /= len(loader.dataset)
            else:
                eval_log[key] /= len(loader.dataset)
        if labels_norm is not None:
            eval_log['rmse'] *= labels_norm[1].item()
            eval_log['mae'] *= labels_norm[1].item()
    return eval_log


def test_classification(
    model,
    loader,
    **kwargs
):
    eval_log = torch.zeros(2)
    device = next(model.parameters()).device
    model.likelihood.eval()
    model.eval()
    mll = gpytorch.mlls.PredictiveLogLikelihood(model.likelihood, model, num_data=len(loader.dataset))

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), x_batch.to(device)
            with gpytorch.settings.num_likelihood_samples(kwargs.get('num_mc_samples')):
                pred = model(x_batch)
                y_pred = model.likelihood(pred).probs.mean(dim=0).argmax(-1)
                eval_log[0] += (y_pred == y_batch).sum().item()
                eval_log[1] -= mll(pred, y_batch).item()
        eval_log /= len(loader.dataset)

    return eval_log
