#!/usr/bin/env python
# encoding: utf-8

from __future__ import annotations

import gpytorch
import torch
import torch.nn as nn
from laplace import Laplace


class LaplaceApproximation(nn.Module):
    def __init__(
        self,
        model,
        dim=None,
        likelihood="regression",
        subset_of_weights="all",
        hessian_structure="full",
        noise_var=1e-2,
        prior_precision=None,
        pred_type="glm",
        link_approx="mc",
        n_samples=100,
        posterior_jitter=1e-6,
        **kwargs,
    ):
        super().__init__()

        self.model = model
        self.dim = dim
        self.likelihood_type = likelihood
        self.is_class = likelihood == "classification"
        self.pred_type = pred_type
        self.link_approx = link_approx
        self.n_samples = n_samples
        self.posterior_jitter = posterior_jitter
        self.likelihood = nn.Identity()

        param = next(self.model.parameters(), None)
        dtype = param.dtype if param is not None else torch.float32
        device = param.device if param is not None else torch.device("cpu")
        if isinstance(noise_var, str):
            noise_var = float(noise_var)
        self.register_buffer(
            "_noise_var", torch.as_tensor(noise_var, dtype=dtype, device=device)
        )

        self._laplace_kwargs = dict(
            subset_of_weights=subset_of_weights,
            hessian_structure=hessian_structure,
            **kwargs,
        )
        if not self.is_class:
            self._laplace_kwargs["sigma_noise"] = self.noise_var.sqrt()
        if prior_precision is not None:
            self._laplace_kwargs["prior_precision"] = prior_precision

        self.la = Laplace(self.model, likelihood, **self._laplace_kwargs)

    @property
    def noise_var(self):
        return self._noise_var

    def _require_laplace(self):
        if getattr(self.la, "n_data", 0) == 0:
            raise RuntimeError(
                "LaplaceApproximation is not fit yet. Train the base model and call `fit(train_loader)` first."
            )
        return self.la

    def _sync_laplace_state(self):
        sigma_noise = self.noise_var.sqrt().to(
            device=self._noise_var.device, dtype=self._noise_var.dtype
        )
        self.la.sigma_noise = sigma_noise

        for name in ["prior_precision", "prior_mean", "mean"]:
            value = getattr(self.la, name, None)
            if isinstance(value, torch.Tensor):
                setattr(
                    self.la,
                    name,
                    value.to(
                        device=self._noise_var.device, dtype=self._noise_var.dtype
                    ),
                )

        H = getattr(self.la, "H", None)
        if H is not None and hasattr(H, "to"):
            self.la.H = H.to(device=self._noise_var.device, dtype=self._noise_var.dtype)

    def _apply(self, fn):
        module = super()._apply(fn)
        module._sync_laplace_state()
        return module

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        module._sync_laplace_state()
        return module

    def fit(self, train_loader, override=True, progress_bar=False):
        self._sync_laplace_state()
        self.la.fit(train_loader, override=override, progress_bar=progress_bar)
        self._sync_laplace_state()
        return self

    def optimize_prior_precision(self, method="marglik", val_loader=None, **kwargs):
        self._require_laplace()

        kwargs.setdefault("pred_type", self.pred_type)
        kwargs.setdefault("link_approx", self.link_approx)
        kwargs.setdefault("n_samples", self.n_samples)
        if val_loader is not None:
            kwargs["val_loader"] = val_loader

        self.la.optimize_prior_precision(method=method, **kwargs)
        self._sync_laplace_state()
        return self

    def forward(self, x, **kwargs):
        posterior = self.posterior(x, **kwargs)
        if self.is_class:
            return posterior
        return posterior.mean.unsqueeze(-1)

    def posterior(self, x, joint=False, pred_type=None, n_samples=None):
        self._require_laplace()
        self._sync_laplace_state()

        pred_type = self.pred_type if pred_type is None else pred_type
        n_samples = self.n_samples if n_samples is None else n_samples
        if joint and pred_type != "glm":
            raise ValueError(
                "Joint predictions are only supported for `pred_type='glm'`."
            )

        predictive_kwargs = {
            "pred_type": pred_type,
            "n_samples": n_samples,
        }
        if pred_type == "glm":
            predictive_kwargs["joint"] = joint
            predictive_kwargs["diagonal_output"] = not joint
        if self.link_approx is not None:
            predictive_kwargs["link_approx"] = self.link_approx

        if self.is_class:
            return self.la(x, **predictive_kwargs)

        mean, variance = self.la(x, **predictive_kwargs)
        mean = self._flatten_scalar_output(mean)

        if joint:
            covariance = self._format_covariance(variance)
        else:
            covariance = torch.diag(self._flatten_scalar_variance(variance))

        covariance = covariance.clone()
        covariance.diagonal().add_(self.noise_var.to(covariance))
        if self.posterior_jitter is not None and self.posterior_jitter > 0:
            covariance.diagonal().add_(self.posterior_jitter)

        return gpytorch.distributions.MultivariateNormal(mean, covariance)

    @staticmethod
    def _flatten_scalar_output(mean):
        if mean.ndim == 0:
            return mean.reshape(1)
        if mean.ndim == 1:
            return mean
        if mean.ndim == 2 and mean.shape[-1] == 1:
            return mean.squeeze(-1)
        raise ValueError(
            "LaplaceApproximation only supports scalar regression outputs."
        )

    @staticmethod
    def _flatten_scalar_variance(variance):
        if variance.ndim == 0:
            return variance.reshape(1)
        if variance.ndim == 1:
            return variance
        if variance.ndim == 2 and variance.shape[-1] == 1:
            return variance.squeeze(-1)
        if variance.ndim == 2 and variance.shape[0] == variance.shape[1]:
            return variance.diagonal()
        if variance.ndim == 3 and variance.shape[-2:] == (1, 1):
            return variance.squeeze(-1).squeeze(-1)
        raise ValueError("Unexpected predictive variance shape for scalar regression.")

    @staticmethod
    def _format_covariance(covariance):
        if covariance.ndim == 0:
            return covariance.reshape(1, 1)
        if covariance.ndim == 1:
            return torch.diag(covariance)
        if covariance.ndim == 2:
            return covariance
        if covariance.ndim == 3 and covariance.shape[-2:] == (1, 1):
            return torch.diag(covariance.squeeze(-1).squeeze(-1))
        raise ValueError(
            "Unexpected predictive covariance shape for scalar regression."
        )


__all__ = ["LaplaceApproximation"]
