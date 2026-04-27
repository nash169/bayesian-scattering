import torch
import gpytorch
import gpytorch.settings as cfg
import tqdm

from torch.utils.data import DataLoader
from torch.nn import Module
from typing import Callable, Optional
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def _compute_mlp_loss(
    loss_fn: Callable, y_pred: torch.Tensor, y_true: torch.Tensor
) -> torch.Tensor:
    if isinstance(y_pred, gpytorch.distributions.MultivariateNormal):
        target = y_true.reshape(y_pred.mean.shape)
        return -y_pred.log_prob(target) / target.numel()
    if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
        return loss_fn(y_pred, y_true.long().reshape(-1))
    return loss_fn(y_pred, y_true.reshape(y_pred.shape))


def train_exact_gp(
    model,
    optimizer,
    scheduler=None,
    epochs=100,
    tol=1e-2,
    patience=10,
    verbose=False,
    **kwargs,
):
    model.train()
    model.likelihood.train()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    prev_loss = 1e6
    count_patience = 0

    epochs_iter = tqdm.tqdm(range(epochs), desc="Epoch", disable=not verbose)
    for i in epochs_iter:
        optimizer.zero_grad()
        output = model(model.train_inputs[0])

        with (
            cfg.max_cholesky_size(kwargs["max_cholesky_size"]),
            cfg.cg_tolerance(kwargs["cg_tolerance"]),
            cfg.max_cg_iterations(kwargs["max_cg_iterations"]),
            cfg.fast_computations(**kwargs["fast_computations"]),
        ):
            loss = -mll(output, model.train_targets)

        if verbose:
            epochs_iter.set_postfix(
                {
                    "Loss": loss.item(),
                    "Lr": scheduler.get_last_lr()[0]
                    if scheduler is not None
                    else optimizer.param_groups[0]["lr"],
                }
            )

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step(loss.detach())

        if abs(loss.item() - prev_loss) <= tol:
            count_patience += 1
            if count_patience == patience:
                break
        else:
            count_patience = 0
        prev_loss = loss.item()

    # gc.collect()
    # torch.cuda.empty_cache()

    return loss.item()


def train_approx_gp(
    model,
    data_loader,
    optimizer,
    scheduler=None,
    epochs=100,
    tol=1e-2,
    patience=10,
    verbose=False,
    **kwargs,
):
    device = next(model.parameters()).device
    model.train()
    model.likelihood.train()
    mll = gpytorch.mlls.VariationalELBO(
        model.likelihood, model, num_data=len(data_loader.dataset)
    )

    curr_loss = 1e6
    count_patience = 0
    epochs_iter = tqdm.tqdm(range(epochs), desc="Epoch", disable=not verbose)

    if data_loader.batch_size >= len(data_loader.dataset):
        x_train, y_train = next(iter(data_loader))
        x_train, y_train = x_train.to(device), y_train.to(device)
        for epoch in epochs_iter:
            optimizer.zero_grad()
            output = model(x_train)

            with (
                cfg.max_cholesky_size(kwargs["max_cholesky_size"]),
                cfg.cg_tolerance(kwargs["cg_tolerance"]),
                cfg.max_cg_iterations(kwargs["max_cg_iterations"]),
                cfg.fast_computations(**kwargs["fast_computations"]),
            ):
                loss = -mll(output, y_train.squeeze())

            if verbose:
                epochs_iter.set_postfix(
                    {
                        "Loss": loss.item(),
                        "Lr": scheduler.get_last_lr()[0]
                        if scheduler is not None
                        else optimizer.param_groups[0]["lr"],
                    }
                )
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step(loss.detach())
            if abs(loss.item() - curr_loss) <= tol:
                count_patience += 1
                if count_patience == patience:
                    break
            else:
                count_patience = 0
            curr_loss = loss.item()
    else:
        for i in epochs_iter:
            minibatch_iter = tqdm.tqdm(
                data_loader,
                desc="Minibatch",
                leave=False,
                disable=not verbose,
            )
            for x_batch, y_batch in minibatch_iter:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                output = model(x_batch)

                with (
                    cfg.max_cholesky_size(kwargs["max_cholesky_size"]),
                    cfg.cg_tolerance(kwargs["cg_tolerance"]),
                    cfg.max_cg_iterations(kwargs["max_cg_iterations"]),
                    cfg.fast_computations(**kwargs["fast_computations"]),
                ):
                    loss = -mll(output, y_batch.squeeze())

                if verbose:
                    epochs_iter.set_postfix(
                        {
                            "Loss": loss.item(),
                            "Lr": scheduler.get_last_lr()[0]
                            if scheduler is not None
                            else optimizer.param_groups[0]["lr"],
                        }
                    )
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step(loss.detach())

            if abs(loss.item() - curr_loss) <= tol:
                count_patience += 1
                if count_patience == patience:
                    break
            else:
                count_patience = 0
            curr_loss = loss.item()
    return curr_loss


def train_mlp(
    model: Module,
    loss_fn: Callable,
    data_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: Optional[LRScheduler] = None,
    epochs: int = 100,
    tol: float = 1e-2,
    patience: int = 10,
    verbose: bool = False,
    **kwargs,
):
    device = next(model.parameters()).device
    model.train()
    epochs_iter = tqdm.tqdm(range(epochs), desc="Epoch", disable=not verbose)

    curr_loss = 1e6
    count_patience = 0

    if data_loader.batch_size >= len(data_loader.dataset):
        x_train, y_train = next(iter(data_loader))
        x_train, y_train = x_train.to(device), y_train.to(device)
        for epoch in epochs_iter:
            optimizer.zero_grad()
            y_pred = model(x_train)
            loss = _compute_mlp_loss(loss_fn, y_pred, y_train)
            if verbose:
                epochs_iter.set_postfix(
                    {
                        "Loss": loss.item(),
                        "Lr": scheduler.get_last_lr()[0]
                        if scheduler is not None
                        else optimizer.param_groups[0]["lr"],
                    }
                )
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step(loss.detach())
            if abs(loss.item() - curr_loss) <= tol:
                count_patience += 1
                if count_patience == patience:
                    break
            else:
                count_patience = 0
            curr_loss = loss.item()
    else:
        for epoch in epochs_iter:
            minibatch_iter = tqdm.tqdm(
                data_loader,
                desc="Minibatch",
                leave=False,
                disable=not verbose,
            )
            for x_batch, y_batch in minibatch_iter:
                optimizer.zero_grad()
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = model(x_batch)
                loss = _compute_mlp_loss(loss_fn, y_pred, y_batch)
                if verbose:
                    epochs_iter.set_postfix(loss=loss.item())
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step(loss.detach())
            if abs(loss.item() - curr_loss) <= tol:
                count_patience += 1
                if count_patience == patience:
                    break
            else:
                count_patience = 0
            curr_loss = loss.item()
    return curr_loss
