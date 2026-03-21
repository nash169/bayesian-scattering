from numpy import full
import torch
import torch.nn as nn
import gpytorch
import pickle

from torch.utils.data import Subset, DataLoader
from torchvision.transforms import Resize
from bayesian_scattering import features
from bayesian_scattering.datasets import *
from bayesian_scattering.features import *
from bayesian_scattering.models import *
from bayesian_scattering.utils.train import train_exact_gp, train_approx_gp, train_mlp
from bayesian_scattering.utils.math import average_distance


def get_dataset(dataset_name, store_path, device=torch.device("cpu"), **kwargs):
    assert dataset_name in [
        "qm2",
        "qm9",
        "poverty",
        "iwildcam",
        "skin_lesion",
        "histology_nuclei",
    ], "Dataset not available"

    if dataset_name == "qm2":
        if kwargs['split']:
            trainset, testset = None, None
        else:
            trainset = QM2(
                store_path=store_path,
                **kwargs
            ).to(device)
            testset = None
        return trainset, testset
    elif dataset_name == "qm9":
        fullset = QM9(
            store_path=store_path,
            **kwargs
        ).to(device)
        if kwargs['split']:
            atoms = dict(H=1, C=6, O=8, N=7, S=16, F=9)
            atom_mask = torch.any(
                fullset.full_charges == atoms[kwargs['shift']],
                axis=1
            )
            trainset = Subset(fullset, torch.nonzero(~atom_mask))
            testset = Subset(fullset, torch.nonzero(~atom_mask))
        else:
            trainset = fullset
            testset = None
        return trainset, testset
    elif dataset_name in ["poverty", "iwildcam"]:
        trainset = WILDS(
            dataset_name=dataset_name, split="train", dataset_path=store_path, **kwargs
        ).to(device)
        testset = WILDS(
            dataset_name=dataset_name, split="test", dataset_path=store_path, **kwargs
        ).to(device)
        return trainset, testset
    elif dataset_name in ["skin_lesion", "histology_nuclei"]:
        trainset = Pixels(
            dataset_name=dataset_name, split="train", dataset_path=store_path, **kwargs
        ).to(device)
        testset = Pixels(
            dataset_name=dataset_name, split="test", dataset_path=store_path, **kwargs
        ).to(device)
        return trainset, testset


def get_feature(
    feature_name,
    store_path,
    dataset,
    indices=None,
    device=torch.device("cpu"),
    **kwargs,
):
    if feature_name.startswith("scattering"):
        feature = WaveletScattering(
            store_path=store_path,
            dataset=dataset,
            transform=Resize((256, 256)) if isinstance(dataset, WILDS) else None,
            **kwargs
        ).to(device)
        if isinstance(dataset, QM2) or isinstance(dataset, QM9):
            dataset.unimol = False
    elif feature_name.startswith("timm"):
        feature = TorchImageModel(
            store_path=store_path, dataset=dataset, **kwargs
        ).to(device)
        if isinstance(dataset, QM2) or isinstance(dataset, QM9):
            dataset.unimol = False
    elif feature_name.startswith("unimol"):
        assert isinstance(dataset, QM2) or isinstance(dataset, QM9), "Unimol features are only compatible with QM9 dataset"
        dataset.unimol = True
        feature = Unimol(store_path=store_path, dataset=dataset, **kwargs)
    elif feature_name.startswith("identity"):
        feature = Identity(
            dataset=dataset, **kwargs
        ).to(device)
    else:
        raise ValueError("Feature not available")

    if indices is not None:
        feature = Subset(
            dataset=feature, indices=indices
        )

    return feature


def get_model(model_name, data=None, device=torch.device("cpu"), **kwargs):
    if model_name.startswith("gp_exact_reg"):
        assert data is not None, "Missing training data."
        assert "kernel" in kwargs, "Missing kernel config."

        train_x, train_y = next(
            iter(
                DataLoader(
                    data,
                    batch_size=len(data),
                    # num_workers=8,
                    # persistent_workers=True,
                    # prefetch_factor=4,
                    # pin_memory=True,
                )
            )
        )

        train_x, train_y = train_x.to(device), train_y.to(device)

        kargs = kwargs.get("kernel").copy()
        if "ard_num_dims" in kargs and kargs["ard_num_dims"] == "auto":
            kargs["ard_num_dims"] = train_x.shape[1]
        if "lengthscale" in kargs and kargs["lengthscale"] == "auto":
            kargs["lengthscale"] = average_distance(train_x)

        model = ExactGP(
            train_x,
            train_y.squeeze(),
            gpytorch.likelihoods.GaussianLikelihood(),
            gpytorch.kernels.ScaleKernel(
                get_kernel(
                    id=kargs.pop("id"),
                    **kargs,
                    # id=kwargs.get('kernel')['id'],
                    # **{k: v for k, v in kwargs.get('kernel').items() if k != "id"},
                )
            ),
        ).to(device)
        hypers = {
            "likelihood.noise_covar.noise": 1e-2,
            "covar_module.outputscale": 1.0,
        }
        model.initialize(**hypers)
    elif model_name.startswith("gp_approx_reg"):
        assert data is not None
        inducing_points, _ = next(
            iter(DataLoader(data, batch_size=kwargs.get("num_ip")))
        )

        kargs = kwargs.get("kernel").copy()
        if "ard_num_dims" in kargs and kargs["ard_num_dims"] == "auto":
            kargs["ard_num_dims"] = inducing_points.shape[1]
        if "lengthscale" in kargs and kargs["lengthscale"] == "auto":
            kargs["lengthscale"] = average_distance(inducing_points)

        model = ApproxGP(
            inducing_points,
            gpytorch.likelihoods.GaussianLikelihood(),
            gpytorch.kernels.ScaleKernel(
                get_kernel(
                    id=kargs.pop("id"),
                    **kargs,
                )
            ),
        ).to(device)
        hypers = {
            "likelihood.noise_covar.noise": 1e-2,
            "covar_module.outputscale": 1.0,
        }
        model.initialize(**hypers)
    elif model_name.startswith("ensemble_reg"):
        assert data is not None, "Missing data info"
        data_dim = data[0][0].shape[0]
        layers_func = dict(
            tanh=nn.Tanh(),
            relu=nn.ReLU(),
            gelu=nn.GELU(),
            layer_norm=nn.LayerNorm(normalized_shape=data_dim)
        )
        model = Ensemble(
            model=MLP,
            dim=kwargs["num_models"] if "num_models" in kwargs else 10,
            layers=[data_dim] + (kwargs["layers"] if "layers" in kwargs else []) + [1],
            activation=layers_func[kwargs["activation"]] if "activation" in kwargs else None,
            features=layers_func[kwargs["features"]] if "features" in kwargs else None,
        ).to(device)
    elif model_name.startswith("ensemble_timm_reg"):
        assert data is not None, "Missing data info"
        num_ch = data[0][0].shape[0]
        layers_func = dict(
            tanh=nn.Tanh(),
            relu=nn.ReLU(),
            gelu=nn.GELU(),
        )
        model = Ensemble(
            model=TIMMRegression,
            num_ch=num_ch,
            features_model=kwargs["features_model"],
            dim=kwargs["num_models"] if "num_models" in kwargs else 10,
            layers=(kwargs["layers"] if "layers" in kwargs else []) + [1],
            activation=layers_func[kwargs["activation"]] if "activation" in kwargs else None,
            features=layers_func[kwargs["features"]] if "features" in kwargs else None,
        )  # .to(device)
    elif model_name.startswith("laplace"):
        assert data is not None, "Missing training data."
        layers = kwargs.get("layers")
        if layers is None:
            raise ValueError("Missing 'layers' for Laplace model.")
        if not isinstance(layers, (list, tuple)):
            raise ValueError("'layers' must be a list of hidden sizes.")

        sample_x, _ = next(iter(DataLoader(data, batch_size=1)))
        input_dim = sample_x.shape[-1] if sample_x.ndim > 1 else 1
        mlp_layers = [input_dim, *list(layers), 1]

        base_model = MLP(layers=mlp_layers).to(device)
        model = LaplaceRegressor(
            base_model,
            subset_of_weights=kwargs.get("subset_of_weights", "last_layer"),
            hessian_structure=kwargs.get("hessian_structure", "diag"),
            noise_var=kwargs.get("noise_var", 1e-2),
            prior_precision=kwargs.get("prior_precision"),
            pred_type=kwargs.get("pred_type", "glm"),
        )
    elif model_name.startswith("baseline"):
        assert data is not None, "Missing training data."
        train_y = next(iter(DataLoader(data, batch_size=len(data))))[1]
        train_y = train_y.to(device)
        model = Baseline(labels=train_y)

    return model


def get_kernel(id, **kwargs):
    assert id in ["linear", "rbf", "matern"], "Kernel not supported"

    if id == "rbf":
        kernel = gpytorch.kernels.RBFKernel(**kwargs)
    elif id == "linear":
        kernel = gpytorch.kernels.LinearKernel(**kwargs)
    elif id == "matern":
        kernel = gpytorch.kernels.MaternKernel(**kwargs)
    else:
        raise ValueError("Kernel not available")

    if "lengthscale" in kwargs:
        kernel.lengthscale = kwargs["lengthscale"]

    return kernel


def train_model(model, data, cfg, device):
    if isinstance(model, ExactGP):
        optimizer = torch.optim.Adam(
            model.parameters(),
            # **cfg["optimizer"],
            lr=1.0e-1,
            weight_decay=0.0
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **cfg["scheduler"]
        )
        loss = train_exact_gp(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            **cfg["train"],
            **cfg["gpytorch"]
        )
    else:
        train_loader = DataLoader(
            data,
            **cfg["dataloader"]
        )
        if isinstance(model, ApproxGP):
            optimizer = torch.optim.Adam(
                model.parameters(),
                # **cfg["optimizer"],
                lr=1.0e-1,
                weight_decay=0.0
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                **cfg["scheduler"]
            )
            loss = train_approx_gp(
                model=model,
                data_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                **cfg["train"],
                **cfg["gpytorch"]
            )
        elif isinstance(model, Ensemble):
            loss = torch.zeros(len(model.ensemble))
            for count, mlp in enumerate(model.ensemble):
                print(f"Train model {count + 1}/{len(model.ensemble)}")
                optimizer = torch.optim.Adam(
                    mlp.parameters(),
                    # **cfg["optimizer"],
                    lr=1.0e-4,
                    weight_decay=0.0
                )
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    **cfg["scheduler"]
                )
                mlp.to(device)
                loss[count] = train_mlp(
                    model=mlp,
                    loss_fn=torch.nn.MSELoss(),
                    data_loader=train_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    **cfg["train"],
                )
                mlp.cpu()
            loss = loss.mean().item()
        else:
            # optimizer = torch.optim.Adam(
            #     model.model.parameters(),
            #     lr=cfg["train"].get("laplace_lr", 1e-3),
            #     weight_decay=0.0,
            # )
            # loss_fn = nn.MSELoss()
            # loss = train_mlp(
            #     model.model,
            #     loss_fn,
            #     train_loader,
            #     optimizer,
            #     epochs=cfg["train"]["epochs"],
            #     tol=cfg["train"]["tol"],
            #     patience=cfg["train"]["patience"],
            #     verbose=False,
            # )
            # fit_loader = DataLoader(
            #     f_train,
            #     batch_size=cfg["train"]["batch_size"],
            #     shuffle=False,
            #     num_workers=0,
            #     pin_memory=True,
            # )
            # model.fit(fit_loader)
            raise ValueError("No training available for the current model")
    return loss


def get_results(
    results_log,
    store_path=None,
):
    import numpy as np
    import pandas as pd
    from datetime import datetime

    rows = {}  # rows[metric_name] = {model_name: value}

    for model_name, repetitions in results_log.items():
        # assume all repetitions have the same metric keys
        for key in repetitions[0].keys():
            values = [r[key] for r in repetitions]

            # ---- Case 1: scalar metric ----
            if isinstance(values[0], (int, float)):
                arr = np.array(values, dtype=float)
                mean_val = arr.mean()
                std_val = arr.std(ddof=1)

                # store in rows
                rows.setdefault(f"{key}_mean", {})[model_name] = mean_val
                rows.setdefault(f"{key}_std", {})[model_name] = std_val

            # ---- Case 2: list-valued metric ----
            elif isinstance(values[0], (list, tuple)):
                arr = np.array(values, dtype=float)
                for i in range(arr.shape[1]):
                    comp = arr[:, i]
                    mean_val = comp.mean()
                    std_val = comp.std(ddof=1)

                    rows.setdefault(f"{key}_{i}_mean", {})[model_name] = mean_val
                    rows.setdefault(f"{key}_{i}_std", {})[model_name] = std_val

    # Convert row dictionary to DataFrame
    df = pd.DataFrame.from_dict(rows, orient="index")

    # Sort rows alphabetically for nice readability
    df = df.sort_index()

    # Add timestamp to filename
    if store_path is not None:
        timestamp = datetime.now().strftime("%y_%m_%d_%H_%M")

        filename = str(store_path) + f"_{timestamp}.csv"
        df.to_csv(filename)
        print("Saved:", filename)

        filename_log = str(store_path) + f"_{timestamp}.pkl"
        with open(filename_log, 'wb') as f:
            pickle.dump(results_log, f)
        print("Saved:", filename_log)

    return df


def get_regrets(x_inds, ys, initial_design_size, n_iters):
    vals = [float(ys[x_inds[:initial_design_size]].max())] * initial_design_size
    for i in range(n_iters):
        vals.append(float(ys[x_inds[:initial_design_size + i]].max()))
    return [abs(float(ys.max()) - x) for x in vals]


def get_aucs(regrets_rep):
    import numpy as np
    regrets_rep = np.asarray(regrets_rep)
    aucs = np.mean(regrets_rep, axis=1)
    return aucs


def plot_regrets(regrets_rep, ax, initial_design_size, n_iters, color="blue", label=""):
    import numpy as np
    regrets_rep = np.asarray(regrets_rep)
    mean = np.mean(regrets_rep, axis=0)
    std = np.std(regrets_rep, axis=0)
    bo_plt_xs = np.arange(initial_design_size + n_iters)

    ax.plot(bo_plt_xs, mean, color=color, label=label)
    ax.fill_between(bo_plt_xs, mean - std, mean + std, alpha=0.5, color=color)
