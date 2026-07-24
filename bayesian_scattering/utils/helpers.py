import timm
import torch
import torchvision.transforms as T
import torch.nn as nn
import gpytorch
import pickle

from torch.utils.data import Subset, DataLoader, default_collate
import wilds
from bayesian_scattering.datasets import *
from bayesian_scattering.features import Identity, PCA, TorchImageModel, WaveletScattering
from bayesian_scattering.models import *
from bayesian_scattering.utils.train import PinballLoss, train_exact_gp, train_approx_gp, train_mlp
from bayesian_scattering.utils.math import average_distance
from bayesian_scattering.utils.transforms import Grayscale


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
        if kwargs["split"]:
            trainset, testset = None, None
        else:
            trainset = QM2(store_path=store_path, **kwargs).to(device)
            testset = None
        return trainset, testset
    elif dataset_name == "qm9":
        fullset = QM9(store_path=store_path, **kwargs).to(device)
        if kwargs["split"]:
            atoms = dict(H=1, C=6, O=8, N=7, S=16, F=9)
            atom_mask = torch.any(
                fullset.full_charges == atoms[kwargs["shift"]], axis=1
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
        if isinstance(dataset, QM2) or isinstance(dataset, QM9):
            dataset.unimol = False

        transforms = None

        feature = WaveletScattering(
            store_path=store_path, dataset=dataset, transforms=transforms, **kwargs
        ).to(device)
    elif feature_name.startswith("timm"):
        if isinstance(dataset, QM2) or isinstance(dataset, QM9):
            dataset.unimol = False

        transforms = None

        if isinstance(dataset, Pixels):
            # resize to the pretrained model's native input size (e.g. 224 for
            # convnext, 518 for dinov2)
            input_size = timm.get_pretrained_cfg(kwargs["pretrained_model"]).input_size[-1]
            transforms = T.Compose(
                [
                    T.Resize(
                        size=input_size,
                        interpolation=T.InterpolationMode.BICUBIC,
                        max_size=None,
                        antialias=True,
                    ),
                ]
            )
            if dataset.dataset_name == "skin_lesion":
                transforms.transforms.append(
                    T.Normalize(
                        mean=torch.tensor([0.5602, 0.5328, 0.7694]),
                        std=torch.tensor([0.1736, 0.1547, 0.1469]),
                    ),
                )
            if dataset.dataset_name == "histology_nuclei":
                transforms.transforms.append(
                    T.Normalize(
                        mean=torch.tensor([0.7317, 0.5654, 0.7403]),
                        std=torch.tensor([0.1864, 0.2278, 0.1995]),
                    ),
                )

        if isinstance(dataset, WILDS):
            if dataset.dataset.dataset_name == "poverty":
                transforms = T.Compose(
                    [
                        Grayscale(),
                        T.Resize(
                            size=256,
                            interpolation=T.InterpolationMode.BICUBIC,
                            max_size=None,
                            antialias=True,
                        ),
                        T.CenterCrop(224),
                        T.Normalize(
                            mean=torch.tensor([0.4850, 0.4560, 0.4060]),
                            std=torch.tensor([0.2290, 0.2240, 0.2250]),
                        ),
#                        T.Normalize(
#                            mean=torch.tensor([-0.0785, -0.0771, -0.0568, -0.0067, -0.0054, -0.0877, -0.0280, 0.1652]),
#                            std=torch.tensor([0.9741, 0.9600, 0.9612, 0.9829, 0.9944, 0.9725, 0.9967, 1.1939]),
#                        ),
                    ]
                )
                #transforms = None

        feature = TorchImageModel(
            store_path=store_path, dataset=dataset, transforms=transforms, **kwargs
        ).to(device)
    elif feature_name.startswith("unimol"):
        assert isinstance(dataset, QM2) or isinstance(dataset, QM9), (
            "Unimol features are only compatible with QM9 dataset"
        )
        from bayesian_scattering.features import Unimol as feature_cls

        dataset.unimol = True
        feature = feature_cls(store_path=store_path, dataset=dataset, **kwargs)
    elif feature_name.startswith("pca"):
        feature = PCA(store_path=store_path, dataset=dataset, **kwargs).to(device)
    elif feature_name.startswith("identity"):
        feature = Identity(dataset=dataset, **kwargs).to(device)
    else:
        raise ValueError("Feature not available")

    if indices is not None:
        feature = Subset(dataset=feature, indices=indices)

    return feature


def get_model(model_name, data=None, device=torch.device("cpu"), **kwargs):
    if model_name.startswith("gp_reg"):
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
    elif model_name.startswith("svgp_reg"):
        assert data is not None
        inducing_points, _ = next(
            iter(DataLoader(data, batch_size=kwargs.get("num_ip"), shuffle=True))
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
    elif model_name.startswith("headens_reg"):
        assert data is not None, "Missing data info"
        data_dim = data[0][0].shape[0]
        layers_func = dict(
            tanh=nn.Tanh(),
            relu=nn.ReLU(),
            gelu=nn.GELU(),
            layer_norm=nn.LayerNorm(normalized_shape=data_dim),
        )
        model = Ensemble(
            model=GaussianMLP,
            dim=kwargs["num_models"] if "num_models" in kwargs else 10,
            is_probabilistic_regression=True,
            layers=[data_dim] + (kwargs["layers"] if "layers" in kwargs else []) + [2],
            activation=layers_func[kwargs["activation"]]
            if "activation" in kwargs
            else None,
            features=layers_func[kwargs["features"]] if "features" in kwargs else None,
        ).to(device)
    elif model_name.startswith("fullens_reg"):
        assert data is not None, "Missing data info"
        num_ch = data[0][0].shape[0]
        layers_func = dict(
            tanh=nn.Tanh(),
            relu=nn.ReLU(),
            gelu=nn.GELU(),
        )
        model = Ensemble(
            model=TIMMGaussianRegression,
            num_ch=num_ch,
            features_model=kwargs["features_model"],
            dim=kwargs["num_models"] if "num_models" in kwargs else 10,
            is_probabilistic_regression=True,
            layers=(kwargs["layers"] if "layers" in kwargs else []) + [2],
            activation=layers_func[kwargs["activation"]]
            if "activation" in kwargs
            else None,
            features=layers_func[kwargs["features"]] if "features" in kwargs else None,
        )  # .to(device)
    elif model_name.startswith("cpmlp_reg"):
        assert data is not None, "Missing data info"
        data_dim = data[0][0].shape[0]
        layers_func = dict(
            tanh=nn.Tanh(),
            relu=nn.ReLU(),
            gelu=nn.GELU(),
            layer_norm=nn.LayerNorm(normalized_shape=data_dim),
        )
        model = ConformalMLP(
            layers=[data_dim] + (kwargs["layers"] if "layers" in kwargs else []) + [1],
            activation=layers_func[kwargs["activation"]]
            if "activation" in kwargs
            else None,
            features=layers_func[kwargs["features"]] if "features" in kwargs else None,
        ).to(device)
    elif model_name.startswith("cqr_reg"):
        assert data is not None, "Missing data info"
        data_dim = data[0][0].shape[0]
        layers_func = dict(
            tanh=nn.Tanh(),
            relu=nn.ReLU(),
            gelu=nn.GELU(),
            layer_norm=nn.LayerNorm(normalized_shape=data_dim),
        )
        model = CQRMLP(
            layers=[data_dim] + (kwargs["layers"] if "layers" in kwargs else []),
            activation=layers_func[kwargs["activation"]]
            if "activation" in kwargs
            else None,
            features=layers_func[kwargs["features"]] if "features" in kwargs else None,
        ).to(device)
    elif model_name.startswith("la_reg"):
        assert data is not None, "Missing training data."
        layers = kwargs.get("layers")
        if layers is None:
            raise ValueError("Missing 'layers' for Laplace model.")
        if not isinstance(layers, (list, tuple)):
            raise ValueError("'layers' must be a list of hidden sizes.")
        layers_func = dict(tanh=nn.Tanh(), relu=nn.ReLU(), gelu=nn.GELU())

        sample_x, _ = next(iter(DataLoader(data, batch_size=1)))
        input_dim = sample_x.shape[-1] if sample_x.ndim > 1 else 1
        mlp_layers = [input_dim, *list(layers), 1]

        base_model = MLP(
            layers=mlp_layers,
            activation=layers_func[kwargs["activation"]]
            if "activation" in kwargs
            else None,
        ).to(device)
        model = LaplaceApproximation(
            base_model,
            subset_of_weights=kwargs.get("subset_of_weights", "last_layer"),
            hessian_structure=kwargs.get("hessian_structure", "diag"),
            noise_var=kwargs.get("noise_var", 1e-2),
            prior_precision=kwargs.get("prior_precision"),
            pred_type=kwargs.get("pred_type", "glm"),
        )
    elif model_name.startswith("base_reg"):
        assert data is not None, "Missing training data."
        train_y = next(iter(DataLoader(data, batch_size=len(data))))[1]
        train_y = train_y.to(device)
        model = Baseline(labels=train_y)
    elif model_name.startswith("svgp_class"):
        assert data is not None, "Missing training data."

        num_classes = (
            data.base.dataset.n_classes
            if hasattr(data, "base")
            else data.dataset.base.dataset.n_classes
        )
        inducing_points, _ = next(
            iter(DataLoader(data, batch_size=kwargs.get("num_ip"), shuffle=True))
        )

        kargs = kwargs.get("kernel").copy()
        kargs["batch_shape"] = torch.Size([num_classes])
        if "ard_num_dims" in kargs and kargs["ard_num_dims"] == "auto":
            kargs["ard_num_dims"] = inducing_points.shape[1]
        if "lengthscale" in kargs and kargs["lengthscale"] == "auto":
            kargs["lengthscale"] = average_distance(inducing_points)
        kernel = gpytorch.kernels.ScaleKernel(
            get_kernel(id=kargs.pop("id"), **kargs), batch_shape=kargs["batch_shape"]
        )

        if num_classes > 2:
            likelihood = gpytorch.likelihoods.SoftmaxLikelihood(
                num_classes=num_classes,
                mixing_weights=False,  # treat each latent GP as a class logit
            )
        else:
            likelihood = gpytorch.likelihoods.BernoulliLikelihood()

        model = MultitaskApproxGP(inducing_points, num_classes, likelihood, kernel).to(
            device
        )

        hypers = {
            "covar_module.outputscale": 1.0,
        }
        model.initialize(**hypers)
    elif model_name.startswith("headens_class"):
        assert data is not None, "Missing data info"
        data_dim = data[0][0].shape[0]
        num_classes = (
            data.base.dataset.n_classes
            if hasattr(data, "base")
            else data.dataset.base.dataset.n_classes
        )
        layers_func = dict(
            tanh=nn.Tanh(),
            relu=nn.ReLU(),
            gelu=nn.GELU(),
            layer_norm=nn.LayerNorm(normalized_shape=data_dim),
        )
        model = Ensemble(
            model=MLP,
            dim=kwargs["num_models"] if "num_models" in kwargs else 10,
            is_class=True,
            layers=[data_dim]
            + (kwargs["layers"] if "layers" in kwargs else [])
            + [num_classes],
            activation=layers_func[kwargs["activation"]]
            if "activation" in kwargs
            else None,
            features=layers_func[kwargs["features"]] if "features" in kwargs else None,
        ).to(device)
    elif model_name.startswith("fullens_class"):
        assert data is not None, "Missing data info"
        num_ch = data[0][0].shape[0]
        num_classes = (
            data.base.dataset.n_classes
            if hasattr(data, "base")
            else data.dataset.base.dataset.n_classes
        )
        model = Ensemble(
            model=TIMMClassification,
            num_ch=num_ch,
            num_classes=num_classes,
            features_model=kwargs["features_model"],
            dim=kwargs["num_models"] if "num_models" in kwargs else 10,
            is_class=True,
        )
    elif model_name.startswith("la_class"):
        assert data is not None, "Missing training data."
        layers = kwargs.get("layers")
        if layers is None:
            raise ValueError("Missing 'layers' for Laplace model.")
        if not isinstance(layers, (list, tuple)):
            raise ValueError("'layers' must be a list of hidden sizes.")
        layers_func = dict(tanh=nn.Tanh(), relu=nn.ReLU(), gelu=nn.GELU())

        sample_x, _ = next(iter(DataLoader(data, batch_size=1)))
        input_dim = sample_x.shape[-1] if sample_x.ndim > 1 else 1
        num_classes = (
            data.base.dataset.n_classes
            if hasattr(data, "base")
            else data.dataset.base.dataset.n_classes
        )
        mlp_layers = [input_dim, *list(layers), num_classes]

        base_model = MLP(
            layers=mlp_layers,
            activation=layers_func[kwargs["activation"]]
            if "activation" in kwargs
            else None,
        ).to(device)
        model = LaplaceApproximation(
            base_model,
            likelihood="classification",
            subset_of_weights=kwargs.get("subset_of_weights", "last_layer"),
            hessian_structure=kwargs.get("hessian_structure", "diag"),
            prior_precision=kwargs.get("prior_precision"),
            pred_type=kwargs.get("pred_type", "glm"),
            link_approx=kwargs.get("link_approx", "mc"),
            n_samples=kwargs.get("n_samples", 100),
            posterior_jitter=kwargs.get("posterior_jitter", 1e-6),
        )
    elif model_name.startswith("base_class"):
        assert data is not None, "Missing training data."
        train_y = next(iter(DataLoader(data, batch_size=len(data))))[1]
        train_y = train_y.to(device)
        num_classes = (
            data.base.dataset.n_classes
            if hasattr(data, "base")
            else data.dataset.base.dataset.n_classes
        )
        model = Baseline(labels=train_y, is_class=True, num_classes=num_classes)
    else:
        raise ValueError("Model not available.")

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
            weight_decay=0.0,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **cfg["scheduler"]
        )
        loss = train_exact_gp(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            **cfg["train"],
            **cfg["gpytorch"],
        )
    else:
        train_loader = DataLoader(data, **cfg["dataloader"])
        if isinstance(model, ApproxGP) or isinstance(model, MultitaskApproxGP):
            optimizer = torch.optim.Adam(
                model.parameters(),
                # **cfg["optimizer"],
                lr=1.0e-1,
                weight_decay=0.0,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **cfg["scheduler"]
            )
            loss = train_approx_gp(
                model=model,
                data_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                **cfg["train"],
                **cfg["gpytorch"],
            )
        elif isinstance(model, LaplaceApproximation):
            optimizer = torch.optim.Adam(model.model.parameters(), **cfg["optimizer"])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **cfg["scheduler"]
            )
            loss = train_mlp(
                model=model.model,
                loss_fn=nn.CrossEntropyLoss() if model.is_class else nn.MSELoss(),
                data_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                **cfg["train"],
            )
            fit_loader_kwargs = dict(cfg["dataloader"])
            fit_loader_kwargs["shuffle"] = False
            if fit_loader_kwargs.get("num_workers", 0) == 0:
                fit_loader_kwargs.pop("prefetch_factor", None)
                fit_loader_kwargs.pop("persistent_workers", None)
            if not model.is_class:
                fit_loader_kwargs["collate_fn"] = lambda batch: (
                    default_collate([sample[0] for sample in batch]),
                    default_collate([sample[1] for sample in batch]).unsqueeze(-1),
                )
            fit_loader = DataLoader(data, **fit_loader_kwargs)
            model.fit(
                fit_loader,
                progress_bar=cfg["train"].get("verbose", False),
            )
        elif isinstance(model, (ConformalMLP, CQRMLP)):
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=1.0e-4,
                weight_decay=0.0,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **cfg["scheduler"]
            )
            loss = train_mlp(
                model=model,
                loss_fn=PinballLoss(model.quantiles)
                if isinstance(model, CQRMLP)
                else torch.nn.MSELoss(),
                data_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                **cfg["train"],
            )
            if isinstance(model, ConformalMLP):
                model.fit_noise_var(train_loader)
        elif isinstance(model, Ensemble):
            loss = torch.zeros(len(model.ensemble))
            for count, mlp in enumerate(model.ensemble):
                print(f"Train model {count + 1}/{len(model.ensemble)}")
                optimizer = torch.optim.Adam(
                    mlp.parameters(),
                    # **cfg["optimizer"],
                    lr=1.0e-4,
                    weight_decay=0.0,
                )
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, **cfg["scheduler"]
                )
                mlp.to(device)
                if model.is_class:
                    loss_fn = torch.nn.CrossEntropyLoss()
                elif model.is_probabilistic_regression:
                    loss_fn = None
                else:
                    loss_fn = torch.nn.MSELoss()
                loss[count] = train_mlp(
                    model=mlp,
                    loss_fn=loss_fn,
                    data_loader=train_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    **cfg["train"],
                )
                mlp.cpu()
            loss = loss.mean().item()
        else:
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
        with open(filename_log, "wb") as f:
            pickle.dump(results_log, f)
        print("Saved:", filename_log)

    return df


def get_regrets(x_inds, ys, initial_design_size, n_iters):
    vals = [float(ys[x_inds[:initial_design_size]].max())] * initial_design_size
    for i in range(n_iters):
        vals.append(float(ys[x_inds[: initial_design_size + i]].max()))
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
