import gc
import os
from re import subn
import torch
import random
import yaml

from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from botorch.acquisition import LogExpectedImprovement
from copy import deepcopy
from importlib.resources import files

from bayesian_scattering.utils.test import test_regression
from bayesian_scattering.utils.helpers import get_dataset, get_feature, get_model, get_kernel, train_model, get_results, get_regrets


def benchmark_regression(dataset_id, models, features, cfg, device):
    # configs
    with open(files("configs").joinpath("datasets.yaml")) as f:
        datasets_opts = yaml.load(f, Loader=yaml.FullLoader)

    with open(files("configs").joinpath("features.yaml")) as f:
        features_opts = yaml.load(f, Loader=yaml.FullLoader)

    with open(files("configs").joinpath("models.yaml")) as f:
        models_opts = yaml.load(f, Loader=yaml.FullLoader)

    with open(files("configs").joinpath("train.yaml")) as f:
        train_opts = yaml.load(f, Loader=yaml.FullLoader)

    # paths
    if dataset_id in ["iwildcam", "poverty"]:
        data_path = Path(os.environ["DATASETS_PATH"]).joinpath("wilds")
    elif dataset_id in ["skin_lesion", "histology_nuclei"]:
        data_path = Path(os.environ["DATASETS_PATH"]).joinpath("pixels")
    else:
        data_path = Path(os.environ["DATA_PATH"]).joinpath(dataset_id)
    data_path.mkdir(parents=True, exist_ok=True)

    features_path = Path(os.environ["FEATURES_PATH"]).joinpath(dataset_id)
    features_path_train = features_path.joinpath("train")
    features_path_test = features_path.joinpath("test")
    features_path_train.mkdir(parents=True, exist_ok=True)
    features_path_test.mkdir(parents=True, exist_ok=True)

    # dataset
    trainset, testset = get_dataset(
        dataset_name=dataset_id,
        store_path=data_path,
        device=device,
        **datasets_opts[dataset_id],
    )

    benchmark_log = {}
    seeds = cfg["seeds"]

    for rep, seed in enumerate(seeds):
        torch.manual_seed(seed)
        train_idx = torch.randperm(len(trainset))[:cfg["max_train"]] if cfg["max_train"] is not None else None
        test_idx = torch.randperm(len(testset))[:cfg["max_test"]] if cfg["max_test"] is not None else None

        if datasets_opts[dataset_id]["normalized_labels"]:
            trainset.normalized_labels(idx=train_idx)

        for feature_id in features:
            f_train = get_feature(
                feature_name=feature_id,
                store_path=features_path_train,
                dataset=trainset,
                indices=train_idx,
                device=device,
                **features_opts[feature_id],
            )

            f_test = get_feature(
                feature_name=feature_id,
                store_path=features_path_test,
                dataset=testset,
                indices=test_idx,
                device=device,
                **features_opts[feature_id],
            )
            test_loader = DataLoader(
                f_test,
                **train_opts["dataloader"]
            )

            for model_id in models:
                print(f"R: {rep+1}/{len(seeds)}, D: {dataset_id}, F: {feature_id}, M: {model_id}, N_TRAIN: {len(f_train)}, N_TEST: {len(f_test)}")
                curr_key = f"F_{feature_id}_M_{model_id}"
                if curr_key not in benchmark_log:
                    benchmark_log[curr_key] = []

                model = get_model(
                    model_name=model_id,
                    data=f_train,
                    device=device,
                    **models_opts[model_id],
                )

                if "base" not in model_id:
                    loss = train_model(
                        model=model,
                        data=f_train,
                        cfg=train_opts,
                        device=device
                    )
                else:
                    loss = 0.0

                model.to(device)
                results_dict = test_regression(
                    model,
                    test_loader,
                    labels_norm=trainset.labels_norm if datasets_opts[dataset_id]["normalized_labels"] else None,
                )
                model.cpu()
                results_dict['loss'] = loss
                benchmark_log[curr_key].append(results_dict)

                gc.collect()
                torch.cuda.empty_cache()

    return benchmark_log


def benchmark_bayesian_opt(dataset_id, models, features, cfg, device, reps=1):
    # configs
    with open(files("configs").joinpath("datasets.yaml")) as f:
        datasets_opts = yaml.load(f, Loader=yaml.FullLoader)

    with open(files("configs").joinpath("features.yaml")) as f:
        features_opts = yaml.load(f, Loader=yaml.FullLoader)

    with open(files("configs").joinpath("models.yaml")) as f:
        models_opts = yaml.load(f, Loader=yaml.FullLoader)

    with open(files("configs").joinpath("train.yaml")) as f:
        train_opts = yaml.load(f, Loader=yaml.FullLoader)

    # paths
    if dataset_id in ["iwildcam", "poverty"]:
        data_path = Path(os.environ["DATASETS_PATH"]).joinpath("wilds")
    elif dataset_id in ["skin_lesion", "histology_nuclei"]:
        data_path = Path(os.environ["DATASETS_PATH"]).joinpath("pixels")
    else:
        data_path = Path(os.environ["DATASETS_PATH"]).joinpath(dataset_id)
    data_path.mkdir(parents=True, exist_ok=True)

    features_path = Path(os.environ["FEATURES_PATH"]).joinpath(f"{dataset_id}/full")
    features_path.mkdir(parents=True, exist_ok=True)

    # dataset
    dataset, _ = get_dataset(
        dataset_name=dataset_id,
        store_path=data_path,
        device=device,
        **datasets_opts[dataset_id],
    )
    sub_idx = torch.randperm(len(dataset))[:datasets_opts[dataset_id]["max_train"]] if datasets_opts[dataset_id]["max_train"] is not None else None

    benchmark_log = {}
    seed_count = 1
    for rep in range(reps):
        random.seed(seed_count)
        rnd_idx = random.sample(
            range(len(sub_idx) if sub_idx is not None else len(dataset)),
            cfg["initial_design_size"] + cfg["n_iters"]
        )
        train_idx = rnd_idx[:cfg["initial_design_size"]]
        seed_count += 1

        for feature_id in features:
            f_data = get_feature(
                feature_name=feature_id,
                store_path=features_path,
                dataset=dataset,
                indices=sub_idx,
                device=device,
                **features_opts[feature_id],
            )
            samples_x, samples_y = next(
                iter(
                    DataLoader(
                        f_data, batch_size=len(f_data)
                    )
                )
            )
            for model_id in models:
                print(f"R: {rep}, F: {feature_id}, M: {model_id}")
                curr_key = f"F_{feature_id}_M_{model_id}"
                if curr_key not in benchmark_log:
                    benchmark_log[curr_key] = []

                if "base" in model_id:
                    benchmark_log[curr_key].append(
                        get_regrets(
                            rnd_idx,
                            samples_y,
                            cfg["initial_design_size"],
                            cfg["n_iters"]
                        )
                    )
                else:
                    curr_idx = deepcopy(train_idx)
                    for bo_iter in range(cfg["n_iters"]):
                        train_x, train_y = samples_x[curr_idx], samples_y[curr_idx]
                        if datasets_opts[dataset_id]["normalized_labels"]:
                            mu_y, std_y = train_y.mean(), train_y.std()
                            train_y.sub_(mu_y).div_(std_y)
                        model = get_model(
                            model_name=model_id,
                            data=TensorDataset(train_x, train_y),
                            device=device,
                            **models_opts[model_id],
                        )
                        loss = train_model(
                            model=model,
                            data=TensorDataset(train_x, train_y),
                            cfg=train_opts,
                            device=device
                        )
                        with torch.no_grad():
                            model.eval()
                            logEI = LogExpectedImprovement(model=model, best_f=train_y.max())
                            curr_idx.append(torch.argmax(logEI(samples_x.unsqueeze(1))).item())
                        del model, logEI
                        torch.cuda.empty_cache()
                    benchmark_log[curr_key].append(
                        get_regrets(
                            curr_idx,
                            samples_y,
                            cfg["initial_design_size"],
                            cfg["n_iters"]
                        )
                    )

    return benchmark_log
