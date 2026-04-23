# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Regression Example
#
# Notebook version of `examples/regression.py`.

# %%
import os
import sys
import time
from importlib.resources import files
from pathlib import Path
import logging

import torch
from torch.nn.modules import activation
from torchvision.ops import MLP
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join("../")))
from bayesian_scattering.utils import test_regression, train_approx_gp, train_exact_gp
from bayesian_scattering.utils.benchmark import get_dataset, get_feature, get_model, train_model

logging.getLogger().setLevel(logging.WARNING)
torch.manual_seed(1337)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ## Settings

# %%
dataset_id = "histology_nuclei"
feature_id = "timm_convnext_atto"
model_id = "gp_reg_rbf"

# %% [markdown]
# ## Configs

# %%
with open(files("configs").joinpath("datasets.yaml")) as f:
    dataset_opts = yaml.load(f, Loader=yaml.FullLoader)[dataset_id]

with open(files("configs").joinpath("features.yaml")) as f:
    feature_opts = yaml.load(f, Loader=yaml.FullLoader)[feature_id]

with open(files("configs").joinpath("models.yaml")) as f:
    model_opts = yaml.load(f, Loader=yaml.FullLoader)[model_id]

with open(files("configs").joinpath("train.yaml")) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

# %% [markdown]
# ## Paths

# %%
if dataset_id in ["iwildcam", "poverty"]:
    data_path = Path(os.environ["DATASETS_PATH"]).joinpath("wilds")
elif dataset_id in ["skin_lesion", "histology_nuclei"]:
    data_path = Path(os.environ["DATASETS_PATH"]).joinpath("pixels")
else:
    data_path = Path(os.environ["DATASETS_PATH"]).joinpath(dataset_id)
data_path.mkdir(parents=True, exist_ok=True)

features_path = Path(os.environ["FEATURES_PATH"]).joinpath(dataset_id)
features_path_train, features_path_test = features_path.joinpath("train"), features_path.joinpath("test")
features_path_train.mkdir(parents=True, exist_ok=True)
features_path_test.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Dataset

# %%
trainset, testset = get_dataset(
    dataset_name=dataset_id,
    store_path=data_path,
    device=device,
    **dataset_opts,
)
train_idx = torch.randperm(len(trainset))[:dataset_opts["max_train"]] if dataset_opts["max_train"] is not None else None
test_idx = torch.randperm(len(testset))[:dataset_opts["max_test"]] if dataset_opts["max_test"] is not None else None

if dataset_opts["normalized_labels"]:
    trainset.normalized_labels(idx=train_idx)

# %% [raw]
# x, y = next(iter(DataLoader(trainset, len(trainset))))

# %% [raw]
# x.mean(dim=(0, 2, 3))

# %% [raw]
# x.std(dim=(0, 2, 3))

# %% [markdown]
# ## Features

# %%
f_train = get_feature(
    feature_name=feature_id,
    store_path=features_path_train,
    dataset=trainset,
    indices=train_idx,
    device=device,
    **feature_opts,
)

f_test = get_feature(
    feature_name=feature_id,
    store_path=features_path_test,
    dataset=testset,
    indices=test_idx,
    device=device,
    **feature_opts,
)

# %% [markdown]
# ## Model

# %%
model = get_model(
    model_name=model_id,
    data=f_train,
    device=device,
    **model_opts,
)

# %% [markdown]
# ## Train

# %%
loss = train_model(
    model=model,
    data=f_train,
    cfg=cfg,
    device=device
)

# %% [markdown]
# ## Test

# %%
test_loader = DataLoader(
    f_test,
    **cfg["dataloader"]
)
results_log = test_regression(
    model,
    test_loader,
    labels_norm=trainset.labels_norm if dataset_opts["normalized_labels"] else None
)

# %%
results_log

# %%
