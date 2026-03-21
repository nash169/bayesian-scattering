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

import torch
from torch.nn.modules import activation
from torchvision.ops import MLP
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join("../")))
from bayesian_scattering.utils import test_regression, train_approx_gp, train_exact_gp
from bayesian_scattering.utils.benchmark import get_dataset, get_feature, get_model, train_model

torch.manual_seed(1337)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
os.environ["DATA_PATH"] =
os.environ["FEATURES_PATH"] =
os.environ["RESULTS_PATH"] =


# %% [markdown]
# ## Settings

# %%
dataset_id = "skin_lesion"
feature_id = "scattering_inv_J5_L8"
model_id = "gp_exact_reg_rbf"

with open(files("benchmarks").joinpath("config.yaml")) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

# %% [markdown]
# ## Dataset

# %%
if dataset_id in ["iwildcam", "poverty"]:
    data_path = Path(os.environ["DATA_PATH"]).joinpath("wilds")
elif dataset_id in ["skin_lesion", "histology_nuclei"]:
    data_path = Path(os.environ["DATA_PATH"]).joinpath("pixels")
else:
    data_path = Path(os.environ["DATA_PATH"]).joinpath(dataset_id)
data_path.mkdir(parents=True, exist_ok=True)

trainset, testset = get_dataset(
    dataset_name=dataset_id,
    store_path=data_path,
    device=device,
    **cfg["datasets"][dataset_id],
)
train_idx = torch.randperm(len(trainset))[:cfg["datasets"][dataset_id]["max_train"]] if cfg["datasets"][dataset_id]["max_train"] is not None else None
test_idx = torch.randperm(len(testset))[:cfg["datasets"][dataset_id]["max_test"]] if cfg["datasets"][dataset_id]["max_test"] is not None else None

if cfg["train"]["normalized_labels"]:
    trainset.normalized_labels(idx=train_idx)

# %% [markdown]
# ## Features

# %%
features_path = Path(os.environ["FEATURES_PATH"]).joinpath(dataset_id)
features_path_train, features_path_test = features_path.joinpath("train"), features_path.joinpath("test")
features_path_train.mkdir(parents=True, exist_ok=True)
features_path_test.mkdir(parents=True, exist_ok=True)

f_train = get_feature(
    feature_name=feature_id,
    store_path=features_path_train,
    dataset=trainset,
    indices=train_idx,
    device=device,
    **cfg["features"][feature_id],
)

f_test = get_feature(
    feature_name=feature_id,
    store_path=features_path_test,
    dataset=testset,
    indices=test_idx,
    device=device,
    **cfg["features"][feature_id],
)

# %% [markdown]
# ## Model

# %%
model = get_model(
    model_name=model_id,
    data=f_train,
    device=device,
    **cfg["models"][model_id],
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
model.to(device)
results_log = test_regression(
    model,
    test_loader,
    labels_norm=trainset.labels_norm if cfg["train"]["normalized_labels"] else None
)
model.cpu()
