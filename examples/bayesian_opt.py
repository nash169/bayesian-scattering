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
# Notebook version of `examples/bayesian_opt.py`.

# %%
import os
import sys
import random
from importlib.resources import files
from pathlib import Path
from copy import deepcopy

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from botorch.acquisition import LogExpectedImprovement

sys.path.insert(0, os.path.abspath(os.path.join("../")))
from bayesian_scattering.utils.helpers import get_dataset, get_feature, get_model, train_model, get_regrets

torch.manual_seed(1337)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ## Settings

# %%
dataset_id = "qm9"
feature_id = "scattering_inv_3D_J4_L3"
model_id = "gp_exact_reg_rbf"

with open(files("benchmarks").joinpath("config.yaml")) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

# %% [markdown]
# ## Storing

# %%
if dataset_id in ["iwildcam", "poverty"]:
    data_path = Path(os.environ["DATA_PATH"]).joinpath("wilds")
elif dataset_id in ["skin_lesion", "histology_nuclei"]:
    data_path = Path(os.environ["DATA_PATH"]).joinpath("pixels")
else:
    data_path = Path(os.environ["DATA_PATH"]).joinpath(dataset_id)
data_path.mkdir(parents=True, exist_ok=True)

features_path = Path(os.environ["FEATURES_PATH"]).joinpath(f"{dataset_id}/full")
features_path.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Dataset

# %%
dataset, _ = get_dataset(
    dataset_name=dataset_id,
    store_path=data_path,
    device=device,
    **cfg["datasets"][dataset_id],
)
sub_idx = torch.randperm(len(dataset))[:cfg["datasets"][dataset_id]["max_train"]] if cfg["datasets"][dataset_id]["max_train"] is not None else None

# %% [markdown]
# ## Features

# %%
features = get_feature(
    feature_name=feature_id,
    store_path=features_path,
    dataset=dataset,
    indices=sub_idx,
    device=device,
    **cfg["features"][feature_id],
)
samples_x, samples_y = next(
    iter(
        DataLoader(
            features, batch_size=len(features)
        )
    )
)

# %% [markdown]
# ## BO

# %%
random.seed(0)
rnd_idx = random.sample(
    range(len(sub_idx) if sub_idx is not None else len(dataset)),
    cfg["bayesian_opt"]["initial_design_size"] + cfg["bayesian_opt"]["n_iters"]
)
train_idx = rnd_idx[:cfg["bayesian_opt"]["initial_design_size"]]

curr_idx = deepcopy(train_idx)
for iter in range(cfg["bayesian_opt"]["n_iters"]):
    train_x, train_y = samples_x[curr_idx], samples_y[curr_idx]
    if cfg["train"]["normalized_labels"]:
        mu_y, std_y = train_y.mean(), train_y.std()
        train_y.sub_(mu_y).div_(std_y)
    model = get_model(
        model_name=model_id,
        data=TensorDataset(train_x, train_y),
        device=device,
        **cfg["models"][model_id],
    )
    loss = train_model(
        model=model,
        data=features,
        cfg=cfg
    )
    with torch.no_grad():
        model.eval()
        logEI = LogExpectedImprovement(model=model, best_f=train_y.max())
        curr_idx.append(torch.argmax(logEI(samples_x.unsqueeze(1))).item())
    del model, logEI

# %% [markdown]
# ## Results

# %%
results_log = get_regrets(
    curr_idx,
    samples_y,
    cfg["bayesian_opt"]["initial_design_size"],
    cfg["bayesian_opt"]["n_iters"]
)
