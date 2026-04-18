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

# %%
os.environ["DATA_PATH"] = "/Users/bernardo/Documents/projects/bayesian-scattering/data/datasets"
os.environ["FEATURES_PATH"] = "/Users/bernardo/Documents/projects/bayesian-scattering/data/features"
os.environ["RESULTS_PATH"] = "/Users/bernardo/Documents/projects/bayesian-scattering/data/results"


# %% [markdown]
# ## Settings

# %%
dataset_id = "skin_lesion"
feature_id = "timm_convnext_base"
model_id = "la_reg_nonlinear"

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
# x, y = next(iter(DataLoader(trainset, len(trainset))))
# x.mean(dim=(0, 2, 3))
# x.std(dim=(0, 2, 3))

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

# %%
trainset[0][0].shape

# %%
f_train.dataset.transforms

# %%
f_train.dataset.features_from_sample(trainset[6025][0]).shape

# %%
f_train[0][0].shape

# %%
import torchvision
for i, t in enumerate(f_train.dataset.transforms.transforms):
    if isinstance(t, torchvision.transforms.Resize):
        print(t.size)

# %%
torchvision.transforms.Compose()

# %%
import timm
model = timm.create_model(
    "vit_small_patch14_reg4_dinov2.lvd142m",
    pretrained=True,
    # features_only=True,
    num_classes=0,  # remove classifier nn.Linear
    in_chans=8,
)
model = model.eval()
transforms = timm.data.create_transform(
    **timm.data.resolve_model_data_config(model),
    is_training=False
)


# %%
model(transforms(x[0:2]))

# %%
transforms

# %%
trainset[0][0].numpy()

# %%
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])

# %%
tmp = x[0]/255.

# %%
tmp.view(tmp.size(0), -1).mean(dim=1)

# %%
transform(x/255.).mean(dim=(0, 2, 3))

# %%
(x/255.).mean(dim=(0, 2, 3))

# %%
f_train.dataset.transforms

# %%
f_train.dataset.ws(f_train.dataset.transform(trainset[0][0][:3]))[0][0]

# %%
# %matplotlib widget
import matplotlib as mpl
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)

for i in torch.randperm(len(f_test))[:2]:
    neg_idx = f_train[i][0] < 0
    y = f_train[i][0][neg_idx]
    ax.plot(torch.arange(len(y)), y)
plt.show()

# %%
fig = plt.figure()
ax = fig.add_subplot(111)

for i in torch.randperm(len(f_train))[:2]:
    y = f_train[i][0]
    ax.plot(torch.arange(len(y)), y)
plt.show()

# %%
X_train = next(iter(Dataloader(f_trian, len(f_train)))
X_test = next(iter(Dataloader(f_trian, len(f_test)))

# %%
fig = plt.figure()
ax = fig.add_subplot(111)

x = torch.arange(len(y))
y = X_train.mean(axis=0)
sigma = X_train.std(axis=0)

ax.plot(x, y)
ax.fill_between(
    *[x, y - 2 * sigma, y + 2 * sigma],
    alpha=0.2,
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
