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
# # Distribution Shift Regression Benchmark
#
# Notebook version of `benchmarks/dist_shift_regression.py`.

# %%
import os
import sys
import yaml
import time
import torch
import pickle
from importlib.resources import files
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join('..')))
from bayesian_scattering.utils.benchmark import benchmark_regression
from bayesian_scattering.utils.helpers import get_results

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
benchmark_id = "dist_shift_cut_skin_lesion_new"

# %% [markdown]
# ## Settings

# %%
with open(files("configs").joinpath("benchmarks.yaml")) as f:
    benchmarks_cfg = yaml.load(f, Loader=yaml.FullLoader)[benchmark_id]
    
dataset = benchmarks_cfg["dataset"]
features = benchmarks_cfg["features"]
models = benchmarks_cfg["models"]

# %% [markdown]
# ## Run benchmark

# %%
t0 = time.time()
benchmark_log = benchmark_regression(
    dataset_id=dataset,
    models=models,
    features=features,
    cfg=benchmarks_cfg,
    device=device,
)
print(f"Benchmark Time: {time.time() - t0}s")

# %% [markdown]
# ## Summarize results

# %%
df = get_results(
    benchmark_log,
    store_path=Path(os.environ["RESULTS_PATH"]).joinpath(f"{benchmark_id}")
)

# %%
df

# %%
