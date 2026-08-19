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
# # Molecular Energies Bayesian Optimization
#
# Notebook version of `benchmarks/mol_bayesian_opt.py`.

# %%
import os
import sys
import yaml
import time
import torch
import pickle
import matplotlib.pyplot as plt
from importlib.resources import files
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join('..')))
from bayesian_scattering.utils.benchmark import benchmark_bayesian_opt
from bayesian_scattering.utils.helpers import get_regrets, get_aucs, plot_regrets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
def infer_dataset_id(log_name):
    parts = log_name.split("_")
    if len(parts) < 7:
        return None
    if parts[0:2] != ["bayesian", "opt"]:
        return None
    dataset_id = "_".join(parts[3:-5])
    return dataset_id or None


# %% [markdown]
# ## Settings

# %%
load_log = True
# name_log =

# 2D
# dataset = "qm2"
# features = [
#     "unimol_qm2",
#     "scattering_inv_J5_L8",
#     "scattering_inv_J6_L8",
#     "scattering_inv_J7_L8",
#     "scattering_cov_J5_L8",
#     "scattering_cov_J6_L8",
#     "scattering_cov_J7_L8",
# ]

# 3D
dataset = "qm9"
features = [
    "unimol_qm9",
    "scattering_inv_3D_J4_L3",
    "scattering_cov_3D_J4_L3",
]

models = [
    "baseline",
    "gp_exact_reg_rbf",
    "gp_exact_reg_rbf_ard",
    "gp_exact_reg_matern52",
    "gp_exact_reg_matern52_ard",
]

with open(files("benchmarks").joinpath("config.yaml")) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)["bayesian_opt"]

# %% [markdown]
# ## Run benchmark

# %%
if load_log:
    inferred_dataset = infer_dataset_id(name_log)
    if inferred_dataset is not None:
        dataset = inferred_dataset
    path_log = Path(os.environ["RESULTS_PATH"]).joinpath(f"{name_log}.pkl")
    with open(path_log, 'rb') as handle:
        benchmark_log = pickle.load(handle)
else:
    t0 = time.time()
    benchmark_log = benchmark_bayesian_opt(
        dataset_id=dataset,
        models=models,
        features=features,
        cfg=cfg,
        device=device,
        reps=5,
    )
    print(f"Benchmark Time: {time.time() - t0}s")
    results_path = Path(os.environ["RESULTS_PATH"]).joinpath(
        f"bayesian_opt_{cfg['datasets'][dataset]['max_train']}_{dataset}_{cfg['datasets'][dataset]['target'] if dataset == "qm9" else ""}_{datetime.now().strftime("%y_%m_%d_%H_%M")}.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(benchmark_log, f)


# %%
def split_benchmark_log(benchmark_log, baseline_model="baseline"):
    baseline_keys = [
        key for key in benchmark_log if key.endswith(f"_M_{baseline_model}")
    ]
    if not baseline_keys:
        raise ValueError("No baseline entry found in benchmark log.")
    baseline_key = baseline_keys[0]
    baseline_val = benchmark_log[baseline_key]
    model_logs = {}
    for key, val in benchmark_log.items():
        if "_M_" not in key:
            continue
        model = key.split("_M_")[-1]
        if model == baseline_model:
            continue
        if model not in model_logs:
            model_logs[model] = {baseline_key: baseline_val}
        model_logs[model][key] = val
    return model_logs


model_logs = split_benchmark_log(benchmark_log)

# %% [markdown]
# ## AUC

# %%
for model_name, model_log in model_logs.items():
    print(model_name)
    for key, val in model_log.items():
        print(key, get_aucs(val).mean())

# %% [markdown]
# ## Plot

# %%
plots_path = Path(os.environ["RESULTS_PATH"]).joinpath("plots")
plots_path.mkdir(parents=True, exist_ok=True)

for model_name, model_log in model_logs.items():
    f, ax = plt.subplots()
    cur_col = 0
    for experiment_name, regrets_rep_bo in model_log.items():
        plot_regrets(
            regrets_rep_bo,
            ax,
            cfg["bayesian_opt"]["initial_design_size"],
            cfg["bayesian_opt"]["n_iters"],
            color=plt.get_cmap("tab20")(cur_col),
            label=experiment_name,
        )
        cur_col += 1
    ax.set_title(model_name)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    figure_path = plots_path.joinpath(f"bo_{dataset}_{model_name}.png")
    f.savefig(figure_path, bbox_inches="tight")
    plt.show()

# %%
