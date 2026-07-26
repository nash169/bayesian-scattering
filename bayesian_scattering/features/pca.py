import torch
import numpy as np

from pathlib import Path
from typing import Self, Optional, Union

from bayesian_scattering.datasets.abstract_dataset import AbstractDataset
from bayesian_scattering.features.abstract_features import AbstractFeatures


class PCA(AbstractFeatures):
    def __init__(
        self,
        store_path,
        dataset: Optional[AbstractDataset] = None,
        indices=None,
        n_components: int = 256,
        **kwargs
    ):
        super().__init__(
            dataset=dataset,
        )
        self.n_components = n_components

        # the basis is refit from scratch on the training subset it is given;
        # the file only hands it over to the test/val splits, it is not a cache
        store_path = Path(store_path)
        basis_path = store_path.parent.joinpath(f"pca_{n_components}_basis.pt")
        if store_path.name in ["train", "full"]:
            fit_idx = range(len(self.base)) if indices is None else indices
            x = torch.stack([self.base[i][0].flatten().cpu() for i in fit_idx])
            self.mean = x.mean(dim=0)
            _, _, self.components = torch.pca_lowrank(
                x - self.mean, q=n_components, center=False
            )
            torch.save(
                {"mean": self.mean, "components": self.components}, basis_path
            )
        elif basis_path.exists():
            basis = torch.load(basis_path)
            self.mean, self.components = basis["mean"], basis["components"]
        else:
            raise FileNotFoundError(
                f"PCA basis not found at {basis_path}. "
                "Build the train features first to fit it."
            )

    def features_from_sample(self, x: torch.Tensor) -> torch.Tensor:
        return (x.flatten().cpu() - self.mean) @ self.components

    def to(self, device) -> Self:
        self.device = device
        return self

    def __getitem__(self, idx: Union[int, torch.Tensor, np.ndarray]):
        sample, label = self.base[idx]
        return self.features_from_sample(sample), label
