import torch
import numpy as np

from typing import Self, Optional, Union

from bayesian_scattering.datasets.abstract_dataset import AbstractDataset
from bayesian_scattering.features.abstract_features import AbstractFeatures


class Identity(AbstractFeatures):
    def __init__(
        self,
        dataset: Optional[AbstractDataset] = None,
        **kwargs
    ):
        super().__init__(
            dataset=dataset,
        )

    def features_from_sample(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def to(self, device) -> Self:
        self.device = device
        return self

    def __getitem__(self, idx: Union[int, torch.Tensor, np.ndarray]):
        return self.base[idx]
