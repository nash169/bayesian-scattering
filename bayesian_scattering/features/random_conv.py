import torch
import numpy as np
import torch.nn.functional as F

from pathlib import Path
from typing import Self, Optional, Union

from bayesian_scattering.datasets.abstract_dataset import AbstractDataset
from bayesian_scattering.features.abstract_features import AbstractFeatures


class RandomConv(AbstractFeatures):
    """Random-filter counterpart of the wavelet scattering transform: the same
    cascade of convolution, modulus and local averaging, with the Morlet
    wavelets replaced by random filters of matched dyadic support. It is a
    control for how much of the scattering performance comes from the wavelet
    structure rather than from the architecture around it, so it mirrors the
    rotation-invariant grouping too and has the same output dimension."""

    def __init__(
        self,
        store_path,
        J: int,
        L: int,
        max_order: int = 2,
        dataset: Optional[AbstractDataset] = None,
        indices=None,
        **kwargs
    ):
        super().__init__(dataset=dataset)
        self.J, self.L, self.max_order = J, L, max_order

        # the filters are redrawn from scratch for every seed, as the pca basis
        # is refit; the file only hands them over to the test/val splits, it is
        # not a cache
        store_path = Path(store_path)
        filters_path = store_path.parent.joinpath(f"randconv_j{J}_l{L}_filters.pt")
        if store_path.name in ["train", "full"]:
            # one zero-mean unit-norm filter per (scale, orientation), with the
            # support growing dyadically as the wavelet one does
            self.filters = []
            for j in range(J):
                size = 2 ** (j + 1) + 1
                f = torch.randn(L, 1, size, size)
                f -= f.mean(dim=(-2, -1), keepdim=True)
                self.filters.append(f / f.flatten(1).norm(dim=1).view(-1, 1, 1, 1))
            torch.save(self.filters, filters_path)
        elif filters_path.exists():
            self.filters = torch.load(filters_path)
        else:
            raise FileNotFoundError(
                f"Random filters not found at {filters_path}. "
                "Build the train features first to draw them."
            )

        # transformed once for the subset this seed uses and kept in memory, so
        # that the filters can change from seed to seed without a stale cache
        self.position = {
            int(idx): k
            for k, idx in enumerate(range(len(self.base)) if indices is None else indices)
        }
        self.features = torch.stack(
            [self.features_from_sample(self.base[idx][0]) for idx in self.position]
        )

    def convolve(self, x: torch.Tensor, j: int) -> torch.Tensor:
        # (N, 1, H, W) -> (N, L, H, W)
        return F.conv2d(x, self.filters[j], padding=self.filters[j].shape[-1] // 2).abs()

    def average(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x, 2 ** self.J)

    def features_from_sample(self, x: torch.Tensor) -> torch.Tensor:
        channels, height, width = x.shape
        # every input channel is filtered independently, as in the 2D scattering
        u = x.cpu().unsqueeze(1)
        coeff = [self.average(u)]

        for j1 in range(self.J):
            u1 = self.convolve(u, j1)
            # average over the orientations, mirroring the rotation-invariant
            # grouping of the scattering coefficients
            coeff.append(self.average(u1).mean(dim=1, keepdim=True))

            if self.max_order > 1:
                for j2 in range(j1 + 1, self.J):
                    u2 = self.convolve(u1.reshape(-1, 1, height, width), j2)
                    # keep the first orientation, average over the second
                    pooled = self.average(u2).mean(dim=1)
                    coeff.append(pooled.reshape(channels, self.L, *pooled.shape[-2:]))

        return torch.cat(coeff, dim=1).flatten()

    def to(self, device) -> Self:
        self.device = device
        return self

    def __getitem__(self, idx: Union[int, torch.Tensor, np.ndarray]):
        return self.features[self.position[int(idx)]], self.base[idx][1]
