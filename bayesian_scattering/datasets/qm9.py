import numpy as np
import torch
from typing import Optional, Tuple, Self
from scipy.spatial.distance import pdist
from kymatio.datasets import fetch_qm9
from bayesian_scattering.datasets.abstract_dataset import AbstractDataset


class QM9(AbstractDataset):
    targets = ["A", "B", "C", "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "U0", "U", "H", "G", "Cv"]

    def __init__(
            self,
            grid_res: Optional[Tuple] = (64, 64, 64),
            rescale: Optional[bool] = True,
            sigma: Optional[float] = 2.0,
            overlapping_precision: Optional[float] = 1e-1,
            target: str = "homo",
            precision: torch.dtype = torch.float32,
            store_path: Optional[str] = None,
            shard_size: int = 10_000,
            dtype=np.float32,
            unimol: bool = False,
            **kwargs
    ):
        qm9 = fetch_qm9(align=True, cache=True, dir=store_path)
        self.coordinates = qm9['positions']
        self.full_charges = qm9['charges']
        self.labels = qm9['energies'][:, self.targets.index(target)]

        mask = self.full_charges <= 2
        self.valence_charges = self.full_charges * mask

        mask = np.logical_and(self.full_charges > 2, self.full_charges <= 10)
        self.valence_charges += (self.full_charges - 2) * mask

        mask = np.logical_and(self.full_charges > 10, self.full_charges <= 18)
        self.valence_charges += (self.full_charges - 10) * mask

        assert sigma is not None
        self.sigma = sigma

        if rescale:
            min_dist = np.inf

            n_molecules = self.coordinates.shape[0]
            for i in range(n_molecules):
                n_atoms = np.sum(self.full_charges[i] != 0)
                pos_i = self.coordinates[i, :n_atoms, :]
                min_dist = min(min_dist, pdist(pos_i).min())

            assert overlapping_precision is not None
            delta = self.sigma * np.sqrt(-8 * np.log(overlapping_precision))
            self.coordinates = self.coordinates * delta / min_dist

        assert grid_res is not None
        self.grid_res = grid_res
        M, N, O = grid_res[0], grid_res[1], grid_res[2]  # 192, 128, 96
        self.grid = np.mgrid[-M // 2:-M // 2 + M, -N // 2:-N // 2 + N, -O // 2:-O // 2 + O]
        self.grid = np.fft.ifftshift(self.grid)

        self.coordinates = torch.from_numpy(self.coordinates).to(precision)
        self.grid = torch.from_numpy(self.grid).to(precision)
        self.labels = torch.from_numpy(self.labels).to(precision)
        self.full_charges = torch.from_numpy(self.full_charges).to(precision)
        self.valence_charges = torch.from_numpy(self.valence_charges).to(precision)

        self.device = torch.device("cpu")
        self.unimol = unimol
        self.atomic_number = {
            1: 'H',
            6: 'C',
            7: 'N',
            8: 'O',
            9: 'F',
            16: 'S'
        }
        super().__init__(
            cache_root=store_path,
            dataset_id=f"qm9_{M}_{N}_{O}",
            shard_size=shard_size,
            dtype=dtype
        )

    def __getitem__(self, idx: int):
        if not self.unimol:
            return super().__getitem__(idx)
        return self.unimol_dict(idx), self.labels[idx]

    def generate_sample(self, idx):
        sample = torch.zeros((self.grid.shape))

        # full
        sample[0] = self.generate_weighted_sum_of_gaussians(
            self.coordinates[idx],
            self.full_charges[idx]
        ).cpu()

        # valence
        sample[1] = self.generate_weighted_sum_of_gaussians(
            self.coordinates[idx],
            self.valence_charges[idx]
        ).cpu()

        # core
        sample[2] = sample[0] - sample[1]

        return sample

    def __len__(self):
        return len(self.labels)

    @property
    def shape(self) -> torch.Size:
        return torch.Size([self.grid_res[0], self.grid_res[1], self.grid_res[2]])

    @property
    def channels(self) -> int:
        return 3

    def generate_weighted_sum_of_gaussians(self, position, weights):
        signal = torch.zeros((self.grid.shape[1:])).to(self.device)
        n_points = position.shape[0]
        for i_point in range(n_points):
            if weights[i_point] == 0:
                break
            weight = weights[i_point]
            center = position[i_point]
            signal = torch.add(signal, weight * torch.exp(
                -0.5 * ((self.grid[0] - center[0]) ** 2 +
                        (self.grid[1] - center[1]) ** 2 +
                        (self.grid[2] - center[2]) ** 2) / self.sigma**2))
        return signal / ((2 * torch.pi) ** 1.5 * self.sigma ** 3)

    def to(self, device) -> Self:
        self.device = device
        self.coordinates = self.coordinates.to(device)
        self.grid = self.grid.to(device)
        self.full_charges = self.full_charges.to(device)
        self.valence_charges = self.valence_charges.to(device)
        return self

    def unimol_dict(self, idx):
        formatted_atoms = [self.atomic_number[int(z)] for z in self.full_charges[idx] if z > 0]
        formatted_coords = self.coordinates[idx][self.full_charges[idx] != 0].tolist()
        return {'atoms': formatted_atoms, 'coordinates': formatted_coords}
