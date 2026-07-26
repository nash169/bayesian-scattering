import torch
import numpy as np

from typing import Optional, Self
from kymatio.torch import Scattering2D, HarmonicScattering3D
from kymatio.scattering3d.backend.torch_backend import TorchBackend3D
from torchvision.ops.boxes import torchvision

from bayesian_scattering.datasets.abstract_dataset import AbstractDataset
from bayesian_scattering.features.abstract_features import AbstractFeatures


class WaveletScattering(AbstractFeatures):
    def __init__(
        self,
        dimension=None,
        rot_invariant=False,
        store_path: Optional[str] = None,
        dataset: Optional[AbstractDataset] = None,
        transforms: Optional[torchvision.transforms.Compose] = None,
        shard_size: int = 5_000,
        dtype=np.float32,
        **kwargs
    ):
        if dataset is None:
            assert dimension is not None, "dimension unknown"
            assert "shape" in kwargs, "shape unknown"
        else:
            dimension = len(dataset.shape)
            kwargs['shape'] = dataset.shape[-dimension:]
            if transforms is not None:
                for t in transforms.transforms:
                    if isinstance(t, torchvision.transforms.Resize):
                        kwargs['shape'] = t.size

        if dimension == 1:
            raise NotImplementedError("1D not implemented")
        elif dimension == 2:
            self.rot_invariant = rot_invariant
            self.ws = Scattering2D(
                out_type="list" if self.rot_invariant else "array",
                **kwargs
            )
            prefix = f'scatter_j{kwargs.get("J")}_l{kwargs.get("L")}_{"ri" if rot_invariant else "rc"}'
            # max_order=2 keeps the historical prefix so existing caches stay valid
            if kwargs.get("max_order", 2) != 2:
                prefix += f'_o{kwargs.get("max_order")}'
        elif dimension == 3:
            self.ws = HarmonicScattering3D(rotation_covariant=rot_invariant, **kwargs)
            prefix = f'scatter_j{kwargs.get("J")}_l{kwargs.get("L")}_p{int(kwargs.get("integral_powers")[-1])}_{"ri" if rot_invariant else "rc"}'
        else:
            raise TypeError(f"Expected dimension equal to 1, 2 or 3, got {dimension}.")

        super().__init__(
            dataset=dataset,
            transforms=transforms,
            cache_root=store_path,
            features_id=prefix,
            shard_size=shard_size,
            dtype=dtype,
        )

    def features_from_sample(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        x = x.to(self.device)
        if self.transforms is not None:
            x = self.transforms(x)

        if len(self.ws.shape) == 1:
            return x
        elif len(self.ws.shape) == 2:
            coeff = self.ws.scattering(x)
            if self.rot_invariant:
                coeff = self.rotation_invariant(coeff)
            return coeff.flatten(start_dim=1 if len(x.shape) > 3 else 0)
        elif len(self.ws.shape) == 3:
            return torch.hstack(
                (
                    TorchBackend3D.compute_integrals(x, self.ws.integral_powers).flatten(start_dim=1 if len(x.shape) > 4 else 0),
                    self.ws.scattering(x).flatten(start_dim=1 if len(x.shape) > 4 else 0)
                ),
            )
        else:
            raise ValueError(f"Scattering dimension equal to 1, 2 or 3, got {len(self.ws.shape)}.")

    def rotation_invariant(
        self,
        coeff_list: list,
    ) -> torch.Tensor:
        coeff_shape = coeff_list[0]["coef"].shape
        scat_dict = {(x["j"], x["n"], x["theta"]): x["coef"] for x in coeff_list}

        scat_dict_inv = {}
        for (j, n, theta), coeff in scat_dict.items():
            assert len(j) == len(n) and len(n) == len(theta) and coeff.shape == coeff_shape
            # n is just a unique identifier without a special meaning, we thus discard it
            if len(j) == 1 or len(j) == 0:  # first order coefficient
                key = (j,)
            elif len(j) == 2:  # second order coefficients
                key = (j, theta[0])
            elif len(j) > 2:
                raise ValueError("We only support 1st and 2nd order scattering coefficients")
            if key not in scat_dict_inv:
                scat_dict_inv[key] = torch.zeros(coeff_shape, device=coeff.device)
            scat_dict_inv[key] += coeff / self.ws.L
        return torch.stack([value for _, value in scat_dict_inv.items()], axis=1)

    def to(self, device) -> Self:
        self.device = device
        self.ws.to(device)
        return self
