import torch
from typing import Optional, Self
from unimol_tools import UniMolRepr

from bayesian_scattering.datasets import QM9
from bayesian_scattering.datasets.qm2 import QM2
from bayesian_scattering.features.abstract_features import AbstractFeatures

# import os
# os.environ.setdefault("TQDM_DISABLE", "1")
#
# import tqdm
# def _quiet_tqdm(*args, **kwargs):
#     kwargs["disable"] = True
#     return _tqdm(*args, **kwargs)
#
#
# _tqdm = tqdm.tqdm
# tqdm.tqdm = _quiet_tqdm
# try:
#     from tqdm import auto as tqdm_auto
#
#     tqdm_auto.tqdm = _quiet_tqdm
# except Exception:
#     pass
#
# import logging
# logging.getLogger().setLevel(logging.WARNING)
# logging.getLogger("Uni-Mol Tools").setLevel(logging.WARNING)


class Unimol(AbstractFeatures):
    def __init__(
        self,
        store_path=None,
        dataset: Optional[QM2 | QM9] = None,
        **kwargs
    ):
        self.model = UniMolRepr(data_type='molecule', remove_hs=False)

        super().__init__(
            dataset=dataset,
            cache_root=store_path,
            features_id="unimol"
        )

    def features_from_sample(self, x: dict) -> torch.Tensor:
        return torch.from_numpy(self.model.get_repr(x, return_atomic_reprs=False)[0])

    def to(self, device) -> Self:
        self.device = device
        self.model.to(device)
        return self
