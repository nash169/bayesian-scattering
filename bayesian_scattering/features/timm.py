import torch
import timm
import torchvision
import numpy as np

from typing import Optional, Self

from bayesian_scattering.datasets.abstract_dataset import AbstractDataset
from bayesian_scattering.features.abstract_features import AbstractFeatures


class TorchImageModel(AbstractFeatures):
    def __init__(
        self,
        pretrained_model="convnext_base.fb_in1k",
        store_path=None,
        dataset: Optional[AbstractDataset] = None,
        transforms: Optional[torchvision.transforms.Compose] = None,
        shard_size: int = 5_000,
        dtype=np.float32,
        **kwargs,
    ):
        self.model = timm.create_model(
            pretrained_model,
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
        )
        self.model = self.model.eval()

        super().__init__(
            dataset=dataset,
            transforms=transforms,
            cache_root=store_path,
            features_id=pretrained_model.replace(".", "_"),
            shard_size=shard_size,
            dtype=dtype,
        )

        if self.transforms is None:
            self.transforms = timm.data.create_transform(
                **timm.data.resolve_model_data_config(self.model), is_training=False
            )

    def features_from_sample(self, x: torch.Tensor) -> torch.Tensor:
        x = (
            self.transforms(x).to(self.device)
            if self.transforms is not None
            else x.to(self.device)
        )
        single_sample = x.ndim == 3
        if single_sample:
            x = x.unsqueeze(0)

        with torch.no_grad():
            y = self.model(x)

        return y.flatten() if single_sample else y.reshape(x.shape[0], -1)

    def to(self, device) -> Self:
        self.device = device
        self.model.to(device)
        return self
