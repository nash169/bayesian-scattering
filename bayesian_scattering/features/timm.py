import torch
import timm

from typing import Optional, Self

from bayesian_scattering.datasets.abstract_dataset import AbstractDataset
from bayesian_scattering.features.abstract_features import AbstractFeatures


class TorchImageModel(AbstractFeatures):
    def __init__(
        self,
        pretrained_model='convnext_base.fb_in1k',
        store_path=None,
        dataset: Optional[AbstractDataset] = None,
        **kwargs
    ):
        self.model = timm.create_model(
            pretrained_model,
            pretrained=True,
            # features_only=True,
            num_classes=0,  # remove classifier nn.Linear
        )
        self.model = self.model.eval()
        self.transforms = timm.data.create_transform(
            **timm.data.resolve_model_data_config(self.model),
            is_training=False
        )

        super().__init__(
            dataset=dataset,
            cache_root=store_path, 
            transform_id=pretrained_model.replace(".", "_"), 
        )

    def features_from_sample(self, x: torch.Tensor) -> torch.Tensor:
        num_ch = x.shape[0] if len(x.shape) == 3 else x.shape[1]
        if num_ch > 3:
            x_in = x.to(self.device).reshape(-1, *x.shape[-2:])
            x_in = x_in.unsqueeze(1).repeat(1, 3, *([1] * len(x.shape[-2:])))
        else:
            x_in = x.to(self.device)

        y = self.transforms(x_in)
        if len(y.shape) == 3:
            y = y.unsqueeze(0)

        with torch.no_grad():
            y = self.model(y)

        return y.flatten() if len(x.shape) == 3 else y.reshape(x.shape[0], -1)

    def to(self, device) -> Self:
        self.device = device
        self.model.to(device)
        return self
