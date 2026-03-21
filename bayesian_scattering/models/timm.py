import torch
import torch.nn as nn
import timm

from typing import Optional, Self, List
from bayesian_scattering.models.mlp import MLP


class TIMMRegression(nn.Module):
    def __init__(
        self,
        num_ch: int,
        layers: List,
        features_model='convnext_base.fb_in1k',
        **kwargs
    ):
        super().__init__()

        self.features = timm.create_model(
            features_model,
            pretrained=False,
            in_chans=num_ch,
            num_classes=0,
        )
        self.transforms = timm.data.create_transform(
            **timm.data.resolve_model_data_config(self.features),
            is_training=False
        )

        self.model = MLP(
            layers=[self.features.num_features] + layers,
            **kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.features(x))
