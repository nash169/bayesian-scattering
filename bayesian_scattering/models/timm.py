import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from typing import List
from bayesian_scattering.models.mlp import MLP


class _TIMMBase(nn.Module):
    def _init_preprocessing(self, model: nn.Module):
        data_config = timm.data.resolve_model_data_config(model)
        self.input_size = data_config["input_size"][-2:]
        mean = torch.tensor(data_config["mean"], dtype=torch.float32)
        std = torch.tensor(data_config["std"], dtype=torch.float32)
        self.register_buffer("mean", mean.view(1, -1, 1, 1))
        self.register_buffer("std", std.view(1, -1, 1, 1))

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != self.input_size:
            x = F.interpolate(
                x,
                size=self.input_size,
                mode="bilinear",
                align_corners=False,
            )
        if x.shape[1] == self.mean.shape[1]:
            x = x.sub(self.mean).div(self.std)
        return x


class TIMMRegression(_TIMMBase):
    def __init__(
        self,
        num_ch: int,
        layers: List,
        features_model="convnext_base.fb_in1k",
        **kwargs,
    ):
        super().__init__()

        self.features = timm.create_model(
            features_model,
            pretrained=False,
            in_chans=num_ch,
            num_classes=0,
        )
        self._init_preprocessing(self.features)
        self.transforms = timm.data.create_transform(
            **timm.data.resolve_model_data_config(self.features), is_training=False
        )

        self.model = MLP(layers=[self.features.num_features] + layers, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._preprocess(x)
        return self.model(self.features(x))


class TIMMClassification(_TIMMBase):
    def __init__(
        self,
        num_ch: int,
        num_classes: int,
        features_model="convnext_base.fb_in1k",
        **kwargs,
    ):
        super().__init__()

        self.model = timm.create_model(
            features_model,
            pretrained=False,
            in_chans=num_ch,
            num_classes=num_classes,
        )
        self._init_preprocessing(self.model)
        self.transforms = timm.data.create_transform(
            **timm.data.resolve_model_data_config(self.model), is_training=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._preprocess(x)
        return self.model(x)
