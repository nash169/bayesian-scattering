import torch
from torchvision.utils import _log_api_usage_once


class Grayscale(torch.nn.Module):
    def __init__(self, num_output_channels=1):
        super().__init__()
        _log_api_usage_once(self)
        self.num_output_channels = num_output_channels

    def forward(self, img):
        return img.reshape(-1, *img.shape[-2:]).unsqueeze(1).repeat(1, 3, *([1] * len(img.shape[-2:])))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
