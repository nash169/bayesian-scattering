import torch.nn as nn
from typing import List, Optional


class MLP(nn.Module):
    def __init__(
        self,
        layers: List,
        activation: Optional[nn.Module] = None,
        features: Optional[nn.Module] = None
    ):
        super().__init__()

        module_list = nn.ModuleList()
        if features is not None:
            module_list.append(features)
        for i, _ in enumerate(layers[:-2]):
            module_list.append(nn.Linear(layers[i], layers[i + 1]))
            if activation is not None:
                module_list.append(activation)
        module_list.append(nn.Linear(layers[-2], layers[-1]))

        self.net = nn.Sequential(*(module_list[i] for i in range(len(module_list))))

    def forward(self, x):
        return self.net(x)
