from typing import List

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def make_mlp(
    dims: List, act: str, l_act: bool = False, bn: bool = True, dropout: float = 0.0
):
    """Create a simple MLP with batch-norm and dropout

    Args:
        dims: (List) a list containing the dimensions of MLP
        act: (str) activation function to be used. Valid activations are [relu, tanh, sigmoid]
        l_act: (bool) whether to use activation after the last linear layer
        bn: (bool) use batch-norm or not. Default is True
        dropout: (float) dropout percentage
    """
    layers = []
    activation = {
        "relu": nn.ReLU(inplace=True),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "elu": nn.ELU(),
    }[act.lower()]

    for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(in_dim, out_dim, bias=not bn))
        if i != (len(dims) - 2):
            if bn:
                layers.append(nn.BatchNorm1d(out_dim))

            layers.append(activation)

            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))

    if l_act:
        layers.append(activation)

    return nn.Sequential(*layers)


class VWBehaviorCloning(nn.Module):
    def __init__(
        self,
        dims=[512, 256, 128, 64, 2],
        act: str = 'relu', l_act: bool = False, bn: bool = False, dropout: float = 0.0
    ):

        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        encoder = resnet18(weights=weights)
        encoder.fc = nn.Identity()
        encoder.conv1 = nn.Conv2d(1, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.encoder = encoder
        self.fc = make_mlp(dims, act, l_act, bn, dropout)

    def forward(self, img):
        # print(f"{img.shape}")
        encoded = self.encoder(img)
        action = self.fc(encoded)
        return action
