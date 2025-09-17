# src/model/Regressor.py
from torch import nn

class Regressor(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int] = [128, 64]):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, 1)]  # salida escalar para regresión
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
