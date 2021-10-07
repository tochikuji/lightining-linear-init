from typing import Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics import MeanSquaredError
from pytorch_lightning import LightningModule


class LitAutoEncoder(LightningModule):
    def __init__(self,
                 units: Iterable[int],
                 activation_fun: Callable = F.relu,
                 optimal_init: bool = True):

        super().__init__()

        self.units = units
        self.depth = len(units)
        self.n_layers = self.depth
        self.activation_fun = activation_fun

        for i in range(self.depth - 1):
            layer = nn.Linear(units[i], units[i + 1])
            if optimal_init:
                nn.init.kaiming_normal_(
                    layer.weight, mode='fan_in', nonlinearity='relu')
            setattr(self, f'encoder{i + 1}', layer)

        for i in range(self.depth - 1):
            layer = nn.Linear(units[self.depth - i - 1],
                              units[self.depth - i - 2])
            if optimal_init:
                nn.init.kaiming_normal_(
                    layer.weight, mode='fan_out', nonlinearity='relu')
            setattr(self, f'decoder{i + 1}', layer)

        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()

    def forward(self, x):
        x = self.activation_fun(self.encode(x))
        x = self.decode(x)

        return x

    def encode(self, x, layer=None):
        if layer is None:
            mark_pos = self.depth - 1
        else:
            mark_pos = layer

        encode_layers = [f'encoder{i + 1}' for i in range(mark_pos)]

        # flatten inputs to vector
        x = torch.flatten(x, start_dim=1, end_dim=-1)

        for layer in encode_layers[:-1]:
            x = getattr(self, layer)(x)
            x = F.relu(x)

        x = getattr(self, encode_layers[-1])(x)

        return x

    def decode(self, x):
        decode_layers = [f'decoder{i + 1}' for i in range(self.depth - 1)]

        for layer in decode_layers[:-1]:
            x = getattr(self, layer)(x)
            x = F.relu(x)

        x = getattr(self, decode_layers[-1])(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = torch.flatten(x, start_dim=1, end_dim=-1)

        recon = self.forward(x)

        loss = F.mse_loss(x, recon)

        self.log('train_loss', loss, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = torch.flatten(x, start_dim=1, end_dim=-1)

        recon = self.forward(x)

        loss = F.mse_loss(x, recon)

        self.log('val_loss', loss, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.002)


if __name__ == '__main__':
    ae = LitAutoEncoder([100, 50, 10])
    xs = torch.rand(10, 100)

    y = ae.encode(xs)

    print(y.shape)

    assert list(y.shape) == [10, 10]
