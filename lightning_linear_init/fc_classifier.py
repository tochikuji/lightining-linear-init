from typing import Iterable, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning import LightningModule
from torchmetrics import Accuracy, F1


class LitFCClassifier(LightningModule):
    def __init__(self,
                 units: Iterable[int],
                 activation_fun: Callable = F.relu,
                 output_fun: Callable = F.log_softmax,
                 optimal_init: bool = True):

        super().__init__()

        self.units = units
        self.n_layers = len(units) - 1
        self.activation_fun = activation_fun
        self.output_fun = output_fun

        for i in range(self.n_layers):
            d_in = units[i]
            d_out = units[i + 1]

            setattr(self, f'fc{i + 1}', nn.Linear(d_in, d_out))

            if optimal_init:
                init_fun = self._init_fun(activation_fun)
                init_fun(getattr(self, f'fc{i + 1}').weight)

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.train_f1 = F1(num_classes=units[-1])
        self.val_f1 = F1(num_classes=units[-1])

    def forward(self, x: torch.Tensor, layer_name: Optional[str] = None):
        layers = [f'fc{i + 1}' for i in range(self.n_layers)]

        h = x.flatten(start_dim=1)

        for layer in layers[:-1]:
            fc_layer: nn.Module = getattr(self, layer)
            h = fc_layer(h)
            if layer_name == layer:
                return h

            h = self.activation_fun(h, inplace=True)

        softmax_layer: nn.Module = getattr(self, layers[-1])

        h = softmax_layer(h)
        h = self.output_fun(h, dim=1)

        return h

    def training_step(self, batch, batch_idx):
        x, t = batch
        y = self.forward(x)
        pred = torch.argmax(y, dim=1)

        loss = F.nll_loss(y, t)
        acc = self.train_acc(pred, t)

        self.log('train_loss', loss)
        self.log('train_acc', acc)

        return loss

    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self.forward(x)
        pred = torch.argmax(y, dim=1)

        loss = F.nll_loss(y, t)
        acc = self.val_acc(pred, t)

        self.log('val_loss', loss)
        self.log('val_acc', acc)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.002)

    def _init_fun(self, activation_fun: str) -> Callable[[nn.Parameter], None]:
        if activation_fun == F.relu or activation_fun == F.relu6:
            return nn.init.kaiming_normal_
        else:
            return nn.init.xavier_normal_


if __name__ == '__main__':
    xs = torch.randn(1000, 100)
    model = LitFCClassifier([100, 50, 10])
    y = model(xs)

    assert list(y.shape) == [1000, 10]
