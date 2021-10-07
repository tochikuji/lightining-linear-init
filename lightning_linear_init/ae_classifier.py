from typing import Callable, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics import Accuracy, F1
from pytorch_lightning import LightningModule

from lightning_linear_init.autoencoder import LitAutoEncoder
from lightning_linear_init.fc_classifier import LitFCClassifier


class LitAEClasiffier(LightningModule):
    def __init__(self,
                 backbone: LitAutoEncoder,
                 units: Iterable[int],
                 activation_fun: Callable = F.relu,
                 output_fun: Callable = F.log_softmax,
                 optimal_init: bool = True):
        super().__init__()

        self.units = units
        self.n_layers = len(units) - 1
        self.activation_fun = activation_fun
        self.output_fun = output_fun

        self.backbone = backbone
        self.classifier = LitFCClassifier(
            units, activation_fun, output_fun, optimal_init)

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.train_f1 = F1(num_classes=units[-1])
        self.val_f1 = F1(num_classes=units[-1])

    def forward(self, x: torch.Tensor, layer_name: Optional[str] = None):
        h = self.backbone.encode(x)
        h = self.classifier(h)

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
    ae = LitAutoEncoder([100, 50, 10])
    module = LitAEClasiffier(ae, [10, 5])

    xs = torch.rand(10, 100)

    y = module(xs)

    assert list(y.shape) == [10, 5]
