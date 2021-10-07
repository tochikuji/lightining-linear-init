from typing import Callable, Iterable, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy, F1

from lightning_linear_init.utils import to_numpy, to_tensor, trap_input


class AEPCICallback(pl.Callback):
    def __init__(self, extractor,
                 target='all',
                 adjust_std=True,
                 verbose=True,
                 whiten=False,
                 extractor_opts={}):
        self.extractor = extractor
        self.target = target
        self.verbose = verbose
        self.adjust_std = adjust_std
        self.whiten = whiten
        self.extractor_opts = extractor_opts

    def on_train_start(self, trainer, module):

        datamodule = trainer.datamodule
        train_loader = datamodule.train_dataloader()
        train_dataset = train_loader.dataset

        onetime_loader = data.DataLoader(
            train_dataset, batch_size=len(train_dataset), shuffle=False)
        X, y = iter(onetime_loader).next()
        X = torch.nn.Flatten()(X).to(module.device)

        units = module.units
        n_layers = module.n_layers

        for i in range(n_layers - 1):
            previous_num = i
            layer_num = i + 1
            decoder_num = n_layers - layer_num
            previous_layer_name = f'encoder{i}'
            target_layer_name = f'encoder{i + 1}'

            if i >= 1:
                previous_layer: nn.Linear = getattr(
                    module, previous_layer_name)
            target_layer: nn.Linear = getattr(module, target_layer_name)
            dim_out, dim_in = target_layer.weight.data.shape

            target_decoder = getattr(module, f'decoder{decoder_num}')

            if self.verbose:
                print(f'PCI for layer "{target_layer_name}""')

            if i == 0:
                X_ = to_numpy(X)
            else:
                X_ = to_tensor(X).to(module.device)
                X_ = trap_input(module, X_, target_layer)

                X_ = to_numpy(X_)

            y_ = to_numpy(y)

            extractor = self.extractor(
                n_components=dim_out, whiten=self.whiten)
            extractor.fit(X_, y_, **self.extractor_opts)

            init_weight = extractor.components_

            if self.adjust_std:
                init_weight = init_weight / init_weight.std() * np.sqrt(2 /
                                                                        init_weight.shape[1])

            target_layer.weight.data = to_tensor(init_weight).to(module.device)
            target_decoder.weight.data = to_tensor(
                init_weight.T).to(module.device)


if __name__ == '__main__':
    from corpca import CorPCA

    callback = AEPCICallback(CorPCA)
