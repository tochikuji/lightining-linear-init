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


class DCICallback(pl.Callback):
    def __init__(self, extractor,
                 target_layer: nn.Module,
                 adjust_std=True,
                 verbose=True,
                 weights_name='coef_',
                 extractor_opts={},
                 fit_opts={}):
        self.extractor = extractor
        self.target_layer = target_layer
        self.verbose = verbose
        self.adjust_std = adjust_std
        self.extractor_opts = extractor_opts
        self.weights_name = weights_name

    def on_train_start(self, trainer, module):
        datamodule = trainer.datamodule
        train_loader: data.DataLoader = datamodule.train_dataloader()
        train_dataset = train_loader.dataset
        onetime_loader = data.DataLoader(
            train_dataset, batch_size=500, shuffle=False, )

        X, y = iter(onetime_loader).next()

        Xs = []
        ys = []

        for X, y in onetime_loader:
            in_feature = trap_input(module, X, self.target_layer)
            in_feature = to_numpy(in_feature)
            y_ = to_numpy(y)
            Xs.append(in_feature)
            ys.append(y_)

        in_feature = np.concatenate(Xs)
        y_ = np.concatenate(ys)

        support_dim = min(self.target_layer.in_features,
                          self.target_layer.out_features)
        extractor = self.extractor(**self.extractor_opts)

        extractor.fit(in_feature, y_)
        init_weight = getattr(extractor, self.weights_name)  # (d_out, d_in)

        if self.adjust_std:
            init_weight = init_weight / init_weight.std() * np.sqrt(2 / init_weight.shape[1])

        init_weight = to_tensor(init_weight).to('cpu')

        self.target_layer.weight.data[:support_dim, :] = init_weight

        if self.verbose:
            print(
                f'DCI for layer "{self.target_layer}" : (algorithm "{extractor}") shape {init_weight.shape}')
