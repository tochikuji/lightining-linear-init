from typing import Callable, Iterable, Optional, List

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy, F1

from lightning_linear_init.utils import to_numpy, to_tensor, trap_input


class gMLPPCICallback(pl.Callback):
    def __init__(self, extractor, target_layer: List[nn.Module], target_inputs_name: List[str], adjust_std=True, verbose=True,
                extractor_opts={}):
        self.extractor = extractor
        self.target_inputs_name = target_inputs_name
        self.target_layer = target_layer
        self.verbose = verbose
        self.adjust_std = adjust_std
        self.extractor_opts = extractor_opts

    def on_train_start(self, trainer, module: pl.LightningModule):
        datamodule = trainer.datamodule
        train_loader: data.DataLoader = datamodule.train_dataloader()
        train_dataset = train_loader.dataset
        onetime_loader = data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)

        X, y = iter(onetime_loader).next()

        module.eval()
        
        with torch.no_grad():
            for inputs_name, layer in zip(self.target_inputs_name, self.target_layer):
                in_feature = module.forward(to_tensor(X).to(module.device), layer=inputs_name)
                in_feature = to_numpy(in_feature)

                y_ = to_numpy(y)

                support_dim = min(layer.in_features, layer.out_features)
                extractor = self.extractor(n_components=support_dim)

                extractor.fit(in_feature, y_, **self.extractor_opts)

                init_weight = extractor.components_ # (d_out, d_in)

                if self.adjust_std:
                    init_weight = init_weight / init_weight.std() * np.sqrt(2 / init_weight.shape[1])

                init_weight = to_tensor(init_weight).to(module.device)

                layer.weight.data[:support_dim, :] = init_weight

                if self.verbose:
                    print(f'LinInit for layer "{layer}" with "({inputs_name})": (algorithm "{extractor}") shape {init_weight.shape}')
