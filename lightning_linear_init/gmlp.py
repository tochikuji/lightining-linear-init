from typing import Callable, Iterable, Optional
import re

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy, F1

from g_mlp_pytorch import gMLPVision
from g_mlp_pytorch.g_mlp_pytorch import dropout_layers

from einops.layers.torch import Rearrange, Reduce


class LitgMLP(pl.LightningModule):
    def __init__(
            self,
            image_size,
            patch_size,
            num_classes,
            dim,
            depth,
            ff_mult=4,
            channels=3,
            attn_dim=None,
            prob_survival=1.0,
            task='default', mode='default'):

        super().__init__()

        self.backbone = gMLPVision(image_size=image_size, patch_size=patch_size, num_classes=num_classes,
                                  dim=dim, depth=depth, ff_mult=ff_mult, channels=channels, attn_dim=attn_dim, prob_survival=prob_survival)

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.train_f1 = F1(num_classes=num_classes)
        self.val_f1 = F1(num_classes=num_classes)

        self.to_patch_embed = self.backbone.to_patch_embed
        self.layers: Iterable[nn.Module] = self.backbone.layers
        self.to_logits = self.backbone.to_logits
        self.prob_survival = prob_survival
        self.attn_dim = attn_dim

    def forward(self, x: torch.Tensor, layer=''):
        y = x

        # if patch embed
        y = self.to_patch_embed[0](y)
        if layer == 'patch_embed':
            y = Rearrange('n o i -> (n o) i')(y)
            return y

        y = self.to_patch_embed[1](y)

        mlp_match = re.match(r'mlp(\d+)_(in|out)', layer)

        if mlp_match:
            block_no, point = mlp_match.groups()
            block_no = int(block_no)
        else:
            block_no = np.infty
            point = 'out'

        #中で止めない場合
        if block_no + 1 > len(self.layers):
            # そのまま伝搬させる
#             y = self.layers(y)
            layers = self.layers if not self.training else dropout_layers(self.layers, self.prob_survival)
            y = nn.Sequential(*layers)(y)

        #中で止める場合
        else:
            for i, block in enumerate(self.layers):
                if block_no == i:
                    mlp_block = block.fn.fn

                    if point == 'in':
                        y = Rearrange('n o i -> (n o) i')(y)

                        return y

                    y = mlp_block.proj_in(y)
                    y = mlp_block.sgu(y)

                    if point == 'out':
                        y = Rearrange('n o i -> (n o) i')(y)

                        return y

                    y = mlp_block.proj_out(y)

                else:
                    y = block(y)

        if layer == 'to_logits':
            y = self.to_logits[0](y)
            y = self.to_logits[1](y)
            # y = Rearrange('n o i -> (n o) i')(y)

            return y
        else:
            y = self.to_logits(y)

        # y = self.backbone(x)

        y = F.log_softmax(y, dim=1)
        return y

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
        return torch.optim.AdamW(self.parameters(), lr=0.001)
