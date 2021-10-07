from typing import List, Iterable, Sized, Union, Collection

import torch
import torch.nn as nn

import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from pytorch_lightning import LightningModule
from lightning_linear_init.utils import divide_by_mask, separate_into_tensor

import numpy

DimArray = Collection[int]


class LitSparseAE(LightningModule):
    '''
    Sparse AE that has the dimensionality
    [d1, d2,..., dn] -(Sparse)-> [m1, m2,..., mn] -> [d_intr]
    -> [m1, m2,..., mn] -(Sparse)-> [d1, d2,..., dn]
    '''

    def __init__(self, d_ins: DimArray, d_intrs: DimArray, d_reduced: int):
        super().__init__()

        self.encoder = SparseEncoder(d_ins, d_intrs, d_reduced)
        self.decoder = SparseDecoder(d_reduced, d_intrs, d_ins)

        self.n_blocks = len(d_ins)
        self.d_reduced = d_reduced
        self.d_concat_intrs = numpy.sum(d_intrs)
        self.d_in = numpy.sum(d_ins)

    def encode(self, xs):
        return self.encoder(xs)

    def decode(self, ys):
        return self.decoder(ys)

    def forward(self, xs):
        ys = F.relu(self.encode(xs), inplace=True)
        return self.decode(ys)

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

        self.log('val_loss', loss.detach().float(), on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.02)


class SparseEncoder(nn.Module):
    def __init__(self, d_ins: DimArray, d_intrs: DimArray, d_reduced: int):
        super().__init__()

        blocks = []
        self.n_blocks = len(d_ins)

        for d_in, d_intr in zip(d_ins, d_intrs):
            module = FcModule(d_in, d_intr)
            blocks.append(module)

        self.blocks = nn.ModuleList(blocks)

        self.d_intrs = d_intrs
        self.d_concat_intrs = numpy.sum(d_intrs)
        self.d_reduced = d_reduced
        self.d_ins = d_ins

        self.transformer = FcModule(self.d_concat_intrs, self.d_reduced)

    def forward(self, xs):
        """

        :param xs: has shape (N, D) or (X, [d1, ..., dm])
        :param mask:
        :return:
        """

        # # the case that xs is list of already separated list
        # if mask is None:
        #     xs_ = separate_into_tensor(xs)
        # # the case that xs is unseparated but mask given
        # else:
        xs_ = divide_by_mask(xs, self.d_ins)

        intr_pool = []
        for x, fc in zip(xs_, self.blocks):
            y = fc(x)
            y = F.relu(y, inplace=True)

            intr_pool.append(y)

        y_intr = torch.cat(intr_pool, dim=1)

        y = self.transformer(y_intr)

        return y


class SparseDecoder(nn.Module):
    def __init__(self, d_reduced, d_intrs: DimArray, d_outs: DimArray):
        super().__init__()

        blocks = []
        self.n_blocks = len(d_outs)

        self.d_concat_intrs = numpy.sum(d_intrs)
        self.d_reduced = d_reduced
        self.d_intrs = d_intrs

        self.transformer = FcModule(self.d_reduced, self.d_concat_intrs)

        for d_out, d_intr in zip(d_outs, d_intrs):
            module = FcModule(d_intr, d_out)
            blocks.append(module)

        self.blocks = nn.ModuleList(blocks)

    def forward(self, xs, relu=False):
        ys = self.transformer(xs)
        ys = F.relu(ys, inplace=True)

        ys = divide_by_mask(ys, self.d_intrs)

        intr_pool = []
        for y, fc in zip(ys, self.blocks):
            y = fc(y)

            intr_pool.append(y)

        ys = torch.cat(intr_pool, dim=1)
        if relu:
            ys = F.relu(ys, inplace=True)

        return ys


class FcModule(nn.Module):
    def __init__(self, d_in, d_intr):
        super().__init__()

        self.fc1 = nn.Linear(d_in, d_intr)

    def forward(self, x):
        x = self.fc1(x)

        return x


if __name__ == '__main__':
    ae = LitAutoEncoder([100, 50, 10])
    module = LitAEClasiffier(ae, [10, 5])

    xs = torch.rand(10, 100)

    y = module(xs)

    assert list(y.shape) == [10, 5]
