from typing import Union, Collection
import numpy as np

import torch
import torch.nn as nn

import pytorch_lightning as pl

DimArray = Collection[int]


def trap_input(model: nn.Module, x, layer: nn.Module):
    activation = []

    def hook(model, input, output):
        activation.append(input[0].detach())

    handle = layer.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        model.forward(x)
    handle.remove()

    return activation[0]


def trap_output(model: nn.Module, x, layer: nn.Module):
    activation = []

    def hook(model, input, output):
        activation.append(output[0].detach())

    handle = layer.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        model.forward(x)

    handle.remove()

    return activation[0]


def to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.to('cpu').detach().numpy()
    else:
        return np.asarray(x, dtype=np.float32)


def to_tensor(x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    # elif isinstance(x, np.ndarray):
    else:
        return torch.from_numpy(x.astype(np.float32))


class LogCallback(pl.Callback):
    def __init__(self, **kargs):
        super().__init__()
        self.param = kargs

    def on_train_start(self, trainer, module):
        trainer.logger.log_hyperparams(self.param)


def divide_by_mask(x, mask: DimArray):
    current_index = 0
    pool = []

    for d in mask:
        pool.append(x[:, current_index: current_index + d])
        current_index += d

    return pool


def separate_into_tensor(xs):
    return [torch.from_numpy(x) for x in xs]
