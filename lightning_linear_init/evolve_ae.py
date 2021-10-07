import numpy as np

import torch
import torch.nn as nn

from corpca import CorPCA

from lightning_linear_init.autoencoder import LitAutoEncoder
from lightning_linear_init.utils import to_numpy, to_tensor


class LitEvolveAE(LitAutoEncoder):
    def __init__(self, xs: torch.Tensor, ccr: float = 0.9,  enc_depth=None, intermediate_dim=None, extractor=CorPCA):
        base_layer = [nn.Flatten()]
        layers = []

        prev_xs = nn.Sequential(*base_layer)(xs)

        n_data, data_dim = prev_xs.shape
        layers.append(data_dim)

        weights = []

        i = 0
        end_flag = False

        while True:
            print(f'initializing {i + 1}th layer...')
            X = to_numpy(prev_xs)
            pca = extractor()
            pca.fit(X)
            result_ccr = np.cumsum(pca.explained_variance_ratio_)
            num_dim_ccr = np.where(result_ccr > ccr)[0].min()

            if intermediate_dim is not None:
                if num_dim_ccr <= intermediate_dim:
                    num_dim_ccr = intermediate_dim
                    end_flag = True

            layers.append(num_dim_ccr)

            sequential = []
            in_outs = np.lib.stride_tricks.sliding_window_view(layers, 2)
            for layer_io in in_outs[:-1]:
                d_in = layer_io[0]
                d_out = layer_io[1]

                sequential.append(nn.Linear(d_in, d_out))
                sequential.append(nn.ReLU())

            for i, w in enumerate(weights):
                print(
                    f'initializing {sequential[2 * i]} with weight w {w.shape}')
                sequential[2 * i].weight.detach().data = w

            d_in = in_outs[-1][0]
            d_out = in_outs[-1][1]

            current_init_weight = to_tensor(pca.components_[:, :num_dim_ccr]).T
            sequential.append(nn.Linear(d_in, d_out))
            sequential[-1].weight.detach().data = current_init_weight

            weights.append(current_init_weight)
            current_seq = nn.Sequential(*sequential)
            current_seq.eval()
            with torch.no_grad():
                prev_xs = current_seq(xs)

            i += 1

            if enc_depth is not None:
                if i >= enc_depth - 1:
                    break

            if end_flag:
                break

        print(f'resulting model: {current_seq}')

        super().__init__(layers)

        for i, w in zip(range(len(layers) - 1), weights):
            encoder = getattr(self, f'encoder{i + 1}')
            decoder = getattr(self, f'decoder{len(layers) - (i + 1)}')

            encoder.weight.data = w
            decoder.weight.data = w.T


if __name__ == '__main__':
    xs = torch.randn(1000, 100)
    ae = LitEvolveAE(xs, ccr=0.5, intermediate_dim=10)

    y = ae.encode(xs)
    ae(xs)
