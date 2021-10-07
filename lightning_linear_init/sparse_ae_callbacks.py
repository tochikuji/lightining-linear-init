import pytorch_lightning as pl
import numpy as np

from corpca import CorPCA
from lightning_linear_init.sparse_ae import LitSparseAE
from torch.utils.data import Dataset, DataLoader
from lightning_linear_init.utils import divide_by_mask, separate_into_tensor, to_numpy, to_tensor, trap_input


class SpAEPCICallback(pl.Callback):
    def __init__(self, extractor=CorPCA,
                 target='all',
                 adjust_std=True,
                 verbose=True,
                 gate_pci: str = 'pci',
                 decoder=True,
                 extractor_opts={}):
        self.extractor = extractor
        self.target = target
        self.verbose = verbose
        self.adjust_std = adjust_std
        self.extractor_opts = extractor_opts

        # 'pci', 'hpca', 'none'
        self.gate_pci = gate_pci

        self.decoder = decoder

    def on_train_start(self, trainer, module: LitSparseAE):

        datamodule = trainer.datamodule
        train_loader = datamodule.train_dataloader()
        train_dataset = train_loader.dataset

        onetime_loader = DataLoader(
            train_dataset, batch_size=len(train_dataset), shuffle=False)
        X_whole, y = iter(onetime_loader).next()

        d_ins = module.encoder.d_ins
        d_intrs = module.encoder.d_intrs
        d_reduced = module.encoder.d_reduced

        xs = divide_by_mask(X_whole, d_ins)

        intr_pca = []

        for x_block, d_intr, enc_block, dec_block in zip(xs, d_intrs, module.encoder.blocks, module.decoder.blocks):
            if self.verbose:
                print(f'PCI for blockwise layer {enc_block.fc1}...')

            x_np = to_numpy(x_block)
            dim_out, dim_in = enc_block.fc1.weight.data.shape

            extractor = self.extractor(n_components=dim_out)
            extractor.fit(x_np, **self.extractor_opts)

            if self.gate_pci == 'hpca':
                intr_pca.append(extractor)

            init_weight = extractor.components_

            if self.adjust_std:
                init_weight = init_weight / init_weight.std() * np.sqrt(2 /
                                                                        init_weight.shape[1])

            enc_block.fc1.weight.data = to_tensor(init_weight).detach()
            if self.decoder:
                dec_block.fc1.weight.data = to_tensor(init_weight.T).detach()

        module.to(module.device)

        if self.gate_pci == 'pci':
            X = to_tensor(X_whole).to(module.device)
            X = trap_input(module, X, module.encoder.transformer.fc1)

            X = to_numpy(X)

            extractor = self.extractor(n_components=d_reduced)
            extractor.fit(X, **self.extractor_opts)

            init_weight = extractor.components_

        elif self.gate_pci == 'hpca':
            ys = []
            for x_block, pca in zip(xs, intr_pca):
                y = pca.transform(x_block)
                ys.append(y)

            y_concat = np.concatenate(ys, axis=1)

            extractor = self.extractor(n_components=d_reduced)
            extractor.fit(y_concat, **self.extractor_opts)

            init_weight = extractor.components_

        elif self.gate_pci == 'none':
            pass

        else:
            raise ValueError()

        if self.gate_pci != 'none':
            if self.adjust_std:
                init_weight = init_weight / init_weight.std() * np.sqrt(2 /
                                                                        init_weight.shape[1])

            module.encoder.transformer.fc1.weight.data = to_tensor(
                init_weight).detach()
            if self.decoder:
                module.decoder.transformer.fc1.weight.data = to_tensor(
                    init_weight.T).detach()

        module.to(module.device)
