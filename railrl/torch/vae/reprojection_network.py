# Adapted from pytorch examples

from __future__ import print_function
from torch import nn, optim
from railrl.core import logger
import numpy as np
from railrl.pythonplusplus import identity
from railrl.torch.core import PyTorchModule
from railrl.torch.networks import Mlp
import railrl.torch.pytorch_util as ptu

class ReprojectionNetworkTrainer():
    def __init__(
            self,
            train_dataset,
            test_dataset,
            model,
            batch_size=128,
            log_interval=0,
            lr=1e-3,
            **kwargs
    ):
        self.log_interval = log_interval
        self.batch_size = batch_size

        if ptu.gpu_enabled():
            model.cuda()

        self.model = model
        self.representation_size = model.representation_size

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.train_dataset, self.test_dataset = train_dataset, test_dataset
        assert self.train_dataset['z'].dtype == np.float32
        assert self.test_dataset['z'].dtype ==np.float32
        assert self.train_dataset['z_proj'].dtype == np.float32
        assert self.test_dataset['z_proj'].dtype == np.float32
        self.mse = nn.MSELoss()

    def get_batch(self, train=True):
        dataset = self.train_dataset if train else self.test_dataset
        ind = np.random.randint(0, len(dataset['z']), self.batch_size)
        return {
            'z': ptu.np_to_var(dataset['z'][ind, :]),
            'z_proj': ptu.np_to_var(dataset['z_proj'][ind, :]),
        }

    def mse_loss(self, z_proj_hat, z_proj):
        return self.mse(z_proj_hat, z_proj)

    def train_epoch(self, epoch, batches=100):
        self.model.train()
        mses = []
        losses = []
        for batch_idx in range(batches):
            data = self.get_batch()
            z = data["z"]
            z_proj = data['z_proj']
            self.optimizer.zero_grad()
            z_proj_hat = self.model(z)
            mse = self.mse_loss(z_proj_hat, z_proj)
            loss = mse
            loss.backward()

            mses.append(mse.data[0])
            losses.append(loss.data[0])

            self.optimizer.step()

        logger.record_tabular("train/epoch", epoch)
        logger.record_tabular("train/MSE", np.mean(mses))
        logger.record_tabular("train/loss", np.mean(losses))

    def test_epoch(self, epoch, save_network=True, batches=100):
        self.model.eval()
        mses = []
        losses = []
        for batch_idx in range(batches):
            data = self.get_batch(train=False)
            z = data["z"]
            z_proj = data['z_proj']
            z_proj_hat = self.model(z)
            mse = self.mse_loss(z_proj_hat, z_proj)
            loss = mse

            mses.append(mse.data[0])
            losses.append(loss.data[0])

        logger.record_tabular("test/epoch", epoch)
        logger.record_tabular("test/MSE", np.mean(mses))
        logger.record_tabular("test/loss", np.mean(losses))

        logger.dump_tabular()
        if save_network:
            logger.save_itr_params(epoch, self.model, prefix='reproj', save_anyway=True)

class ReprojectionNetwork(PyTorchModule):
    def __init__(
            self,
            vae,
            hidden_sizes=list([64, 128, 64]),
            init_w=1e-3,
            hidden_init=ptu.fanin_init,
            output_activation=identity,
            layer_norm=False,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__()
        self.vae = vae
        self.representation_size = self.vae.representation_size
        self.hidden_init = hidden_init
        self.output_activation = output_activation
        # self.dist_mu = np.zeros(self.representation_size)
        # self.dist_std = np.ones(self.representation_size)
        self.dist_mu = self.vae.dist_mu
        self.dist_std = self.vae.dist_std
        self.relu = nn.ReLU()
        self.init_w = init_w
        hidden_sizes = list(hidden_sizes)
        self.network=Mlp(hidden_sizes,
                         self.representation_size,
                         self.representation_size,
                         layer_norm=layer_norm,
                         hidden_init=hidden_init,
                         output_activation=output_activation,
                         init_w=init_w)

    def forward(self, z):
        z = z.view(-1, self.representation_size)
        return self.network(z)

    def __getstate__(self):
        d = super().__getstate__()
        # Add these explicitly in case they were modified
        d["_dist_mu"] = self.dist_mu
        d["_dist_std"] = self.dist_std
        return d

    def __setstate__(self, d):
        super().__setstate__(d)
        self.dist_mu = d["_dist_mu"]
        self.dist_std = d["_dist_std"]