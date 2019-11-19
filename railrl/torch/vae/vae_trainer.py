from __future__ import print_function

import torch
import torch.utils.data
from torch import nn, optim
from railrl.core import logger
import numpy as np
import psutil
import os
from railrl.misc.ml_util import ConstantSchedule
import railrl.torch.pytorch_util as ptu

class VAETrainer():
    def __init__(
            self,
            train_dataset,
            test_dataset,
            model,
            batch_size=128,
            beta=0.5,
            beta_schedule=None,
            lr=1e-3,
            extra_recon_logging=dict(),
            recon_weights=None,
            recon_loss_type='mse',
            **kwargs
    ):
        assert recon_loss_type in ['mse', 'wse']
        self.batch_size = batch_size
        self.beta = beta
        self.beta_schedule = beta_schedule
        if self.beta_schedule is None:
            self.beta_schedule = ConstantSchedule(self.beta)

        if ptu.gpu_enabled():
            model.cuda()

        self.model = model
        self.representation_size = model.representation_size

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.train_dataset, self.test_dataset = train_dataset, test_dataset
        assert self.train_dataset['next_obs'].dtype == np.float32
        assert self.test_dataset['next_obs'].dtype ==np.float32
        assert self.train_dataset['obs'].dtype == np.float32
        assert self.test_dataset['obs'].dtype == np.float32
        self.normalize = model.normalize
        self.mse = nn.MSELoss()

        if self.normalize:
            self.train_data_mean = ptu.np_to_var(np.mean(self.train_dataset['next_obs'], axis=0))
            np_std = np.std(self.train_dataset['next_obs'], axis=0)
            for i in range(len(np_std)):
                if np_std[i] < 1e-3:
                    np_std[i] = 1.0
            self.train_data_std = ptu.np_to_var(np_std)

            self.model.train_data_mean = self.train_data_mean
            self.model.train_data_std = self.train_data_std

        self.extra_recon_logging = extra_recon_logging
        self.recon_weights = recon_weights
        self.recon_loss_type = recon_loss_type

    def get_batch(self, train=True):
        dataset = self.train_dataset if train else self.test_dataset
        ind = np.random.randint(0, len(dataset['obs']), self.batch_size)
        samples_obs = dataset['obs'][ind, :]
        samples_actions = dataset['actions'][ind, :]
        samples_next_obs = dataset['next_obs'][ind, :]
        return {
            'obs': ptu.np_to_var(samples_obs),
            'actions': ptu.np_to_var(samples_actions),
            'next_obs': ptu.np_to_var(samples_next_obs),
        }

    def logprob(self, recon_x, x, normalize=None, idx=None, unorm_weights=None):
        if normalize is None:
            normalize = self.normalize
        if normalize:
            x = (x - self.train_data_mean) / self.train_data_std
            recon_x = (recon_x - self.train_data_mean) / self.train_data_std
        if idx is not None:
            x = x[:,idx]
            recon_x = recon_x[:,idx]
            if unorm_weights is not None:
                unorm_weights = unorm_weights[idx]
        if unorm_weights is not None:
            dim = x.shape[1]
            norm_weights = unorm_weights / (np.sum(unorm_weights) / dim)
            norm_weights = ptu.np_to_var(norm_weights)
            recon_x = recon_x * norm_weights
            x = x * norm_weights

        return self.mse(recon_x, x)

    def kl_divergence(self, mu, logvar):
        kl = - torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        return kl

    def train_epoch(self, epoch, batches=100):
        self.model.train()
        losses = []
        kles = []
        mses = []
        beta = self.beta_schedule.get_value(epoch)
        for batch_idx in range(batches):
            data = self.get_batch()
            obs = data['obs']
            next_obs = data['next_obs']
            actions = data['actions']
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(next_obs)
            mse = self.logprob(recon_batch, next_obs)
            kle = self.kl_divergence(mu, logvar)
            if self.recon_loss_type == 'mse':
                loss = mse + beta * kle
            elif self.recon_loss_type == 'wse':
                wse = self.logprob(recon_batch, next_obs, unorm_weights=self.recon_weights)
                loss = wse + beta * kle
            loss.backward()

            losses.append(loss.data[0])
            mses.append(mse.data[0])
            kles.append(kle.data[0])

            self.optimizer.step()

        logger.record_tabular("train/epoch", epoch)
        logger.record_tabular("train/MSE", np.mean(mses))
        logger.record_tabular("train/KL", np.mean(kles))
        logger.record_tabular("train/loss", np.mean(losses))

    def test_epoch(self, epoch, save_vae=True, **kwargs):
        self.model.eval()
        losses = []
        kles = []
        zs = []

        recon_logging_dict = {
            'MSE': [],
            'WSE': [],
        }
        for k in self.extra_recon_logging:
            recon_logging_dict[k] = []

        beta = self.beta_schedule.get_value(epoch)
        for batch_idx in range(100):
            data = self.get_batch(train=False)
            obs = data['obs']
            next_obs = data['next_obs']
            actions = data['actions']
            recon_batch, mu, logvar = self.model(next_obs)
            mse = self.logprob(recon_batch, next_obs)
            wse = self.logprob(recon_batch, next_obs, unorm_weights=self.recon_weights)
            for k, idx in self.extra_recon_logging.items():
                recon_loss = self.logprob(recon_batch, next_obs, idx=idx)
                recon_logging_dict[k].append(recon_loss.data[0])
            kle = self.kl_divergence(mu, logvar)
            if self.recon_loss_type == 'mse':
                loss = mse + beta * kle
            elif self.recon_loss_type == 'wse':
                loss = wse + beta * kle
            z_data = ptu.get_numpy(mu.cpu())
            for i in range(len(z_data)):
                zs.append(z_data[i, :])
            losses.append(loss.data[0])
            recon_logging_dict['WSE'].append(wse.data[0])
            recon_logging_dict['MSE'].append(mse.data[0])
            kles.append(kle.data[0])
        zs = np.array(zs)
        self.model.dist_mu = zs.mean(axis=0)
        self.model.dist_std = zs.std(axis=0)

        for k in recon_logging_dict:
            logger.record_tabular("/".join(["test", k]), np.mean(recon_logging_dict[k]))
        logger.record_tabular("test/KL", np.mean(kles))
        logger.record_tabular("test/loss", np.mean(losses))
        logger.record_tabular("beta", beta)

        process = psutil.Process(os.getpid())
        logger.record_tabular("RAM Usage (Mb)", int(process.memory_info().rss / 1000000))

        num_active_dims = 0
        for std in self.model.dist_std:
            if std > 0.15:
                num_active_dims += 1
        logger.record_tabular("num_active_dims", num_active_dims)

        logger.dump_tabular()
        if save_vae:
            logger.save_itr_params(epoch, self.model, prefix='vae', save_anyway=True)  # slow...