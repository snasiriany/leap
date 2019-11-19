from __future__ import print_function

import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision.utils import save_image
from railrl.misc.ml_util import ConstantSchedule
from railrl.torch import pytorch_util as ptu
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler, BatchSampler
from railrl.core import logger
import os.path as osp
import numpy as np
import psutil
import os
from multiworld.core.image_env import normalize_image
from railrl.core.serializable import Serializable

class ConvVAETrainer(Serializable):
    def __init__(
            self,
            train_dataset,
            test_dataset,
            model,
            batch_size=128,
            beta=0.5,
            beta_schedule=None,
            lr=None,
            linearity_weight=0.0,
            use_linear_dynamics=False,
            noisy_linear_dynamics=False,
            scale_linear_dynamics=False,
            use_parallel_dataloading=True,
            train_data_workers=2,
    ):
        self.quick_init(locals())
        self.batch_size = batch_size
        self.beta = beta
        if lr is None:
            lr = 1e-3
        self.beta_schedule = beta_schedule
        if self.beta_schedule is None:
            self.beta_schedule = ConstantSchedule(self.beta)
        self.imsize = model.imsize

        if ptu.gpu_enabled():
            model.cuda()

        self.model = model
        self.representation_size = model.representation_size
        self.input_channels = model.input_channels
        self.imlength = model.imlength

        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.train_dataset, self.test_dataset = train_dataset, test_dataset
        assert self.train_dataset['next_obs'].dtype == np.uint8
        assert self.test_dataset['next_obs'].dtype == np.uint8
        assert self.train_dataset['obs'].dtype == np.uint8
        assert self.test_dataset['obs'].dtype == np.uint8

        self.use_parallel_dataloading = use_parallel_dataloading
        self.train_data_workers = train_data_workers
        self.gaussian_decoder_loss = self.model.gaussian_decoder
        if use_parallel_dataloading:
            self.train_dataset_pt = ImageDataset(
                train_dataset,
                should_normalize=True
            )
            self.test_dataset_pt = ImageDataset(
                test_dataset,
                should_normalize=True
            )

            self._train_weights = None
            base_sampler = InfiniteRandomSampler(self.train_dataset)
            self.train_dataloader = DataLoader(
                self.train_dataset_pt,
                sampler=BatchSampler(
                    base_sampler,
                    batch_size=batch_size,
                    drop_last=False,
                ),
                num_workers=train_data_workers,
                pin_memory=True,
            )
            self.test_dataloader = DataLoader(
                self.test_dataset_pt,
                sampler=BatchSampler(
                    InfiniteRandomSampler(self.test_dataset),
                    batch_size=batch_size,
                    drop_last=False,
                ),
                num_workers=0,
                pin_memory=True,
            )
            self.train_dataloader = iter(self.train_dataloader)
            self.test_dataloader = iter(self.test_dataloader)

        self.linearity_weight = linearity_weight
        self.use_linear_dynamics = use_linear_dynamics
        self.noisy_linear_dynamics = noisy_linear_dynamics
        self.scale_linear_dynamics = scale_linear_dynamics
        self.vae_logger_stats_for_rl = {}
        self._extra_stats_to_log = None

    def _kl_np_to_np(self, np_imgs):
        torch_input = ptu.np_to_var(normalize_image(np_imgs))
        mu, log_var = self.model.encode(torch_input)
        return ptu.get_numpy(
            - torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        )

    def set_vae(self, vae):
        self.model = vae
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def get_batch(self, train=True):
        if self.use_parallel_dataloading:
            if not train:
                dataloader = self.test_dataloader
            else:
                dataloader = self.train_dataloader
            samples = next(dataloader)
            return {
                'obs': ptu.Variable(samples[0][0]),
                'actions': ptu.Variable(samples[1][0]),
                'next_obs': ptu.Variable(samples[2][0]),
            }

        dataset = self.train_dataset if train else self.test_dataset
        ind = np.random.randint(0, len(dataset), self.batch_size)
        samples = normalize_image(dataset[ind, :])
        return ptu.np_to_var(samples)

    def logprob_iwae(self, recon_x, x):
        if self.gaussian_decoder_loss:
            error = -(recon_x - x) ** 2
        else:
            error = x * torch.log(torch.clamp(recon_x, min=1e-30)) \
                    + (1-x) * torch.log(torch.clamp(1-recon_x, min=1e-30))
        return error

    def logprob_vae(self, recon_x, x):
        batch_size = recon_x.shape[0]

        if self.gaussian_decoder_loss:
            return -((recon_x - x) ** 2).sum() / batch_size
        else:
            # Divide by batch_size rather than setting size_average=True because
            # otherwise the averaging will also happen across dimension 1 (the
            # pixels)
            return -F.binary_cross_entropy(
                recon_x,
                x.narrow(start=0, length=self.imlength,
                         dimension=1).contiguous().view(-1, self.imlength),
                size_average=False,
            ) / batch_size

    def kl_divergence(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def compute_vae_loss(self, x_recon, x, z_mu, z_logvar, z_sampled, beta):
        batch_size = x_recon.shape[0]
        k = x_recon.shape[1]

        x_recon = x_recon.view((batch_size*k, -1))
        x = x.view((batch_size * k, -1))
        z_mu = z_mu.view((batch_size * k, -1))
        z_logvar = z_logvar.view((batch_size * k, -1))

        de = -self.logprob_vae(x_recon, x)
        kle = self.kl_divergence(z_mu, z_logvar)
        loss = de + beta*kle
        return loss, de, kle

    def compute_iwae_loss(self, x_recon, x, z_mu, z_logvar, z_sampled, beta):
        batch_size = x_recon.shape[0]
        log_p_xgz = self.logprob_iwae(x_recon, x).sum(dim=-1)

        prior_dist = torch.distributions.Normal(ptu.Variable(torch.zeros(z_sampled.shape)),
                                           ptu.Variable(torch.ones(z_sampled.shape)))
        log_p_z = prior_dist.log_prob(z_sampled).sum(dim=-1)

        z_std = torch.exp(0.5*z_logvar)
        encoder_dist = torch.distributions.Normal(z_mu, z_std)
        log_q_zgx = encoder_dist.log_prob(z_sampled).sum(dim=-1)

        log_w = log_p_xgz + beta * (log_p_z - log_q_zgx)
        w_tilde = F.softmax(log_w, dim=-1).detach()

        loss = -(log_w * w_tilde).sum() / batch_size

        return loss

    def state_linearity_loss(self, obs, next_obs, actions):
        latent_obs_mu, latent_obs_logvar = self.model.encode(obs)
        latent_next_obs_mu, latent_next_obs_logvar = self.model.encode(next_obs)
        if self.noisy_linear_dynamics:
            latent_obs = self.model.reparameterize(latent_obs_mu, latent_obs_logvar)
        else:
            latent_obs = latent_obs_mu
        action_obs_pair = torch.cat([latent_obs, actions], dim=1)
        prediction = self.model.linear_constraint_fc(action_obs_pair)
        if self.scale_linear_dynamics:
            std = latent_next_obs_logvar.mul(0.5).exp_()
            scaling = 1 / std
        else:
            scaling = 1.0
        return torch.norm(scaling*(prediction - latent_next_obs_mu)) ** 2 / self.batch_size

    def train_epoch(self, epoch, batches=100):
        self.model.train()
        vae_losses = []
        losses = []
        des = []
        kles = []
        linear_losses = []
        beta = float(self.beta_schedule.get_value(epoch))
        for batch_idx in range(batches):
            data = self.get_batch()
            obs = data['obs']
            next_obs = data['next_obs']
            actions = data['actions']
            self.optimizer.zero_grad()
            x_recon, z_mu, z_logvar, z = self.model(next_obs)
            batch_size = x_recon.shape[0]
            k = x_recon.shape[1]
            x = next_obs.view((batch_size, 1, -1)).repeat(torch.Size([1, k, 1]))
            vae_loss, de, kle = self.compute_vae_loss(x_recon, x, z_mu, z_logvar, z, beta)
            loss = vae_loss
            if self.use_linear_dynamics:
                linear_dynamics_loss = self.state_linearity_loss(
                    obs, next_obs, actions
                )
                loss += self.linearity_weight * linear_dynamics_loss
                linear_losses.append(linear_dynamics_loss.data[0])
            loss.backward()

            vae_losses.append(vae_loss.data[0])
            losses.append(loss.data[0])
            des.append(de.data[0])
            kles.append(kle.data[0])

            self.optimizer.step()

        logger.record_tabular("train/epoch", epoch)
        logger.record_tabular("train/decoder_loss", np.mean(des))
        logger.record_tabular("train/KL", np.mean(kles))
        if self.use_linear_dynamics:
            logger.record_tabular("train/linear_loss",
                                  np.mean(linear_losses))
        logger.record_tabular("train/vae_loss", np.mean(vae_losses))
        logger.record_tabular("train/loss", np.mean(losses))

    def test_epoch(self, epoch, save_reconstruction=True, save_interpolation=True, save_vae=True):
        self.model.eval()
        vae_losses = []
        iwae_losses = []
        losses = []
        des = []
        kles = []
        linear_losses = []
        zs = []
        beta = float(self.beta_schedule.get_value(epoch))

        for batch_idx in range(10):
            data = self.get_batch(train=False)
            obs = data['obs']
            next_obs = data['next_obs']
            actions = data['actions']
            x_recon, z_mu, z_logvar, z = self.model(next_obs, n_imp=25)
            x_recon = x_recon.detach()
            z_mu = z_mu.detach()
            z_logvar = z_logvar.detach()
            z = z.detach()
            batch_size = x_recon.shape[0]
            k = x_recon.shape[1]
            x = next_obs.view((batch_size, 1, -1)).repeat(torch.Size([1, k, 1]))
            vae_loss, de, kle = self.compute_vae_loss(x_recon, x, z_mu, z_logvar, z, beta)
            iwae_loss = self.compute_iwae_loss(x_recon, x, z_mu, z_logvar, z, beta)
            loss = vae_loss
            if self.use_linear_dynamics:
                linear_dynamics_loss = self.state_linearity_loss(
                    obs, next_obs, actions
                )
                loss += self.linearity_weight * linear_dynamics_loss
                linear_losses.append(linear_dynamics_loss.data[0])

            z_data = ptu.get_numpy(z_mu[:,0].cpu())
            for i in range(len(z_data)):
                zs.append(z_data[i, :])
            vae_losses.append(vae_loss.data[0])
            iwae_losses.append(iwae_loss.data[0])
            losses.append(loss.data[0])
            des.append(de.data[0])
            kles.append(kle.data[0])

            if batch_idx == 0 and save_reconstruction:
                n = min(data['next_obs'].size(0), 16)
                comparison = torch.cat([
                    data['next_obs'][:n].narrow(start=0, length=self.imlength, dimension=1)
                    .contiguous().view(
                        -1, self.input_channels, self.imsize, self.imsize
                    ),
                    x_recon[:,0].contiguous().view(
                        self.batch_size,
                        self.input_channels,
                        self.imsize,
                        self.imsize,
                    )[:n]
                ])
                save_dir = osp.join(logger.get_snapshot_dir(), 'r_%d.png' % epoch)
                save_image(comparison.data.cpu(), save_dir, nrow=n)

            if batch_idx == 0 and save_interpolation:
                n = min(data['next_obs'].size(0), 10)

                z1 = z_mu[:n,0]
                z2 = z_mu[n:2*n, 0]

                num_steps = 8

                z_interp = []
                for i in np.linspace(0.0, 1.0, num_steps):
                    z_interp.append(float(i) * z1 + float(1 - i) * z2)
                z_interp = torch.cat(z_interp)

                imgs = self.model.decode(z_interp)
                imgs = imgs.view((num_steps, n, 3, self.imsize, self.imsize))
                imgs = imgs.permute([1, 0, 2, 3, 4])
                imgs = imgs.contiguous().view((n * num_steps, 3, self.imsize, self.imsize))

                save_dir = osp.join(logger.get_snapshot_dir(), 'i_%d.png' % epoch)
                save_image(
                    imgs.data,
                    save_dir,
                    nrow=num_steps,
                )

        zs = np.array(zs)
        self.model.dist_mu = zs.mean(axis=0)
        self.model.dist_std = zs.std(axis=0)

        logger.record_tabular("test/decoder_loss", np.mean(des))
        logger.record_tabular("test/KL", np.mean(kles))
        if self.use_linear_dynamics:
            logger.record_tabular("test/linear_loss", np.mean(linear_losses))
        logger.record_tabular("test/loss", np.mean(losses))
        logger.record_tabular("test/vae_loss", np.mean(vae_losses))
        logger.record_tabular("test/iwae_loss", np.mean(iwae_losses))
        logger.record_tabular("test/iwae_vae_diff", np.mean(np.array(iwae_losses) - np.array(vae_losses)))
        logger.record_tabular("beta", beta)

        process = psutil.Process(os.getpid())
        logger.record_tabular("RAM Usage (Mb)", int(process.memory_info().rss / 1000000))

        num_active_dims = 0
        num_active_dims2 = 0
        for std in self.model.dist_std:
            if std > 0.15:
                num_active_dims += 1
            if std > 0.05:
                num_active_dims2 += 1
        logger.record_tabular("num_active_dims", num_active_dims)
        logger.record_tabular("num_active_dims2", num_active_dims2)

        logger.dump_tabular()
        if save_vae:
            logger.save_itr_params(epoch, self.model, prefix='vae', save_anyway=True)  # slow...

class ImageDataset(Dataset):
    def __init__(self, images, should_normalize=True):
        super().__init__()
        self.dataset = images
        self.dataset_len = len(self.dataset['next_obs'])
        assert should_normalize == (images['next_obs'].dtype == np.uint8)
        self.should_normalize = should_normalize

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idxs):
        next_obs = self.dataset['next_obs'][idxs, :]
        actions = self.dataset['actions'][idxs, :]
        obs = self.dataset['obs'][idxs, :]
        if self.should_normalize:
            next_obs = normalize_image(next_obs)
            obs = normalize_image(obs)
        return np.float32(obs), np.float32(actions), np.float32(next_obs)

class InfiniteRandomSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.iter = iter(torch.randperm(len(self.data_source['next_obs'])).tolist())

    def __iter__(self):
        return self

    def __next__(self):
        try:
            idx = next(self.iter)
        except StopIteration:
            self.iter = iter(torch.randperm(len(self.data_source['next_obs'])).tolist())
            idx = next(self.iter)
        return idx

    def __len__(self):
        return 2 ** 62