from __future__ import print_function

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from railrl.pythonplusplus import identity
from railrl.torch import pytorch_util as ptu
import numpy as np
from railrl.torch.core import PyTorchModule

class ConvVAE(PyTorchModule):
    def __init__(
            self,
            representation_size,
            init_w=1e-3,
            input_channels=1,
            imsize=84,
            added_fc_size=0,
            hidden_init=ptu.fanin_init,
            output_activation=identity,
            min_variance=1e-4,
            use_min_variance=True,
            state_size=0,
            action_dim=None,
            large_arch=False,
            n_imp=1,
            gaussian_decoder=True,
            use_sigmoid_for_decoder=True,
            learned_decoder_variance=False,
    ):
        assert imsize in [48, 84]
        self.save_init_params(locals())
        super().__init__()
        self.representation_size = representation_size
        self.hidden_init = hidden_init
        self.output_activation = output_activation
        self.input_channels = input_channels
        self.imsize = imsize
        self.imlength = self.imsize ** 2 * self.input_channels
        if min_variance is None:
            self.log_min_variance = None
        else:
            self.log_min_variance = float(np.log(min_variance))
        self.dist_mu = np.zeros(self.representation_size)
        self.dist_std = np.ones(self.representation_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.added_fc_size = added_fc_size
        self.init_w = init_w
        self.large_arch = large_arch

        self.n_imp = n_imp # number of importance samples for IWAE objective

        self.gaussian_decoder = gaussian_decoder
        self.use_sigmoid_for_decoder = use_sigmoid_for_decoder
        self.learned_decoder_variance = learned_decoder_variance

        if imsize == 84:
            self.enc_conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=3)
            self.enc_bn1 = nn.BatchNorm2d(16)
            self.enc_conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=3)
            self.enc_bn2 = nn.BatchNorm2d(32)
            self.enc_conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=3)
            self.enc_bn3 = nn.BatchNorm2d(32)

            self.conv_output_dim = 32 * 4

            self.enc_fc_mu = nn.Linear(self.conv_output_dim, representation_size)
            self.enc_fc_logvar = nn.Linear(self.conv_output_dim, representation_size)

            self.dec_fc_sample = nn.Linear(representation_size, self.conv_output_dim)
            self.dec_conv3 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=3)
            self.dec_conv2 = nn.ConvTranspose2d(32, 16, kernel_size=6, stride=3)
            self.dec_conv1 = nn.ConvTranspose2d(16, input_channels, kernel_size=6, stride=3)
        elif imsize == 48:
            self.enc_conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=3)
            self.enc_bn1 = nn.BatchNorm2d(16)
            self.enc_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
            self.enc_bn2 = nn.BatchNorm2d(32)
            self.enc_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
            self.enc_bn3 = nn.BatchNorm2d(64)

            self.conv_output_dim = 64 * 9

            self.enc_fc_mu = nn.Linear(self.conv_output_dim, representation_size)
            self.enc_fc_logvar = nn.Linear(self.conv_output_dim, representation_size)

            self.dec_fc_sample = nn.Linear(representation_size, self.conv_output_dim)
            self.dec_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2)
            self.dec_conv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2)
            self.dec_conv1 = nn.ConvTranspose2d(16, input_channels, kernel_size=6, stride=3)

        if action_dim is not None:
            self.linear_constraint_fc = \
                nn.Linear(
                    self.representation_size + action_dim,
                    self.representation_size
                )
        else:
            self.linear_constraint_fc = None

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.hidden_init(self.enc_conv1.weight)
        self.enc_conv1.bias.data.fill_(0)
        self.hidden_init(self.enc_conv2.weight)
        self.enc_conv2.bias.data.fill_(0)
        self.hidden_init(self.enc_conv3.weight)
        self.enc_conv3.bias.data.fill_(0)

        self.hidden_init(self.dec_conv3.weight)
        self.dec_conv3.bias.data.fill_(0)
        self.hidden_init(self.dec_conv2.weight)
        self.dec_conv2.bias.data.fill_(0)
        self.hidden_init(self.dec_conv1.weight)
        self.dec_conv1.bias.data.fill_(0)

        self.hidden_init(self.enc_fc_mu.weight)
        self.enc_fc_mu.bias.data.fill_(0)
        self.enc_fc_mu.weight.data.uniform_(-init_w, init_w)
        self.enc_fc_mu.bias.data.uniform_(-init_w, init_w)
        self.hidden_init(self.enc_fc_logvar.weight)
        self.enc_fc_logvar.bias.data.fill_(0)
        self.enc_fc_logvar.weight.data.uniform_(-init_w, init_w)
        self.enc_fc_logvar.bias.data.uniform_(-init_w, init_w)

        self.dec_fc_sample.weight.data.uniform_(-init_w, init_w)
        self.dec_fc_sample.bias.data.uniform_(-init_w, init_w)

        if self.linear_constraint_fc is not None:
            self.linear_constraint_fc.weight.data.uniform_(-init_w, init_w)
            self.linear_constraint_fc.bias.data.uniform_(-init_w, init_w)

    def encode(self, input):
        input = input.view(-1, self.imlength + self.added_fc_size)
        conv_input = input.narrow(start=0, length=self.imlength, dimension=1)

        x = conv_input.contiguous().view(-1, self.input_channels, self.imsize, self.imsize)
        x = F.relu(self.enc_bn1(self.enc_conv1(x)))
        x = F.relu(self.enc_bn2(self.enc_conv2(x)))
        x = F.relu(self.enc_bn3(self.enc_conv3(x)))
        h = x.view(-1, self.conv_output_dim)
        if self.added_fc_size != 0:
            fc_input = input.narrow(start=self.imlength,
                                    length=self.added_fc_size, dimension=1)
            h = torch.cat((h, fc_input), dim=1)
        if self.large_arch:
            h = F.relu(self.enc_fc1(h))
        mu = self.output_activation(self.enc_fc_mu(h))
        if self.log_min_variance is None:
            logvar = self.output_activation(self.enc_fc_logvar(h))
        else:
            logvar = self.log_min_variance + torch.abs(self.enc_fc_logvar(h))
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = ptu.Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def decode(self, z, clamp=True):
        z = z.view(-1, self.representation_size)
        h = self.relu(self.dec_fc_sample(z))
        if self.imsize == 84:
            x = h.view(-1, 32, 2, 2)
        elif self.imsize == 48:
            x = h.view(-1, 64, 3, 3)
        x = F.relu(self.dec_conv3(x))
        x = F.relu(self.dec_conv2(x))
        x = self.dec_conv1(x)
        x = x.view(-1, self.imsize * self.imsize * self.input_channels)
        if self.use_sigmoid_for_decoder:
            x = self.sigmoid(x)
        if clamp:
            x = torch.clamp(x, 0, 1)
        return x

    def forward(self, x, n_imp=None):
        if n_imp is None:
            n_imp = self.n_imp

        mu, logvar = self.encode(x)
        batch_size = x.shape[0]
        mu = mu.view((batch_size, 1, self.representation_size)).repeat(torch.Size([1, n_imp, 1]))
        logvar = logvar.view((batch_size, 1, self.representation_size)).repeat(torch.Size([1, n_imp, 1]))
        z = self.reparameterize(mu, logvar)
        decoding = self.decode(z, clamp=False).view((batch_size, n_imp, -1))
        return decoding, mu, logvar, z

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
