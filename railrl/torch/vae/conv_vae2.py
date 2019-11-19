from __future__ import print_function

import torch
import torch.utils.data
from torch import nn
from railrl.pythonplusplus import identity
from railrl.torch import pytorch_util as ptu
import numpy as np
from railrl.torch.core import PyTorchModule

class Flatten(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, input):
        return input.view(-1, self.dim)

class Unflatten(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims
    def forward(self, input):
        return input.view(-1, *self.dims)

class Encoder(nn.Module):
    def __init__(self,
                 input_width,
                 input_height,
                 repr_size=64,
                 input_channels=3,
                 num_filters=16,
                 use_bias_for_conv=False,
                 extra_fc_layer=False,
                 ):
        super(Encoder, self).__init__()
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.use_bias_for_conv = use_bias_for_conv
        self.extra_fc_layer = extra_fc_layer

        conv1 = nn.Conv2d(self.input_channels, num_filters, kernel_size=4, stride=2, padding=1, bias=use_bias_for_conv)
        bn1 = nn.BatchNorm2d(num_filters)

        conv2 = nn.Conv2d(num_filters, num_filters*2, kernel_size=4, stride=2, padding=1, bias=use_bias_for_conv)
        bn2 = nn.BatchNorm2d(num_filters*2)

        conv3 = nn.Conv2d(num_filters*2, num_filters*4, kernel_size=4, stride=2, padding=1, bias=use_bias_for_conv)
        bn3 = nn.BatchNorm2d(num_filters*4)

        conv4 = nn.Conv2d(num_filters*4, num_filters*8, kernel_size=4, stride=2, padding=1, bias=use_bias_for_conv)
        bn4 = nn.BatchNorm2d(num_filters*8)

        self.conv_output_dim = (num_filters*8)*(input_width // 16)*(input_height // 16)
        network = [
            conv1, bn1, nn.LeakyReLU(0.2, inplace=True),
            conv2, bn2, nn.LeakyReLU(0.2, inplace=True),
            conv3, bn3, nn.LeakyReLU(0.2, inplace=True),
            conv4, bn4, nn.LeakyReLU(0.2, inplace=True),
            Flatten(self.conv_output_dim),
        ]
        if self.extra_fc_layer:
            network.append(nn.Linear(self.conv_output_dim, 512))
            network.append(nn.LeakyReLU(0.2, inplace=True))
            self.network = nn.Sequential(*network)
            self.fc_mu = nn.Linear(512, repr_size)
            self.fc_logvar = nn.Linear(512, repr_size)
        else:
            self.network = nn.Sequential(*network)
            self.fc_mu = nn.Linear(self.conv_output_dim, repr_size)
            self.fc_logvar = nn.Linear(self.conv_output_dim, repr_size)

    def forward(self, x):
        h = self.network(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self,
                 input_width,
                 input_height,
                 repr_size=64,
                 input_channels=3,
                 num_filters=16,
                 use_bias_for_conv=False,
                 fix_logvar=False,
                 extra_fc_layer=False,
                 ):
        super(Decoder, self).__init__()
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.use_bias_for_conv = use_bias_for_conv
        self.extra_fc_layer = extra_fc_layer

        if input_width == 84:
            self.paddings = [1, 1, 0, 1]
        elif input_width == 48:
            self.paddings = [1, 1, 1, 1]

        conv4 = nn.ConvTranspose2d(num_filters*8, num_filters*4, kernel_size=4, stride=2, padding=self.paddings[0], bias=use_bias_for_conv)
        bn4 = nn.BatchNorm2d(num_filters*4)

        conv3 = nn.ConvTranspose2d(num_filters*4, num_filters*2, kernel_size=4, stride=2, padding=self.paddings[1], bias=use_bias_for_conv)
        bn3 = nn.BatchNorm2d(num_filters*2)

        conv2 = nn.ConvTranspose2d(num_filters*2, num_filters, kernel_size=4, stride=2, padding=self.paddings[2], bias=use_bias_for_conv)
        bn2 = nn.BatchNorm2d(num_filters)

        conv1 = nn.ConvTranspose2d(num_filters, input_channels*2, kernel_size=4, stride=2, padding=self.paddings[3], bias=use_bias_for_conv)

        conv_output_dim = (num_filters*8)*(input_width // 16)*(input_height // 16)
        if self.extra_fc_layer:
            network = [
                nn.Linear(repr_size, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, conv_output_dim),
                nn.ReLU(inplace=True),
            ]
        else:
            network = [
                nn.Linear(repr_size, conv_output_dim),
                nn.ReLU(inplace=True),
            ]
        network = network + [
            Unflatten(num_filters*8, input_width // 16, input_height // 16),
            conv4, bn4, nn.ReLU(inplace=True),
            conv3, bn3, nn.ReLU(inplace=True),
            conv2, bn2, nn.ReLU(inplace=True),
            conv1,
        ]

        self.network = nn.Sequential(*network)

        self.fix_logvar = fix_logvar

    def forward(self, z):
        output = self.network(z)
        mu, logvar = output[:, :self.input_channels, :, :], output[:, self.input_channels:, :, :]
        return mu, logvar

class ConvVAE2(PyTorchModule):
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
            num_filters=16,
            extra_fc_layer=False,
    ):
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
        self.sigmoid = nn.Sigmoid()
        self.init_w = init_w
        self.large_arch = large_arch

        self.n_imp = n_imp # number of importance samples for IWAE objective

        self.gaussian_decoder = gaussian_decoder
        self.use_sigmoid_for_decoder = use_sigmoid_for_decoder
        self.learned_decoder_variance = learned_decoder_variance

        self.num_filters = num_filters

        self.encoder = Encoder(
            imsize, imsize,
            repr_size=representation_size,
            num_filters=num_filters,
            extra_fc_layer=extra_fc_layer,
        )
        self.decoder = Decoder(
            imsize, imsize,
            repr_size=representation_size,
            num_filters=num_filters,
            extra_fc_layer=extra_fc_layer,
        )
        if action_dim is not None:
            self.linear_constraint_fc = \
                nn.Linear(
                    self.representation_size + action_dim,
                    self.representation_size
                )
        else:
            self.linear_constraint_fc = None

        self.init_weights()

    def init_weights(self):
        pass

    def encode(self, input):
        input = input.view(-1, self.imlength)
        conv_input = input.narrow(start=0, length=self.imlength, dimension=1)
        x = conv_input.contiguous().view(-1, self.input_channels, self.imsize, self.imsize)
        mu, logvar = self.encoder(x)
        if self.log_min_variance is None:
            logvar = self.output_activation(logvar)
        else:
            logvar = self.log_min_variance + torch.abs(logvar)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = ptu.Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def decode(self, z, clamp=True):
        z = z.view(-1, self.representation_size)
        mu, logvar = self.decoder(z)
        mu = mu.contiguous().view(-1, self.imsize * self.imsize * self.input_channels)
        x = mu
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
