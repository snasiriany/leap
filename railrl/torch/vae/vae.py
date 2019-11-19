from __future__ import print_function

from torch import nn
from torch.autograd import Variable
import numpy as np
from railrl.pythonplusplus import identity
from railrl.torch.core import PyTorchModule
from railrl.torch.networks import Mlp, TwoHeadMlp
import railrl.torch.pytorch_util as ptu

class VAE(PyTorchModule):
    def __init__(
            self,
            representation_size,
            input_size,
            hidden_sizes=list([64, 128, 64]),
            init_w=1e-3,
            hidden_init=ptu.fanin_init,
            output_activation=identity,
            output_scale=1,
            layer_norm=False,
            normalize=True,
            train_data_mean=None,
            train_data_std=None,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__()
        self.representation_size = representation_size
        self.hidden_init = hidden_init
        self.output_activation = output_activation
        self.dist_mu = np.zeros(self.representation_size)
        self.dist_std = np.ones(self.representation_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.init_w = init_w
        hidden_sizes = list(hidden_sizes)
        self.input_size = input_size
        self.encoder = TwoHeadMlp(hidden_sizes,
                                  representation_size, representation_size,
                                  input_size,
                                  layer_norm=layer_norm,
                                  hidden_init=hidden_init,
                                  output_activation=output_activation,
                                  init_w=init_w)
        hidden_sizes.reverse()
        self.decoder=Mlp(hidden_sizes,
                         input_size,
                         representation_size,
                         layer_norm=layer_norm,
                         hidden_init=hidden_init,
                         output_activation=output_activation,
                         init_w=init_w)
        self.output_scale = output_scale

        self.normalize = normalize
        if train_data_mean is None:
            self.train_data_mean = ptu.np_to_var(np.zeros(input_size))
        else:
            self.train_data_mean = train_data_mean
        if train_data_std is None:
            self.train_data_std = ptu.np_to_var(np.ones(input_size))
        else:
            self.train_data_std = train_data_std

    def encode(self, input):
        input = input.view(-1, self.input_size)
        if self.normalize:
            input = (input - self.train_data_mean) / self.train_data_std
        mu, logvar = self.encoder(input)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        z = z.view(-1, self.representation_size)
        output = self.decoder(z)
        if self.normalize:
            output = output * self.train_data_std + self.train_data_mean
        return output * self.output_scale

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def __getstate__(self):
        d = super().__getstate__()
        # Add these explicitly in case they were modified
        d["_dist_mu"] = self.dist_mu
        d["_dist_std"] = self.dist_std
        d["_normalize"] = self.normalize
        d["_train_data_mean"] = self.train_data_mean
        d["_train_data_std"] = self.train_data_std
        return d

    def __setstate__(self, d):
        super().__setstate__(d)
        self.dist_mu = d["_dist_mu"]
        self.dist_std = d["_dist_std"]
        self.normalize = d["_normalize"]
        self.train_data_mean = d["_train_data_mean"]
        self.train_data_std = d["_train_data_std"]