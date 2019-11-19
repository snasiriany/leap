"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

from railrl.policies.base import Policy
from railrl.pythonplusplus import identity
from railrl.torch import pytorch_util as ptu
from railrl.torch.core import PyTorchModule
from railrl.torch.data_management.normalizer import TorchFixedNormalizer
from railrl.torch.modules import SelfOuterProductLinear, LayerNorm

import numpy as np
from PIL import Image

class CNN(PyTorchModule):
    def __init__(self,
                input_width,
                input_height,
                input_channels,
                output_size,
                kernel_sizes,
                n_channels,
                strides,
                pool_sizes,
                paddings,
                hidden_sizes=[],
                added_fc_input_size=0,
                use_batch_norm=False,
                init_w=1e-4,
                hidden_activation=nn.ReLU(),
                output_activation=identity
        ):
        assert len(kernel_sizes) == \
               len(n_channels) == \
               len(strides) == \
               len(pool_sizes) == \
               len(paddings)
        self.save_init_params(locals())
        super().__init__()

        self.hidden_sizes = hidden_sizes
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.output_size = output_size
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        self.use_batch_norm = use_batch_norm
        self.added_fc_input_size = added_fc_input_size
        self.conv_input_length = self.input_width * self.input_height * self.input_channels

        self.conv_layers = nn.ModuleList()
        self.conv_norm_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.fc_norm_layers = nn.ModuleList()

        for out_channels, kernel_size, stride, pool, padding in \
            zip(n_channels, kernel_sizes, strides, pool_sizes, paddings):

            conv = nn.Conv2d(input_channels,
                             out_channels,
                             kernel_size,
                             stride=stride,
                             padding=padding)
            nn.init.xavier_uniform(conv.weight)

            conv_layer = nn.Sequential(
                            conv,
                            nn.MaxPool2d(pool, pool),
            )
            self.conv_layers.append(conv_layer)
            input_channels = out_channels

        # find output dimension of conv_layers by trial and add normalization conv layers
        test_mat = Variable(torch.zeros(1, self.input_channels, self.input_width, self.input_height))
        for conv_layer in self.conv_layers:
            test_mat = conv_layer(test_mat)
            self.conv_norm_layers.append(nn.BatchNorm2d(test_mat.shape[1]))

        fc_input_size = int(np.prod(test_mat.shape))
        # used only for injecting input directly into fc layers
        fc_input_size += added_fc_input_size

        for idx, hidden_size in enumerate(hidden_sizes):
            fc_layer = nn.Linear(fc_input_size, hidden_size)

            norm_layer = nn.BatchNorm1d(hidden_size)
            nn.init.xavier_uniform(fc_layer.weight)

            self.fc_layers.append(fc_layer)
            self.fc_norm_layers.append(norm_layer)
            fc_input_size = hidden_size

        self.last_fc = nn.Linear(fc_input_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input):
        fc_input = (self.added_fc_input_size != 0)

        conv_input = input.narrow(start=0,
                                  length=self.conv_input_length,
                                  dimension=1).contiguous()
        if fc_input:
            extra_fc_input = input.narrow(start=self.conv_input_length,
                                        length=self.added_fc_input_size,
                                        dimension=1)
        # need to reshape from batch of flattened images into (channsls, w, h)
        h = conv_input.view(conv_input.shape[0],
                       self.input_channels,
                       self.input_height,
                       self.input_width)

        #from PIL import Image
        #import pdb; pdb.set_trace()
        h = self.apply_forward(h, self.conv_layers, self.conv_norm_layers)
        # flatten channels for fc layers
        h = h.view(h.size(0), -1)
        if fc_input:
            h = torch.cat((h, extra_fc_input), dim=1)
        h = self.apply_forward(h, self.fc_layers, self.fc_norm_layers)

        output = self.output_activation(self.last_fc(h))
        return output

    def apply_forward(self, input, hidden_layers, norm_layers):
        h = input
        for layer, norm_layer in zip(hidden_layers, norm_layers):
            h = layer(h)
            if self.use_batch_norm:
                h = norm_layer(h)
            h = self.hidden_activation(h)
        return h


class MergedCNN(CNN):
    '''
    CNN that supports input directly into fully connected layers
    '''

    def __init__(self,
                added_fc_input_size,
                **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(added_fc_input_size=added_fc_input_size,
                         **kwargs)


    def forward(self, conv_input, fc_input):
        h = torch.cat((conv_input, fc_input), dim=1)
        output = super().forward(h)
        return output


class CNNPolicy(CNN, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().forward(obs, **kwargs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return self.eval_np(obs)



class Mlp(PyTorchModule):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        self.save_init_params(locals())
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.fill_(0)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class FlattenMlp(Mlp):
    """
    Flatten inputs along dimension 1 and then pass through MLP.
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs, **kwargs)


class MlpQf(FlattenMlp):
    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer = None,
            action_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer
        self.action_normalizer = action_normalizer

    def forward(self, obs, actions, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        if self.action_normalizer:
            actions = self.action_normalizer.normalize(actions)
        return super().forward(obs, actions, **kwargs)


class MlpPolicy(Mlp, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().forward(obs, **kwargs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return self.eval_np(obs)


class TanhMlpPolicy(MlpPolicy):
    """
    A helper class since most policies have a tanh output activation.
    """
    def __init__(self, *args, **kwargs):
        self.save_init_params(locals())
        super().__init__(*args, output_activation=torch.tanh, **kwargs)


class FeedForwardQFunction(PyTorchModule):
    def __init__(
            self,
            obs_dim,
            action_dim,
            observation_hidden_size,
            embedded_hidden_size,
            init_w=3e-3,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            batchnorm_obs=False,
    ):
        print("WARNING: This class will soon be deprecated.")
        self.save_init_params(locals())
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.observation_hidden_size = observation_hidden_size
        self.embedded_hidden_size = embedded_hidden_size
        self.hidden_init = hidden_init
        self.obs_fc = nn.Linear(obs_dim, observation_hidden_size)
        self.embedded_fc = nn.Linear(observation_hidden_size + action_dim,
                                     embedded_hidden_size)

        self.last_fc = nn.Linear(embedded_hidden_size, 1)
        self.output_activation = output_activation

        self.init_weights(init_w)
        self.batchnorm_obs = batchnorm_obs
        if self.batchnorm_obs:
            self.bn_obs = nn.BatchNorm1d(obs_dim)

    def init_weights(self, init_w):
        self.hidden_init(self.obs_fc.weight)
        self.obs_fc.bias.data.fill_(0)
        self.hidden_init(self.embedded_fc.weight)
        self.embedded_fc.bias.data.fill_(0)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs, action):
        if self.batchnorm_obs:
            obs = self.bn_obs(obs)
        h = obs
        h = F.relu(self.obs_fc(h))
        h = torch.cat((h, action), dim=1)
        h = F.relu(self.embedded_fc(h))
        return self.output_activation(self.last_fc(h))


class FeedForwardPolicy(PyTorchModule):
    def __init__(
            self,
            obs_dim,
            action_dim,
            fc1_size,
            fc2_size,
            init_w=1e-3,
            hidden_init=ptu.fanin_init,
    ):
        print("WARNING: This class will soon be deprecated.")
        self.save_init_params(locals())
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.hidden_init = hidden_init

        self.fc1 = nn.Linear(obs_dim, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.last_fc = nn.Linear(fc2_size, action_dim)

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.hidden_init(self.fc1.weight)
        self.fc1.bias.data.fill_(0)
        self.hidden_init(self.fc2.weight)
        self.fc2.bias.data.fill_(0)

        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs):
        h = F.relu(self.fc1(obs))
        h = F.relu(self.fc2(h))
        return F.tanh(self.last_fc(h))

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return self.eval_np(obs)


"""
Random Networks Below
"""
class TwoHeadMlp(PyTorchModule):
    def __init__(
            self,
            hidden_sizes,
            first_head_size,
            second_head_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        self.save_init_params(locals())
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.first_head_size = first_head_size
        self.second_head_size = second_head_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.first_head = nn.Linear(in_size, self.first_head_size)
        self.first_head.weight.data.uniform_(-init_w, init_w)

        self.second_head = nn.Linear(in_size, self.second_head_size)
        self.second_head.weight.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.first_head(h)
        first_output = self.output_activation(preactivation)
        preactivation = self.second_head(h)
        second_output = self.output_activation(preactivation)

        return first_output, second_output

class OuterProductFF(PyTorchModule):
    """
    An interesting idea that I had where you first take the outer product of
    all inputs, flatten it, and then pass it through a linear layer. I
    haven't really tested this, but I'll leave it here to tempt myself later...
    """

    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.,
    ):
        self.save_init_params(locals())
        super().__init__()

        self.sops = []
        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            sop = SelfOuterProductLinear(in_size, next_size)
            in_size = next_size
            hidden_init(sop.fc.weight)
            sop.fc.bias.data.fill_(b_init_value)
            self.__setattr__("sop{}".format(i), sop)
            self.sops.append(sop)
        self.output_activation = output_activation
        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.fill_(b_init_value)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, sop in enumerate(self.sops):
            h = self.hidden_activation(sop(h))
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class AETanhPolicy(MlpPolicy):
    """
    A helper class since most policies have a tanh output activation.
    """
    def __init__(
            self,
            ae,
            env,
            history_length,
            *args,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs, output_activation=torch.tanh)
        self.ae = ae
        self.history_length = history_length
        self.env = env

    def get_action(self, obs_np):
        obs = obs_np
        obs = ptu.np_to_var(obs)
        image_obs, fc_obs = self.env.split_obs(obs)
        latent_obs = self.ae.history_encoder(image_obs, self.history_length)
        if fc_obs is not None:
            latent_obs = torch.cat((latent_obs, fc_obs), dim=1)
        obs_np = ptu.get_numpy(latent_obs)[0]
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}


class FeatPointMlp(PyTorchModule):
    def __init__(
            self,
            downsample_size,
            input_channels,
            num_feat_points,
            temperature=1.0,
            init_w=1e-3,
            input_size=32,
            hidden_init=ptu.fanin_init,
            output_activation=identity,
    ):
        self.save_init_params(locals())
        super().__init__()

        self.downsample_size = downsample_size
        self.temperature = temperature
        self.num_feat_points = num_feat_points
        self.hidden_init = hidden_init
        self.output_activation = output_activation
        self.input_channels = input_channels
        self.input_size = input_size

#        self.bn1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(input_channels, 48, kernel_size=5, stride=2)
#        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(48, 48, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(48, self.num_feat_points, kernel_size=5, stride=1)

        test_mat = Variable(torch.zeros(1, self.input_channels, self.input_size, self.input_size))
        test_mat = self.conv1(test_mat)
        test_mat = self.conv2(test_mat)
        test_mat = self.conv3(test_mat)
        self.out_size = int(np.prod(test_mat.shape))
        self.fc1 = nn.Linear(2 * self.num_feat_points, 400)
        self.fc2 = nn.Linear(400, 300)
        self.last_fc = nn.Linear(300, self.input_channels * self.downsample_size * self.downsample_size)

        self.init_weights(init_w)
        self.i = 0

    def init_weights(self, init_w):
        self.hidden_init(self.conv1.weight)
        self.conv1.bias.data.fill_(0)
        self.hidden_init(self.conv2.weight)
        self.conv2.bias.data.fill_(0)

    def forward(self, input):
        h = self.encoder(input)
        out = self.decoder(h)
        return out


    def encoder(self, input):
        x = input.contiguous().view(-1, self.input_channels, self.input_size, self.input_size)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        d = int((self.out_size // self.num_feat_points)**(1/2))
        x = x.view(-1, self.num_feat_points, d*d)
        x = F.softmax(x / self.temperature, 2)
        x = x.view(-1, self.num_feat_points, d, d)

        maps_x = torch.sum(x, 2)
        maps_y = torch.sum(x, 3)

        weights = ptu.np_to_var(np.arange(d) / (d + 1))

        fp_x = torch.sum(maps_x * weights, 2)
        fp_y = torch.sum(maps_y * weights, 2)

        x = torch.cat([fp_x, fp_y], 1)
#        h = x.view(-1, 2, self.num_feat_points).transpose(1, 2).contiguous().view(-1, self.num_feat_points * 2)
        h = x.view(-1, self.num_feat_points * 2)
        return h

    def decoder(self, input):
        h = input
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = self.last_fc(h)
        return h

    def history_encoder(self, input, history_length):
        input = input.contiguous().view(-1,
                                        self.input_channels,
                                        self.input_size,
                                        self.input_size)
        latent = self.encoder(input)

        assert latent.shape[0] % history_length == 0
        n_samples = latent.shape[0] // history_length
        latent = latent.view(n_samples, -1)
        return latent


