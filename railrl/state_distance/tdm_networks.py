import numpy as np
import torch

from railrl.state_distance.policies import UniversalPolicy
from railrl.torch.networks import TanhMlpPolicy, FlattenMlp
from railrl.torch.sac.policies import TanhGaussianPolicy


class TdmQf(FlattenMlp):
    def __init__(
            self,
            env,
            vectorized,
            structure='norm_difference',
            learn_offset=False,
            observation_dim=None,
            action_dim=None,
            goal_dim=None,
            norm_order=1,
            output_dim=None,
            **flatten_mlp_kwargs
    ):
        """

        :param env:
        :param hidden_sizes:
        :param vectorized: Boolean. Vectorized or not?
        :param structure: String defining output structure of network:
            - 'norm_difference': Q = -||g - f(inputs)||
            - 'squared_difference': Q = -(g - f(inputs))^2
            - 'squared_difference_offset': Q = -(goal - f(inputs))^2 + f2(s, goal, tau)
            - 'none': Q = f(inputs)

        :param kwargs:
        """
        assert structure in [
            'norm_difference',
            'squared_difference',
            'none',
        ]
        self.save_init_params(locals())

        if observation_dim is None:
            self.observation_dim = env.observation_space.low.size
        else:
            self.observation_dim = observation_dim

        if action_dim is None:
            self.action_dim = env.action_space.low.size
        else:
            self.action_dim = action_dim

        if goal_dim is None:
            self.goal_dim = env.goal_dim
        else:
            self.goal_dim = goal_dim

        if output_dim is None:
            output_dim = self.goal_dim if vectorized else 1

        super().__init__(
            input_size=(
                    self.observation_dim + self.action_dim + self.goal_dim + 1
            ),
            output_size=output_dim,
            **flatten_mlp_kwargs
        )
        self.env = env
        self.vectorized = vectorized
        self.structure = structure
        self.norm_order = norm_order
        self.learn_offset = learn_offset
        if learn_offset:
            self.offset_network = FlattenMlp(
                input_size=(
                    self.observation_dim + self.action_dim + self.goal_dim + 1
                ),
                output_size=self.goal_dim if vectorized else 1,
                **flatten_mlp_kwargs
            )

    def forward(
            self,
            observations,
            actions,
            goals,
            num_steps_left,
            return_predictions=False
    ):
        predictions = super().forward(
            observations, actions, goals, num_steps_left
        )
        if return_predictions:
            return predictions

        if self.structure == 'norm_difference':
            output = - torch.abs(goals - predictions)
        elif self.structure == 'squared_difference':
            output = - (goals - predictions)**2
        elif self.structure == 'none':
            output = predictions
        else:
            raise TypeError("Invalid structure: {}".format(self.structure))
        if not self.vectorized:
            output = - torch.norm(output, p=self.norm_order, dim=1, keepdim=True)

        if self.learn_offset:
            offset = self.offset_network(
                observations, actions, goals, num_steps_left
            )
            output = output + offset

        return output


class TdmVf(FlattenMlp):
    def __init__(
            self,
            env,
            vectorized,
            structure='norm_difference',
            observation_dim=None,
            goal_dim=None,
            norm_order=1,
            output_dim=None,
            **kwargs
    ):
        assert structure in [
            'norm_difference',
            'squared_difference',
            'none',
        ]
        self.save_init_params(locals())

        if observation_dim is None:
            self.observation_dim = env.observation_space.low.size
        else:
            self.observation_dim = observation_dim

        if goal_dim is None:
            self.goal_dim = env.goal_dim
        else:
            self.goal_dim = goal_dim

        if output_dim is None:
            output_dim = self.goal_dim if vectorized else 1

        super().__init__(
            input_size= self.observation_dim + self.goal_dim + 1,
            output_size=output_dim,
            **kwargs
        )
        self.env = env
        self.vectorized = vectorized
        self.structure = structure
        self.norm_order = norm_order

    def forward(
            self,
            observations,
            goals,
            num_steps_left,
    ):
        predictions = super().forward(
            observations, goals, num_steps_left
        )

        if self.structure == 'norm_difference':
            output = - torch.abs(goals - predictions)
        elif self.structure == 'squared_difference':
            output = - (goals - predictions)**2
        elif self.structure == 'none':
            output = predictions
        else:
            raise TypeError("Invalid structure: {}".format(self.structure))
        if not self.vectorized:
            output = - torch.norm(output, p=self.norm_order, dim=1, keepdim=True)

        return output


class TdmPolicy(TanhMlpPolicy):
    """
    Rather than giving `g`, give `g - goalify(s)` as input.
    """
    def __init__(
            self,
            env,
            observation_dim=None,
            action_dim=None,
            goal_dim=None,
            reward_scale=None,
            **kwargs
    ):
        self.save_init_params(locals())

        if observation_dim is None:
            self.observation_dim = env.observation_space.low.size
        else:
            self.observation_dim = observation_dim

        if action_dim is None:
            self.action_dim = env.action_space.low.size
        else:
            self.action_dim = action_dim

        if goal_dim is None:
            self.goal_dim = env.goal_dim
        else:
            self.goal_dim = goal_dim

        self.reward_scale = reward_scale

        super().__init__(
            input_size=self.observation_dim + self.goal_dim + 1,
            output_size=self.action_dim,
            **kwargs
        )
        self.env = env

    def forward(
            self,
            observations,
            goals,
            num_steps_left,
            return_preactivations=False,
    ):
        flat_input = torch.cat((observations, goals, num_steps_left), dim=1)
        return super().forward(
            flat_input,
            return_preactivations=return_preactivations,
        )

    def get_action(self, ob_np, goal_np, tau_np):
        actions = self.eval_np(
            ob_np[None],
            goal_np[None],
            tau_np[None],
        )
        return actions[0, :], {}

    def get_actions(self, ob_np, goal_np, tau_np):
        actions = self.eval_np(
            ob_np,
            goal_np,
            tau_np,
        )
        return actions

class StochasticTdmPolicy(TanhGaussianPolicy, UniversalPolicy):
    def __init__(
            self,
            env,
            observation_dim=None,
            action_dim=None,
            goal_dim=None,
            reward_scale=None,
            **kwargs
    ):
        self.save_init_params(locals())

        if observation_dim is None:
            self.observation_dim = env.observation_space.low.size
        else:
            self.observation_dim = observation_dim

        if action_dim is None:
            self.action_dim = env.action_space.low.size
        else:
            self.action_dim = action_dim

        if goal_dim is None:
            self.goal_dim = env.goal_dim
        else:
            self.goal_dim = goal_dim

        self.reward_scale = reward_scale

        super().__init__(
            obs_dim=self.observation_dim + self.goal_dim + 1,
            action_dim=self.action_dim,
            **kwargs
        )
        self.env = env

    def forward(
            self,
            observations,
            goals,
            num_steps_left,
            **kwargs
    ):
        flat_input = torch.cat((observations, goals, num_steps_left), dim=1)
        return super().forward(flat_input, **kwargs)

    def get_action(self, ob_np, goal_np, tau_np, deterministic=False):
        actions = self.get_actions(
            ob_np[None],
            goal_np[None],
            tau_np[None],
            deterministic=deterministic
        )
        return actions[0, :], {}

    def get_actions(self, obs_np, goals_np, taus_np, deterministic=False):
        return self.eval_np(
            obs_np, goals_np, taus_np, deterministic=deterministic
        )[0]
