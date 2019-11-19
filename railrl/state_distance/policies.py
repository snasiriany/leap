"""
Policies to be used with a state-distance Q function.
"""
import abc
from itertools import product

import numpy as np
from scipy import optimize
from torch import nn
from torch import optim

from railrl.policies.base import ExplorationPolicy, Policy
from railrl.torch import pytorch_util as ptu
from railrl.core import logger


class UniversalPolicy(ExplorationPolicy, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_action(self, observation, goal, tau, **kwargs):
        pass

    def reset(self):
        pass

    def get_param_values(self):
        return None

    def set_param_values(self, param_values):
        return


class SampleBasedUniversalPolicy(
    UniversalPolicy, ExplorationPolicy, metaclass=abc.ABCMeta
):
    def __init__(self, sample_size, env, sample_actions_from_grid=False):
        super().__init__()
        self.sample_size = sample_size
        self.env = env
        self.sample_actions_from_grid = sample_actions_from_grid
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high
        self._goal_batch = None
        self._goal_batch_np = None
        self._tau_batch = None

    def set_goal(self, goal_np):
        super().set_goal(goal_np)
        self._goal_batch = self.expand_np_to_var(goal_np)
        self._goal_batch_np = np.repeat(
            np.expand_dims(goal_np, 0),
            self.sample_size,
            axis=0
        )

    def set_tau(self, tau):
        super().set_tau(tau)
        self._tau_batch = self.expand_np_to_var(np.array([tau]))

    def expand_np_to_var(self, array):
        array_expanded = np.repeat(
            np.expand_dims(array, 0),
            self.sample_size,
            axis=0
        )
        return ptu.np_to_var(array_expanded, requires_grad=False)

    def sample_actions(self):
        if self.sample_actions_from_grid:
            action_dim = self.env.action_dim
            resolution = int(np.power(self.sample_size, 1./action_dim))
            values = []
            for dim in range(action_dim):
                values.append(np.linspace(
                    self.action_low[dim],
                    self.action_high[dim],
                    num=resolution
                ))
            actions = np.array(list(product(*values)))
            if len(actions) < self.sample_size:
                # Add extra actions in case the grid can't perfectly create
                # `self.sample_size` actions. e.g. sample_size is 30, but the
                # grid is 5x5.
                actions = np.concatenate(
                    (
                        actions,
                        self.env.sample_actions(
                            self.sample_size - len(actions)
                        ),
                    ),
                    axis=0,
                )
            return actions
        else:
            return self.env.sample_actions(self.sample_size)

    def sample_states(self):
        return self.env.sample_states(self.sample_size)
