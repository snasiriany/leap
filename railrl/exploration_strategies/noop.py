from railrl.exploration_strategies.base import RawExplorationStrategy
import numpy as np


class NoopStrategy(RawExplorationStrategy):
    """
    Exploration strategy that does nothing other than clip the action.
    """

    def __init__(self, action_space, **kwargs):
        self.action_space = action_space

    def get_action(self, t, observation, policy, **kwargs):
        return policy.get_action(observation)

    def get_action_from_raw_action(self, action, **kwargs):
        return np.clip(action, self.action_space.low, self.action_space.high)
