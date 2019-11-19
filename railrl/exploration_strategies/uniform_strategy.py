from railrl.exploration_strategies.base import RawExplorationStrategy
from railrl.core.serializable import Serializable
import numpy as np


class UniformStrategy(RawExplorationStrategy, Serializable):
    """
    This strategy adds noise sampled uniformly to the action taken by the
    deterministic policy.
    """
    def __init__(self, action_space, low=0., high=1.):
        Serializable.quick_init(self, locals())
        self._low = action_space.low
        self._high = action_space.high

    def get_action_from_raw_action(self, action, t=None, **kwargs):
        return np.clip(
            action + np.random.uniform(
                self._low,
                self._high,
                size=action.shape,
            ),
            self._low,
            self._high,
        )
