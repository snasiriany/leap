from typing import Iterable

from railrl.exploration_strategies.base import RawExplorationStrategy


class ProductStrategy(RawExplorationStrategy):
    """
    Apply different strategies to different parts parts of a Product space.
    """

    def __init__(
            self,
            exploration_strategies: Iterable[RawExplorationStrategy],
    ):
        """
        :param exploration_strategies: List[ExplorationStrategy]
        """
        self._wrapped_strategies = exploration_strategies

    def get_action(self, t, observation, policy, **kwargs):
        action, agent_info = policy.get_action(observation)
        return self.get_action_from_raw_action(
            action,
            t=t,
            observation=observation,
        ), agent_info

    def get_action_from_raw_action(self, action, **kwargs):
        return tuple(
            es.get_action_from_raw_action(partial_action, **kwargs)
            for es, partial_action in zip(self._wrapped_strategies, action)
        )
