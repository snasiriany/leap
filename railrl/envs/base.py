import abc


class RolloutEnv(object):
    """ Environment that supports full rollouts."""

    @abc.abstractmethod
    def rollout(self, *args, **kwargs):
        pass
