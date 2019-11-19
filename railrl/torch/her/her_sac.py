import numpy as np
from railrl.data_management.her_replay_buffer import SimpleHerReplayBuffer, \
    RelabelingReplayBuffer
from railrl.data_management.obs_dict_replay_buffer import \
    ObsDictRelabelingBuffer
from railrl.torch.her.her import HER
from railrl.torch.sac.sac import SoftActorCritic


class HerSac(HER, SoftActorCritic):
    def __init__(
            self,
            *args,
            observation_key=None,
            desired_goal_key=None,
            **kwargs
    ):
        HER.__init__(
            self,
            observation_key=observation_key,
            desired_goal_key=desired_goal_key,
        )
        SoftActorCritic.__init__(self, *args, **kwargs)
        assert isinstance(
            self.replay_buffer, SimpleHerReplayBuffer
        ) or isinstance(
            self.replay_buffer, RelabelingReplayBuffer
        ) or isinstance(
            self.replay_buffer, ObsDictRelabelingBuffer
        )

    def get_eval_action(self, observation, goal):
        if self.observation_key:
            observation = observation[self.observation_key]
        if self.desired_goal_key:
            goal = goal[self.desired_goal_key]
        new_obs = np.hstack((observation, goal))
        return self.policy.get_action(new_obs, deterministic=True)
