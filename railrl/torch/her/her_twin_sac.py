import numpy as np
from railrl.data_management.her_replay_buffer import SimpleHerReplayBuffer, \
    RelabelingReplayBuffer
from railrl.data_management.obs_dict_replay_buffer import \
    ObsDictRelabelingBuffer
from railrl.torch.her.her import HER
from railrl.torch.sac.twin_sac import TwinSAC


class HerTwinSAC(HER, TwinSAC):
    def __init__(
            self,
            *args,
            twin_sac_kwargs,
            her_kwargs,
            base_kwargs,
            **kwargs
    ):
        HER.__init__(
            self,
            **her_kwargs,
        )
        TwinSAC.__init__(self, *args, **kwargs, **twin_sac_kwargs, **base_kwargs)
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
