from railrl.data_management.her_replay_buffer import SimpleHerReplayBuffer, \
    RelabelingReplayBuffer
from railrl.data_management.obs_dict_replay_buffer import \
    ObsDictRelabelingBuffer
from railrl.torch.her.her import HER
from railrl.torch.td3.td3 import TD3


class HerTd3(HER, TD3):
    def __init__(
            self,
            *args,
            td3_kwargs,
            her_kwargs,
            base_kwargs,
            **kwargs
    ):
        HER.__init__(
            self,
            **her_kwargs,
        )
        TD3.__init__(self, *args, **kwargs, **td3_kwargs, **base_kwargs)
        assert isinstance(
            self.replay_buffer, SimpleHerReplayBuffer
        ) or isinstance(
            self.replay_buffer, RelabelingReplayBuffer
        ) or isinstance(
            self.replay_buffer, ObsDictRelabelingBuffer
        )
