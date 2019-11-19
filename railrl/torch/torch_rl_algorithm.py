import abc
from typing import Iterable

from railrl.core.rl_algorithm import RLAlgorithm
from railrl.torch.core import PyTorchModule, np_to_pytorch_batch


class TorchRLAlgorithm(RLAlgorithm, metaclass=abc.ABCMeta):
    def get_batch(self):
        batch = self.replay_buffer.random_batch(self.batch_size)
        return np_to_pytorch_batch(batch)

    @property
    @abc.abstractmethod
    def networks(self) -> Iterable[PyTorchModule]:
        pass

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def cuda(self):
        for net in self.networks:
            net.cuda()
