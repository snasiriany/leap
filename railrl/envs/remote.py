import ray

from railrl.envs.base import RolloutEnv
from railrl.envs.wrappers import NormalizedBoxEnv, ProxyEnv
from railrl.core.serializable import Serializable
import numpy as np
import torch
import railrl.torch.pytorch_util as ptu
import math
from torch.multiprocessing import Process, Pipe
from multiprocessing.connection import wait
import torch.multiprocessing as mp
import railrl.envs.env_utils as env_utils
import cloudpickle

class RemoteRolloutEnv(ProxyEnv, RolloutEnv, Serializable):
    """
    An interface for a rollout environment where the rollouts are performed
    asynchronously.

    This "environment" just talks to the remote environment. The advantage of
    this environment over calling WorkerEnv directly is that rollout will return
    `None` if a path is not ready, rather than returning a promise (an
    `ObjectID` in Ray-terminology).

    Rather than doing
    ```
    env = CarEnv(foo=1)
    path = env.rollout()  # blocks until rollout is done
    # do some computation
    ```
    you can do
    ```
    remote_env = RemoteRolloutEnv(CarEnv, {'foo': 1})
    path = remote_env.rollout()
    while path is None:
        # do other computation asynchronously
        path = remote_env.rollout() ```
    ```
    So you pass the environment class (CarEnv) and parameters to create the
    environment to RemoteRolloutEnv. What happens under the door is that the
    RemoteRolloutEnv will create its own instance of CarEnv with those
    parameters.

    This Env can be considered the master env, collecting and managing the
    rollouts of worker envs under the hood. Communication between workers and
    master env are achieved through pipes. No explicit synchronization is
    necessary as the worker/master will alternate between listening and sending
    data through the pipe. Thus, there should never be a case where master and
    worker are both sending or listening.

    """
    def __init__(
            self,
            env,
            policy,
            exploration_policy,
            max_path_length,
            train_rollout_function,
            eval_rollout_function,
            num_workers=2,
    ):
        Serializable.quick_init(self, locals())
        super().__init__(env)
        self.num_workers = num_workers
        # Let self.worker_limits[True] be the max number of workers for training
        # and self.worker_limits[False] be the max number of workers for eval.
        self.worker_limits = {
            True: math.ceil(self.num_workers / 2),
            False: math.ceil(self.num_workers / 2),
        }

        self.parent_pipes = []
        self.child_pipes = []

        for _ in range(num_workers):
            parent_conn, child_conn = Pipe()
            self.parent_pipes.append(parent_conn)
            self.child_pipes.append(child_conn)

        self._workers = [
            Process(
                target=RemoteRolloutEnv._worker_loop,
                args=(
                    self.child_pipes[i],
                    env,
                    policy,
                    exploration_policy,
                    max_path_length,
                    cloudpickle.dumps(train_rollout_function),
                    cloudpickle.dumps(eval_rollout_function),
                )
            )
        for i in range(num_workers)]

        for worker in self._workers:
            worker.start()

        self.free_pipes = set(self.parent_pipes)
        # self.pipe_info[pipe] stores (epoch, train_type)
        self.pipe_info = {}
        # Let self.promise_list[True] be the promises for training
        # and self.promise_list[False] be the promises for eval.
        self.rollout_promise_list = {
            True: [],
            False: [],
        }

    def rollout(self, policy, train, epoch, discard_other_rollout_type=False):
        if not self.workers_alive():
            raise RuntimeError("Worker has died prematurely.")
        # prevent starvation if only one worker
        if discard_other_rollout_type:
            self._discard_rollout_promises(not train)

        self._alloc_rollout_promise(policy, train, epoch)
        # Check if remote path has been collected.
        ready_promises = wait(self.rollout_promise_list[train], timeout=0)
        for rollout_promise in ready_promises:
            rollout = rollout_promise.recv()
            path_epoch, _ = self.pipe_info[rollout_promise]
            self._free_rollout_promise(rollout_promise)
            # Throw away eval paths from previous epochs
            if path_epoch != epoch and train == False:
                continue
            self._alloc_rollout_promise(policy, train, epoch)
            return rollout
        return None

    def workers_alive(self):
        return all(worker.is_alive() for worker in self._workers)

    def update_worker_envs(self, update):
        self.env_update = update

    def shutdown(self):
        for worker in self._workers:
            worker.terminate()

    def _alloc_rollout_promise(self, policy, train, epoch):
        if len(self.free_pipes) == 0 or \
           len(self.rollout_promise_list[train]) >= self.worker_limits[train]:
            return
        policy_params = policy.get_param_values_np()

        free_pipe = self.free_pipes.pop()
        free_pipe.send((self.env_update, (policy_params, train,)))
        self.pipe_info[free_pipe] = (epoch, train)
        self.rollout_promise_list[train].append(free_pipe)
        return free_pipe

    def _free_rollout_promise(self, pipe):
        _, train = self.pipe_info[pipe]
        assert pipe not in self.free_pipes
        if wait([pipe], timeout=0):
            pipe.recv()
        self.free_pipes.add(pipe)
        del self.pipe_info[pipe]
        self.rollout_promise_list[train].remove(pipe)

    def _discard_rollout_promises(self, train_type):
        ready_promises = wait(self.rollout_promise_list[train_type], timeout=0)
        for rollout_promise in ready_promises:
            self._free_rollout_promise(rollout_promise)


    def _worker_loop(pipe, *worker_env_args, **worker_env_kwargs):
        env = RemoteRolloutEnv.WorkerEnv(*worker_env_args, **worker_env_kwargs)
        while True:
            wait([pipe])
            env_update, rollout_args = pipe.recv()
            if env_update is not None:
                env._env.update_env(**env_update)
            rollout = env.rollout(*rollout_args)
            pipe.send(rollout)

    class WorkerEnv:
        def __init__(
                self,
                env,
                policy,
                exploration_policy,
                max_path_length,
                train_rollout_function,
                eval_rollout_function,
        ):
            torch.set_num_threads(1)
            self._env = env
            self._policy = policy
            self._exploration_policy = exploration_policy
            self._max_path_length = max_path_length
            self.train_rollout_function = cloudpickle.loads(train_rollout_function)
            self.eval_rollout_function = cloudpickle.loads(eval_rollout_function)

        def rollout(self, policy_params, use_exploration_strategy):
            if use_exploration_strategy:
                self._exploration_policy.set_param_values_np(policy_params)
                policy = self._exploration_policy
                rollout_function = self.train_rollout_function
                env_utils.mode(self._env, 'train')
            else:
                self._policy.set_param_values_np(policy_params)
                policy = self._policy
                rollout_function = self.eval_rollout_function
                env_utils.mode(self._env, 'eval')

            rollout = rollout_function(self._env, policy, self._max_path_length)
            if 'full_observations' in rollout:
                rollout['observations'] = rollout['full_observations'][:-1]
            return rollout
