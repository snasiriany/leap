import abc
import copy
import os.path as osp
import pickle
import time
from collections import OrderedDict
import os
import psutil

import gtimer as gt
import numpy as np

from railrl.core import logger
from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from railrl.data_management.path_builder import PathBuilder
from railrl.envs.remote import RemoteRolloutEnv
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.misc import eval_util
from railrl.misc.asset_loader import local_path_from_s3_or_local_path
from railrl.policies.base import ExplorationPolicy
from railrl.samplers.in_place import InPlacePathSampler
import railrl.envs.env_utils as env_utils


class RLAlgorithm(metaclass=abc.ABCMeta):
    def __init__(
            self,
            env,
            exploration_policy: ExplorationPolicy,
            training_env=None,
            eval_sampler=None,
            eval_policy=None,
            collection_mode='online',

            num_epochs=100,
            epoch_freq=1,
            num_steps_per_epoch=10000,
            num_steps_per_eval=1000,
            num_updates_per_env_step=1,
            num_updates_per_epoch=None,
            batch_size=1024,
            max_path_length=1000,
            discount=0.99,
            reward_scale=1,
            min_num_steps_before_training=None,
            replay_buffer_size=1000000,
            replay_buffer=None,
            train_on_eval_paths=False,
            do_training=True,
            oracle_transition_data=None,

            # I/O parameters
            render=False,
            render_during_eval=False,
            save_replay_buffer=False,
            save_algorithm=False,
            save_environment=False,
            normalize_env=True,

            # Remote env parameters
            sim_throttle=True,
            parallel_step_ratio=1,
            parallel_env_params=None,

            env_info_sizes=None,

            num_rollouts_per_eval=None,

            epoch_list=None,

            **kwargs
    ):
        """
        Base class for RL Algorithms
        :param env: Environment used to evaluate.
        :param exploration_policy: Policy used to explore
        :param training_env: Environment used by the algorithm. By default, a
        copy of `env` will be made.
        :param num_epochs:
        :param num_steps_per_epoch:
        :param num_steps_per_eval:
        :param num_updates_per_env_step: Used by online training mode.
        :param num_updates_per_epoch: Used by batch training mode.
        :param batch_size:
        :param max_path_length:
        :param discount:
        :param replay_buffer_size:
        :param reward_scale:
        :param render:
        :param save_replay_buffer:
        :param save_algorithm:
        :param save_environment:
        :param eval_sampler:
        :param eval_policy: Policy to evaluate with.
        :param collection_mode:
        :param sim_throttle:
        :param normalize_env:
        :param parallel_step_to_train_ratio:
        :param replay_buffer:
        """
        assert collection_mode in ['online', 'online-parallel', 'offline',
                                   'batch']
        if collection_mode == 'batch':
            assert num_updates_per_epoch is not None
        self.training_env = training_env or pickle.loads(pickle.dumps(env))
        self.normalize_env = normalize_env
        self.exploration_policy = exploration_policy
        self.num_epochs = num_epochs
        self.epoch_freq = epoch_freq
        self.epoch_list = epoch_list
        self.num_env_steps_per_epoch = num_steps_per_epoch
        if collection_mode == 'online' or collection_mode == 'online-parallel':
            self.num_updates_per_train_call = num_updates_per_env_step
        else:
            self.num_updates_per_train_call = num_updates_per_epoch
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_rollouts_per_eval = num_rollouts_per_eval
        if self.num_rollouts_per_eval is not None:
            self.num_steps_per_eval = self.num_rollouts_per_eval * self.max_path_length
        else:
            self.num_steps_per_eval = num_steps_per_eval
        self.discount = discount
        self.replay_buffer_size = replay_buffer_size
        self.reward_scale = reward_scale
        self.render = render
        self.render_during_eval = render_during_eval
        self.collection_mode = collection_mode
        self.save_replay_buffer = save_replay_buffer
        self.save_algorithm = save_algorithm
        self.save_environment = save_environment
        if min_num_steps_before_training is None:
            min_num_steps_before_training = self.num_env_steps_per_epoch
        self.min_num_steps_before_training = min_num_steps_before_training
        if eval_sampler is None:
            if eval_policy is None:
                eval_policy = exploration_policy
            eval_sampler = InPlacePathSampler(
                env=env,
                policy=eval_policy,
                max_samples=self.num_steps_per_eval + self.max_path_length,
                max_path_length=self.max_path_length,
                render=render_during_eval,
            )
        self.eval_policy = eval_policy
        self.eval_sampler = eval_sampler
        self.eval_statistics = OrderedDict()
        self.need_to_update_eval_statistics = True

        self.action_space = env.action_space
        self.obs_space = env.observation_space
        self.env = env
        if replay_buffer is None:
            self.replay_buffer = EnvReplayBuffer(
                self.replay_buffer_size,
                self.env,
                env_info_sizes=env_info_sizes,
            )
        else:
            self.replay_buffer = replay_buffer

        self._n_env_steps_total = 0
        self._n_train_steps_total = 0
        self._n_rollouts_total = 0
        self._do_train_time = 0
        self._epoch_start_time = None
        self._algo_start_time = None
        self._old_table_keys = None
        self._current_path_builder = PathBuilder()
        self._exploration_paths = []
        self.train_on_eval_paths = train_on_eval_paths
        self.do_training = do_training
        self.oracle_transition_data = oracle_transition_data

        self.parallel_step_ratio = parallel_step_ratio
        self.sim_throttle = sim_throttle
        self.parallel_env_params = parallel_env_params or {}
        self.init_rollout_function()
        self.post_epoch_funcs = []
        self._exploration_policy_noise = 0

    def train(self, start_epoch=0):
        self.pretrain()
        if start_epoch == 0:
            params = self.get_epoch_snapshot(-1)
            logger.save_itr_params(-1, params)
        self.training_mode(False)
        self._n_env_steps_total = start_epoch * self.num_env_steps_per_epoch
        gt.reset()
        gt.set_def_unique(False)
        if self.collection_mode == 'online':
            self.train_online(start_epoch=start_epoch)
        elif self.collection_mode == 'online-parallel':
            try:
                self.train_parallel(start_epoch=start_epoch)
            except:
                import traceback
                traceback.print_exc()
                self.parallel_env.shutdown()
        elif self.collection_mode == 'batch':
            self.train_batch(start_epoch=start_epoch)
        elif self.collection_mode == 'offline':
            self.train_offline(start_epoch=start_epoch)
        else:
            raise TypeError("Invalid collection_mode: {}".format(
                self.collection_mode
            ))
        self.cleanup()

    def cleanup(self):
        pass

    def pretrain(self):
        if self.oracle_transition_data is not None:
            filename = local_path_from_s3_or_local_path(self.oracle_transition_data)
            data = np.load(filename).item()
            print("adding data to replay buffer...")

            states, actions, next_states = data['states'], data['actions'], data['next_states']
            idx = np.random.permutation(len(states))
            states, actions, next_states = states[idx], actions[idx], next_states[idx]
            cap = self.replay_buffer.max_size
            states, actions, next_states = states[:cap], actions[:cap], next_states[:cap]

            dummy_goal = self.env.sample_goal_for_rollout()
            for (s, a, next_s, i) in zip(states, actions, next_states, range(len(states))):
                if i % 10000 == 0:
                    print(i)
                obs = dict(
                    observation=s,
                    desired_goal=dummy_goal['desired_goal'],
                    achieved_goal=s,
                    state_observation=s,
                    state_desired_goal=dummy_goal['state_desired_goal'],
                    state_achieved_goal=s,
                )
                next_obs = dict(
                    observation=next_s,
                    desired_goal=dummy_goal['desired_goal'],
                    achieved_goal=next_s,
                    state_observation=next_s,
                    state_desired_goal=dummy_goal['state_desired_goal'],
                    state_achieved_goal=next_s,
                )

                self._handle_step(
                    obs,
                    a,
                    np.array([0]),
                    next_obs,
                    np.array([0]),
                    agent_info={},
                    env_info={},
                )
                self._handle_rollout_ending()

    def train_online(self, start_epoch=0):
        self._current_path_builder = PathBuilder()
        if self.epoch_list is not None:
            iters = list(self.epoch_list)
        else:
            iters = list(range(start_epoch, self.num_epochs, self.epoch_freq))
        if self.num_epochs - 1 not in iters and self.num_epochs - 1 > iters[-1]:
            iters.append(self.num_epochs - 1)
        for epoch in gt.timed_for(
                iters,
                save_itrs=True,
        ):
            self._start_epoch(epoch)
            env_utils.mode(self.training_env, 'train')
            observation = self._start_new_rollout()
            for _ in range(self.num_env_steps_per_epoch):
                if self.do_training:
                    observation = self._take_step_in_env(observation)

                gt.stamp('sample')
                self._try_to_train()
                gt.stamp('train')
            env_utils.mode(self.env, 'eval')
            # TODO steven: move dump_tabular to be conditionally called in
            # end_epoch and move post_epoch after eval
            self._post_epoch(epoch)
            self._try_to_eval(epoch)
            gt.stamp('eval')
            self._end_epoch()

    def _take_step_in_env(self, observation):
        action, agent_info = self._get_action_and_info(
            observation,
        )
        if self.render:
            self.training_env.render()
        next_ob, reward, terminal, env_info = (
            self.training_env.step(action)
        )
        self._n_env_steps_total += 1
        self._handle_step(
            observation,
            action,
            np.array([reward]),
            next_ob,
            np.array([terminal]),
            agent_info=agent_info,
            env_info=env_info,
        )
        if isinstance(self.exploration_policy.es, OUStrategy):
            self._exploration_policy_noise = np.linalg.norm(self.exploration_policy.es.state, ord=1)
        if terminal or len(self._current_path_builder) >= self.max_path_length:
            self._handle_rollout_ending()
            new_observation = self._start_new_rollout()
        else:
            new_observation = next_ob
        return new_observation

    def train_batch(self, start_epoch):
        self._current_path_builder = PathBuilder()
        observation = self._start_new_rollout()
        for epoch in gt.timed_for(
                range(start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self._start_epoch(epoch)
            for _ in range(self.num_env_steps_per_epoch):
                action, agent_info = self._get_action_and_info(
                    observation,
                )
                if self.render:
                    self.training_env.render()
                next_ob, reward, terminal, env_info = (
                    self.training_env.step(action)
                )
                self._n_env_steps_total += 1
                terminal = np.array([terminal])
                reward = np.array([reward])
                self._handle_step(
                    observation,
                    action,
                    reward,
                    next_ob,
                    terminal,
                    agent_info=agent_info,
                    env_info=env_info,
                )
                if terminal or len(
                        self._current_path_builder) >= self.max_path_length:
                    self._handle_rollout_ending()
                    observation = self._start_new_rollout()
                else:
                    observation = next_ob

            gt.stamp('sample')
            self._try_to_train()
            gt.stamp('train')

            self._try_to_eval(epoch)
            gt.stamp('eval')
            self._end_epoch()

    def init_rollout_function(self):
        from railrl.samplers.rollout_functions import rollout
        self.train_rollout_function = rollout
        self.eval_rollout_function = self.train_rollout_function

    def train_parallel(self, start_epoch=0):
        self.parallel_env = RemoteRolloutEnv(
            env=self.env,
            train_rollout_function=self.train_rollout_function,
            eval_rollout_function=self.eval_rollout_function,
            policy=self.eval_policy,
            exploration_policy=self.exploration_policy,
            max_path_length=self.max_path_length,
            **self.parallel_env_params,
        )
        self.training_mode(False)
        last_epoch_policy = copy.deepcopy(self.policy)
        for epoch in range(start_epoch, self.num_epochs):
            self._start_epoch(epoch)
            if hasattr(self.env, "get_env_update"):
                env_update = self.env.get_env_update()
            else:
                env_update = None
            self.parallel_env.update_worker_envs(env_update)
            should_gather_data = True
            should_eval = True
            should_train = True
            last_epoch_policy.load_state_dict(self.policy.state_dict())
            eval_paths = []
            n_env_steps_current_epoch = 0
            n_eval_steps = 0
            n_train_steps = 0

            while should_gather_data or should_eval  or should_train:
                if should_gather_data:
                    path = self.parallel_env.rollout(
                        self.exploration_policy,
                        train=True,
                        discard_other_rollout_type=not should_eval,
                        epoch=epoch,
                    )
                    if path:
                        path_length = len(path['observations'])
                        self._handle_path(path)
                        n_env_steps_current_epoch += path_length
                        self._n_env_steps_total += path_length
                if should_eval:
                    # label as epoch but actually evaluating previous epoch
                    path = self.parallel_env.rollout(
                        last_epoch_policy,
                        train=False,
                        discard_other_rollout_type=not should_gather_data,
                        epoch=epoch,
                    )
                    if path:
                        eval_paths.append(dict(path))
                        n_eval_steps += len(path['observations'])
                if should_train:
                    if self._n_env_steps_total > 0:
                        self._try_to_train()
                        n_train_steps += 1
                should_gather_data &= \
                        n_env_steps_current_epoch < self.num_env_steps_per_epoch
                should_eval &= n_eval_steps < self.num_steps_per_eval

                if self.sim_throttle:
                    should_train &= self.parallel_step_ratio * n_train_steps < \
                        self.num_env_steps_per_epoch
                else:
                    should_train &= (should_eval or should_gather_data)
            self._post_epoch(epoch)
            self._try_to_eval(epoch, eval_paths=eval_paths)
            self._end_epoch()

    def train_offline(self, start_epoch=0):
        self.training_mode(False)
        params = self.get_epoch_snapshot(-1)
        logger.save_itr_params(-1, params)
        for epoch in range(start_epoch, self.num_epochs):
            self._start_epoch(epoch)
            self._try_to_train()
            self._try_to_offline_eval(epoch)
            self._end_epoch()

    def _try_to_train(self):
        if self._can_train():
            self.training_mode(True)
            for i in range(self.num_updates_per_train_call):
                self._do_training()
                self._n_train_steps_total += 1
            self.training_mode(False)

    def _try_to_eval(self, epoch, eval_paths=None):
        logger.save_extra_data(self.get_extra_data_to_save(epoch))

        params = self.get_epoch_snapshot(epoch)
        logger.save_itr_params(epoch, params)

        if self._can_evaluate():
            self.evaluate(epoch, eval_paths=eval_paths)

            # params = self.get_epoch_snapshot(epoch)
            # logger.save_itr_params(epoch, params)
            table_keys = logger.get_table_key_set()
            if self._old_table_keys is not None:
                assert table_keys == self._old_table_keys, (
                    "Table keys cannot change from iteration to iteration."
                )
            self._old_table_keys = table_keys

            logger.record_tabular(
                "Number of train steps total",
                self._n_train_steps_total,
            )
            logger.record_tabular(
                "Number of env steps total",
                self._n_env_steps_total,
            )
            logger.record_tabular(
                "Number of rollouts total",
                self._n_rollouts_total,
            )

            if self.collection_mode != 'online-parallel':
                times_itrs = gt.get_times().stamps.itrs
                train_time = times_itrs['train'][-1]
                sample_time = times_itrs['sample'][-1]
                if 'eval' in times_itrs:
                    eval_time = times_itrs['eval'][-1] if epoch > 0 else -1
                else:
                    eval_time = -1
                epoch_time = train_time + sample_time + eval_time
                total_time = gt.get_times().total

                logger.record_tabular('Train Time (s)', train_time)
                logger.record_tabular('(Previous) Eval Time (s)', eval_time)
                logger.record_tabular('Sample Time (s)', sample_time)
                logger.record_tabular('Epoch Time (s)', epoch_time)
                logger.record_tabular('Total Train Time (s)', total_time)
            else:
                logger.record_tabular('Epoch Time (s)',
                                      time.time() - self._epoch_start_time)
            logger.record_tabular("Epoch", epoch)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
        else:
            logger.log("Skipping eval for now.")

    def _try_to_offline_eval(self, epoch):
        start_time = time.time()
        logger.save_extra_data(self.get_extra_data_to_save(epoch))
        self.offline_evaluate(epoch)
        params = self.get_epoch_snapshot(epoch)
        logger.save_itr_params(epoch, params)
        table_keys = logger.get_table_key_set()
        if self._old_table_keys is not None:
            assert table_keys == self._old_table_keys, (
                "Table keys cannot change from iteration to iteration."
            )
        self._old_table_keys = table_keys
        logger.dump_tabular(with_prefix=False, with_timestamp=False)
        logger.log("Eval Time: {0}".format(time.time() - start_time))

    def evaluate(self, epoch, eval_paths=None):
        statistics = OrderedDict()
        statistics.update(self.eval_statistics)

        logger.log("Collecting samples for evaluation")
        if eval_paths:
            test_paths = eval_paths
        else:
            test_paths = self.get_eval_paths()
        statistics.update(eval_util.get_generic_path_information(
            test_paths, stat_prefix="Test",
        ))
        if len(self._exploration_paths) > 0:
            statistics.update(eval_util.get_generic_path_information(
                self._exploration_paths, stat_prefix="Exploration",
            ))
        if hasattr(self.env, "log_diagnostics"):
            self.env.log_diagnostics(test_paths, logger=logger)
        if hasattr(self.env, "get_diagnostics"):
            statistics.update(self.env.get_diagnostics(test_paths))

        if hasattr(self.eval_policy, "log_diagnostics"):
            self.eval_policy.log_diagnostics(test_paths, logger=logger)
        if hasattr(self.eval_policy, "get_diagnostics"):
            statistics.update(self.eval_policy.get_diagnostics(test_paths))

        process = psutil.Process(os.getpid())
        statistics['RAM Usage (Mb)'] = int(process.memory_info().rss / 1000000)

        statistics['Exploration Policy Noise'] = self._exploration_policy_noise

        average_returns = eval_util.get_average_returns(test_paths)
        statistics['AverageReturn'] = average_returns
        for key, value in statistics.items():
            logger.record_tabular(key, value)
        self.need_to_update_eval_statistics = True

    def get_eval_paths(self):
        paths = self.eval_sampler.obtain_samples()
        if self.train_on_eval_paths:
            for path in paths:
                self._handle_path(path)
        return paths

    def offline_evaluate(self, epoch):
        for key, value in self.eval_statistics.items():
            logger.record_tabular(key, value)
        self.need_to_update_eval_statistics = True

    def _can_evaluate(self):
        """
        One annoying thing about the logger table is that the keys at each
        iteration need to be the exact same. So unless you can compute
        everything, skip evaluation.
        """
        return (
            not self.do_training or
            (len(self._exploration_paths) > 0 and not self.need_to_update_eval_statistics)
        )

    def _can_train(self):
        return (
            self.do_training and
            self.replay_buffer.num_steps_can_sample() >=
            self.min_num_steps_before_training
        )

    def _get_action_and_info(self, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        self.exploration_policy.set_num_steps_total(self._n_env_steps_total)
        return self.exploration_policy.get_action(
            observation,
        )

    def _start_epoch(self, epoch):
        self._epoch_start_time = time.time()
        self._exploration_paths = []
        self._do_train_time = 0
        logger.push_prefix('Iteration #%d | ' % epoch)

    def _end_epoch(self):
        logger.log("Epoch Duration: {0}".format(
            time.time() - self._epoch_start_time
        ))
        logger.log("Started Training: {0}".format(self._can_train()))
        logger.pop_prefix()

    def _post_epoch(self, epoch):
        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, epoch)

    def _start_new_rollout(self):
        self.exploration_policy.reset()
        return self.training_env.reset()

    def _handle_path(self, path):
        """
        Naive implementation: just loop through each transition.
        :param path:
        :return:
        """
        for (
                ob,
                action,
                reward,
                next_ob,
                terminal,
                agent_info,
                env_info
        ) in zip(
            path["observations"],
            path["actions"],
            path["rewards"],
            path["next_observations"],
            path["terminals"],
            path["agent_infos"],
            path["env_infos"],
        ):
            self._handle_step(
                ob,
                action,
                reward,
                next_ob,
                terminal,
                agent_info=agent_info,
                env_info=env_info,
            )
        self._handle_rollout_ending()

    def _handle_step(
            self,
            observation,
            action,
            reward,
            next_observation,
            terminal,
            agent_info,
            env_info,
    ):
        """
        Implement anything that needs to happen after every step
        :return:
        """
        self._current_path_builder.add_all(
            observations=observation,
            actions=action,
            rewards=reward,
            next_observations=next_observation,
            terminals=terminal,
            agent_infos=agent_info,
            env_infos=env_info,
        )
        self.replay_buffer.add_sample(
            observation=observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation,
            agent_info=agent_info,
            env_info=env_info,
        )

    def _handle_rollout_ending(self):
        """
        Implement anything that needs to happen after every rollout.
        """
        self.replay_buffer.terminate_episode()
        self._n_rollouts_total += 1
        if len(self._current_path_builder) > 0:
            self._exploration_paths.append(
                self._current_path_builder.get_all_stacked()
            )
            self._current_path_builder = PathBuilder()

    def get_epoch_snapshot(self, epoch):
        data_to_save = dict(
            epoch=epoch,
            n_env_steps_total=self._n_env_steps_total,
            exploration_policy=self.exploration_policy,
            eval_policy=self.eval_policy,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        return data_to_save

    def get_extra_data_to_save(self, epoch):
        """
        Save things that shouldn't be saved every snapshot but rather
        overwritten every time.
        :param epoch:
        :return:
        """
        data_to_save = dict(
            epoch=epoch,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        if self.save_replay_buffer:
            data_to_save['replay_buffer'] = self.replay_buffer
        if self.save_algorithm:
            data_to_save['algorithm'] = self
        return data_to_save

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass

    @abc.abstractmethod
    def cuda(self):
        """
        Turn cuda on.
        :return:
        """
        pass

    @abc.abstractmethod
    def _do_training(self):
        """
        Perform some update, e.g. perform one gradient step.
        :return:
        """
        pass
