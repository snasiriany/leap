import numpy as np
import random

from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from railrl.data_management.segment_tree import SumSegmentTree, MinSegmentTree
from railrl.misc.np_util import truncated_geometric


class HerReplayBuffer(EnvReplayBuffer):
    """
    Save goals from the same trajectory into the replay buffer.
    Only add_path is implemented.

    Implementation details:
     - Every sample from [0, self._size] will be valid.
    """
    def __init__(
            self,
            max_size,
            env,
            num_goals_to_sample=4,
            fraction_goals_are_rollout_goals=None,
            resampling_strategy='uniform',
            truncated_geom_factor=1.,
    ):
        """

        :param max_size:
        :param observation_dim:
        :param action_dim:
        :param num_goals_to_sample:
        :param fraction_goals_are_rollout_goals:
        :param resampling_strategy: How to resample states from the rest of
        the trajectory?
        - 'uniform': Sample them uniformly
        - 'truncated_geometric': Used a truncated geometric distribution
        """
        assert resampling_strategy in [
            'uniform',
            'truncated_geometric',
        ]
        super().__init__(max_size, env)
        self.num_goals_to_sample = num_goals_to_sample
        self._goals = np.zeros((max_size, self.env.goal_dim))
        self._num_steps_left = np.zeros((max_size, 1))
        if fraction_goals_are_rollout_goals is None:
            fraction_goals_are_rollout_goals = (
                1. / num_goals_to_sample
            )
        self.fraction_goals_are_rollout_goals = (
            fraction_goals_are_rollout_goals
        )
        self.truncated_geom_factor = float(truncated_geom_factor)
        self.resampling_strategy = resampling_strategy

        # Let j be any index in self._idx_to_future_obs_idx[i]
        # Then self._next_obs[j] is a valid next observation for observation i
        self._idx_to_future_obs_idx = [None] * max_size

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        raise NotImplementedError("Only use add_path")

    def add_path(self, path):
        obs = path["observations"]
        actions = path["actions"]
        rewards = path["rewards"]
        next_obs = path["next_observations"]
        terminals = path["terminals"]
        goals = path["goals"]
        num_steps_left = path["num_steps_left"]
        path_len = len(rewards)

        actions = flatten_n(actions)
        obs = flatten_n(obs)
        next_obs = flatten_n(next_obs)

        if self._top + path_len >= self._max_replay_buffer_size:
            num_pre_wrap_steps = self._max_replay_buffer_size - self._top
            # numpy slice
            pre_wrap_buffer_slice = np.s_[
                self._top:self._top + num_pre_wrap_steps, :
            ]
            pre_wrap_path_slice = np.s_[0:num_pre_wrap_steps, :]

            num_post_wrap_steps = path_len - num_pre_wrap_steps
            post_wrap_buffer_slice = slice(0, num_post_wrap_steps)
            post_wrap_path_slice = slice(num_pre_wrap_steps, path_len)
            for buffer_slice, path_slice in [
                (pre_wrap_buffer_slice, pre_wrap_path_slice),
                (post_wrap_buffer_slice, post_wrap_path_slice),
            ]:
                self._observations[buffer_slice] = obs[path_slice]
                self._actions[buffer_slice] = actions[path_slice]
                self._rewards[buffer_slice] = rewards[path_slice]
                self._next_obs[buffer_slice] = next_obs[path_slice]
                self._terminals[buffer_slice] = terminals[path_slice]
                self._goals[buffer_slice] = goals[path_slice]
                self._num_steps_left[buffer_slice] = num_steps_left[path_slice]
            # Pointers from before the wrap
            for i in range(self._top, self._max_replay_buffer_size):
                self._idx_to_future_obs_idx[i] = np.hstack((
                    # Pre-wrap indices
                    np.arange(i, self._max_replay_buffer_size),
                    # Post-wrap indices
                    np.arange(0, num_post_wrap_steps)
                ))
            # Pointers after the wrap
            for i in range(0, num_post_wrap_steps):
                self._idx_to_future_obs_idx[i] = np.arange(
                    i,
                    num_post_wrap_steps,
                )
        else:
            slc = np.s_[self._top:self._top + path_len, :]
            self._observations[slc] = obs
            self._actions[slc] = actions
            self._rewards[slc] = rewards
            self._next_obs[slc] = next_obs
            self._terminals[slc] = terminals
            self._goals[slc] = goals
            self._num_steps_left[slc] = num_steps_left
            for i in range(self._top, self._top + path_len):
                self._idx_to_future_obs_idx[i] = np.arange(
                    i, self._top + path_len
                )
        self._top = (self._top + path_len) % self._max_replay_buffer_size
        self._size = min(self._size + path_len, self._max_replay_buffer_size)

    def _sample_indices(self, batch_size):
        return np.random.randint(0, self._size, batch_size)

    def random_batch(self, batch_size):
        indices = self._sample_indices(batch_size)
        next_obs_idxs = []
        for i in indices:
            possible_next_obs_idxs = self._idx_to_future_obs_idx[i]
            # This is generally faster than random.choice. Makes you wonder what
            # random.choice is doing
            num_options = len(possible_next_obs_idxs)
            if num_options == 1:
                next_obs_i = 0
            else:
                if self.resampling_strategy == 'uniform':
                    next_obs_i = int(np.random.randint(0, num_options))
                elif self.resampling_strategy == 'truncated_geometric':
                    next_obs_i = int(truncated_geometric(
                        p=self.truncated_geom_factor/num_options,
                        truncate_threshold=num_options-1,
                        size=1,
                        new_value=0
                    ))
                else:
                    raise ValueError("Invalid resampling strategy: {}".format(
                        self.resampling_strategy
                    ))
            next_obs_idxs.append(possible_next_obs_idxs[next_obs_i])
        next_obs_idxs = np.array(next_obs_idxs)
        resampled_goals = self.env.convert_obs_to_goals(
            self._next_obs[next_obs_idxs]
        )
        num_goals_are_from_rollout = int(
            batch_size * self.fraction_goals_are_rollout_goals
        )
        if num_goals_are_from_rollout > 0:
            resampled_goals[:num_goals_are_from_rollout] = self._goals[
                indices[:num_goals_are_from_rollout]
            ]
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
            goals_used_for_rollout=self._goals[indices],
            resampled_goals=resampled_goals,
            num_steps_left=self._num_steps_left[indices],
            indices=np.array(indices).reshape(-1, 1),
        )


class PrioritizedHerReplayBuffer(HerReplayBuffer):
    """
    Building off of openai baselines code
    """
    def __init__(
            self,
            max_size,
            env,
            alpha=0.6,
            eps=1e-6,
            **kwargs
    ):
        super().__init__(max_size, env, **kwargs)
        assert alpha > 0
        self._alpha = alpha
        self._eps = eps

        it_capacity = 1
        while it_capacity < max_size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add_path(self, path):
        """
        This implementation is the same, but I added a few:
        ```
        self._it_sum[i] = self._max_priority ** self._alpha
        self._it_min[i] = self._max_priority ** self._alpha
        ```
        """
        obs = path["observations"]
        actions = path["actions"]
        rewards = path["rewards"]
        next_obs = path["next_observations"]
        terminals = path["terminals"]
        goals = path["goals"]
        num_steps_left = path["num_steps_left"]
        path_len = len(rewards)

        actions = flatten_n(actions)
        obs = flatten_n(obs)
        next_obs = flatten_n(next_obs)

        if self._top + path_len >= self._max_replay_buffer_size:
            num_pre_wrap_steps = self._max_replay_buffer_size - self._top
            # numpy slice
            pre_wrap_buffer_slice = np.s_[
                                    self._top:self._top + num_pre_wrap_steps, :
                                    ]
            pre_wrap_path_slice = np.s_[0:num_pre_wrap_steps, :]

            num_post_wrap_steps = path_len - num_pre_wrap_steps
            post_wrap_buffer_slice = slice(0, num_post_wrap_steps)
            post_wrap_path_slice = slice(num_pre_wrap_steps, path_len)
            for buffer_slice, path_slice in [
                (pre_wrap_buffer_slice, pre_wrap_path_slice),
                (post_wrap_buffer_slice, post_wrap_path_slice),
            ]:
                self._observations[buffer_slice] = obs[path_slice]
                self._actions[buffer_slice] = actions[path_slice]
                self._rewards[buffer_slice] = rewards[path_slice]
                self._next_obs[buffer_slice] = next_obs[path_slice]
                self._terminals[buffer_slice] = terminals[path_slice]
                self._goals[buffer_slice] = goals[path_slice]
                self._num_steps_left[buffer_slice] = num_steps_left[path_slice]
            # Pointers from before the wrap
            for i in range(self._top, self._max_replay_buffer_size):
                self._idx_to_future_obs_idx[i] = np.hstack((
                    # Pre-wrap indices
                    np.arange(i, self._max_replay_buffer_size),
                    # Post-wrap indices
                    np.arange(0, num_post_wrap_steps)
                ))
                self._it_sum[i] = self._max_priority ** self._alpha
                self._it_min[i] = self._max_priority ** self._alpha
            # Pointers after the wrap
            for i in range(0, num_post_wrap_steps):
                self._idx_to_future_obs_idx[i] = np.arange(
                    i,
                    num_post_wrap_steps,
                )
                self._it_sum[i] = self._max_priority ** self._alpha
                self._it_min[i] = self._max_priority ** self._alpha
        else:
            slc = np.s_[self._top:self._top + path_len, :]
            self._observations[slc] = obs
            self._actions[slc] = actions
            self._rewards[slc] = rewards
            self._next_obs[slc] = next_obs
            self._terminals[slc] = terminals
            self._goals[slc] = goals
            self._num_steps_left[slc] = num_steps_left
            for i in range(self._top, self._top + path_len):
                self._idx_to_future_obs_idx[i] = np.arange(
                    i, self._top + path_len
                )
                self._it_sum[i] = self._max_priority ** self._alpha
                self._it_min[i] = self._max_priority ** self._alpha
        self._top = (self._top + path_len) % self._max_replay_buffer_size
        self._size = min(self._size + path_len, self._max_replay_buffer_size)

    def random_batch(self, batch_size, beta=1):
        batch = super().random_batch(batch_size)
        weights = self._get_weights(batch['indices'], beta)
        batch['is_weights'] = weights
        return batch

    def _get_weights(self, idxes, beta):
        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * self._size) ** (-beta)

        for idx in idxes:
            idx = int(idx)
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * self._size) ** (-beta)
            weights.append(weight / max_weight)
        return np.array(weights).reshape(-1, 1)

    def _sample_indices(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, self._size - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            idx = int(idx)
            priority = priority + self._eps
            assert priority > 0
            assert 0 <= idx < self._size
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


class SimplePrioritizedHerReplayBuffer(PrioritizedHerReplayBuffer):
    def __init__(
            self,
            max_size,
            env,
            max_time_to_next_goal=None,
            **kwargs
    ):
        super().__init__(max_size, env, **kwargs)
        if max_time_to_next_goal is not None:
            assert max_time_to_next_goal >= self.num_goals_to_sample
        self.max_time_to_next_goal = max_time_to_next_goal
        self.last_path_start_idx = None


    def add_sample(self, observation, action, reward, terminal,
                   next_observation, goal, num_steps_left, **kwargs):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation
        self._goals[self._top] = goal
        self._num_steps_left[self._top] = num_steps_left
        self._it_sum[self._top] = self._max_priority ** self._alpha
        self._it_min[self._top] = self._max_priority ** self._alpha
        self._advance()

    def add_path(self, path):
        self.last_path_start_idx = self._top
        obs = path["observations"]
        actions = path["actions"]
        rewards = path["rewards"]
        next_obs = path["next_observations"]
        terminals = path["terminals"]
        goals = path["goals"]
        num_steps_left = path["num_steps_left"]
        path_len = len(terminals)

        actions = flatten_n(actions)
        obs = flatten_n(obs)
        next_obs = flatten_n(next_obs)

        for i, (
            observation,
            action,
            reward,
            terminal,
            next_observation,
            goal,
            num_steps_left,
        ) in enumerate(zip(
            obs,
            actions,
            rewards,
            terminals,
            next_obs,
            goals,
            num_steps_left,
        )):
            num_goals_to_sample = min(self.num_goals_to_sample, path_len-i)
            if self.max_time_to_next_goal is None:
                max_i = path_len
            else:
                # Don't add +1 because we index into next_obs and not ob
                max_i = min(i + self.max_time_to_next_goal, path_len)
            if self.max_time_to_next_goal == self.num_goals_to_sample:
                goal_idxs = np.arange(
                    i,
                    min(i + self.num_goals_to_sample, path_len)
                )
            else:
                goal_idxs = np.random.randint(i, max_i, num_goals_to_sample)
            for goal_i in goal_idxs:
                # Add both to the replay buffer to keep it balanced between
                # HER data and real data.
                self.add_sample(observation, action, reward, terminal,
                                next_observation, goal, np.array([goal_i - i]))
                self.add_sample(observation, action, reward, terminal,
                                next_observation, next_obs[goal_i],
                                np.array([goal_i - i]))

    def random_batch(self, batch_size, beta=1.0):
        indices = self._sample_indices(batch_size)
        weights = self._get_weights(indices, beta)
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
            goals=self._goals[indices],
            num_steps_left=self._num_steps_left[indices],
            indices=np.array(indices).reshape(-1, 1),
            is_weights=weights.reshape(-1, 1),
        )

    def most_recent_path_batch(self, beta=1.0):
        last_path_start_idx = self.last_path_start_idx
        # We just looped around the replay buffer
        if last_path_start_idx > self._top:
            last_path_start_idx = 0

        indices = np.arange(last_path_start_idx, self._top)
        weights = self._get_weights(indices, beta)
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
            goals=self._goals[indices],
            num_steps_left=self._num_steps_left[indices],
            indices=np.array(indices).reshape(-1, 1),
            is_weights=weights.reshape(-1, 1),
        )

class SimpleHerReplayBuffer(EnvReplayBuffer):
    """
    Only add relabel goals when putting paths into the replay buffer,
    like in the orignial HER paper.
    """
    def __init__(
            self,
            max_size,
            env,
            num_goals_to_sample=4,
            **kwargs
    ):
        super().__init__(max_size, env, **kwargs)
        self.num_goals_to_sample = num_goals_to_sample
        self._goals = np.zeros((max_size, self.env.goal_dim))

    def add_path(self, path):
        self.last_path_start_idx = self._top
        obs = path["observations"]
        actions = path["actions"]
        rewards = path["rewards"]
        next_obs = path["next_observations"]
        terminals = path["terminals"]
        goals = path["goals"]
        env_infos = path["env_infos"]
        path_len = len(terminals)

        for i, (
                observation,
                action,
                reward,
                terminal,
                next_observation,
                goal,
                env_info,
        ) in enumerate(zip(
            obs,
            actions,
            rewards,
            terminals,
            next_obs,
            goals,
            env_infos,
        )):
            # It's not really necessary to recompute the reward, but just to
            # make sure I always use the same reward as in HER
            new_reward = self.env.compute_her_reward_np(
                observation,
                action,
                next_observation,
                goal,
                env_info
            )
            self.add_sample(
                observation, action, new_reward, terminal,
                next_observation, goal, env_info=env_info
            )
            num_goals_to_sample = min(self.num_goals_to_sample, path_len-i)
            max_i = path_len
            goal_idxs = np.random.randint(i, max_i, num_goals_to_sample)
            for goal_i in goal_idxs:
                # Add both to the replay buffer to keep it balanced between
                # HER data and real data.
                new_goal = self.env.convert_ob_to_goal(next_obs[goal_i])
                new_reward = self.env.compute_her_reward_np(
                    observation,
                    action,
                    next_observation,
                    new_goal,
                    env_info
                )
                self.add_sample(
                    observation, action, new_reward, terminal,
                    next_observation, new_goal, env_info=env_info
                )

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, goal, **kwargs):
        self._goals[self._top] = goal
        super().add_sample(
            observation, action, reward, terminal, next_observation, **kwargs
        )

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self._size, batch_size)
        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
            goals=self._goals[indices],
        )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        return batch


class RelabelingReplayBuffer(EnvReplayBuffer):
    """
    Save goals from the same trajectory into the replay buffer.
    Only add_path is implemented.
    Implementation details:
     - Every sample from [0, self._size] will be valid.
    """
    def __init__(
            self,
            max_size,
            env,
            fraction_goals_are_rollout_goals=1.0, # default, no HER
            fraction_resampled_goals_are_env_goals=0.0, # this many goals are just sampled from environment directly
            resampling_strategy='uniform', # 'uniform' is the HER 'future' strategy
            truncated_geom_factor=1.,
            **kwargs
    ):
        """
        :param resampling_strategy: How to resample states from the rest of
        the trajectory?
        - 'uniform': Sample them uniformly
        - 'truncated_geometric': Used a truncated geometric distribution
        """
        assert resampling_strategy in [
            'uniform',
            'truncated_geometric',
        ]
        super().__init__(max_size, env, **kwargs)
        self._goals = np.zeros((max_size, self.env.goal_dim))
        self._num_steps_left = np.zeros((max_size, 1))
        self.fraction_goals_are_rollout_goals = fraction_goals_are_rollout_goals
        self.fraction_resampled_goals_are_env_goals = fraction_resampled_goals_are_env_goals
        self.truncated_geom_factor = float(truncated_geom_factor)
        self.resampling_strategy = resampling_strategy

        # Let j be any index in self._idx_to_future_obs_idx[i]
        # Then self._next_obs[j] is a valid next observation for observation i
        self._idx_to_future_obs_idx = [None] * max_size

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        raise NotImplementedError("Only use add_path")

    def add_path(self, path):
        obs = path["observations"]
        actions = path["actions"]
        rewards = path["rewards"]
        next_obs = path["next_observations"]
        terminals = path["terminals"]
        goals = path["goals"]
        num_steps_left = path["rewards"].copy() # path["num_steps_left"] # irrelevant for non-TDM
        env_infos = np.array(path["env_infos"])
        path_len = len(rewards)

        actions = flatten_n(actions)
        obs = flatten_n(obs)
        next_obs = flatten_n(next_obs)
        env_infos = flatten_env_info(env_infos, self._env_info_keys)

        if self._top + path_len >= self._max_replay_buffer_size:
            num_pre_wrap_steps = self._max_replay_buffer_size - self._top
            # numpy slice
            pre_wrap_buffer_slice = np.s_[
                self._top:self._top + num_pre_wrap_steps, :
            ]
            pre_wrap_path_slice = np.s_[0:num_pre_wrap_steps, :]

            num_post_wrap_steps = path_len - num_pre_wrap_steps
            post_wrap_buffer_slice = slice(0, num_post_wrap_steps)
            post_wrap_path_slice = slice(num_pre_wrap_steps, path_len)
            for buffer_slice, path_slice in [
                (pre_wrap_buffer_slice, pre_wrap_path_slice),
                (post_wrap_buffer_slice, post_wrap_path_slice),
            ]:
                self._observations[buffer_slice] = obs[path_slice]
                self._actions[buffer_slice] = actions[path_slice]
                self._rewards[buffer_slice] = rewards[path_slice]
                self._next_obs[buffer_slice] = next_obs[path_slice]
                self._terminals[buffer_slice] = terminals[path_slice]
                self._goals[buffer_slice] = goals[path_slice]
                self._num_steps_left[buffer_slice] = num_steps_left[path_slice]
                for key in self._env_info_keys:
                    self._env_infos[key][buffer_slice] = env_infos[key][path_slice]
            # Pointers from before the wrap
            for i in range(self._top, self._max_replay_buffer_size):
                self._idx_to_future_obs_idx[i] = np.hstack((
                    # Pre-wrap indices
                    np.arange(i, self._max_replay_buffer_size),
                    # Post-wrap indices
                    np.arange(0, num_post_wrap_steps)
                ))
            # Pointers after the wrap
            for i in range(0, num_post_wrap_steps):
                self._idx_to_future_obs_idx[i] = np.arange(
                    i,
                    num_post_wrap_steps,
                )
        else:
            slc = np.s_[self._top:self._top + path_len, :]
            self._observations[slc] = obs
            self._actions[slc] = actions
            self._rewards[slc] = rewards
            self._next_obs[slc] = next_obs
            self._terminals[slc] = terminals
            self._goals[slc] = goals
            self._num_steps_left[slc] = num_steps_left
            for key in self._env_info_keys:
                self._env_infos[key][slc] = env_infos[key]
            for i in range(self._top, self._top + path_len):
                self._idx_to_future_obs_idx[i] = np.arange(
                    i, self._top + path_len
                )
        self._top = (self._top + path_len) % self._max_replay_buffer_size
        self._size = min(self._size + path_len, self._max_replay_buffer_size)

    def _sample_indices(self, batch_size):
        return np.random.randint(0, self._size, batch_size)

    def random_batch(self, batch_size):
        indices = self._sample_indices(batch_size)
        next_obs_idxs = []
        for i in indices:
            possible_next_obs_idxs = self._idx_to_future_obs_idx[i]
            # This is generally faster than random.choice. Makes you wonder what
            # random.choice is doing
            num_options = len(possible_next_obs_idxs)
            if num_options == 1:
                next_obs_i = 0
            else:
                if self.resampling_strategy == 'uniform':
                    next_obs_i = int(np.random.randint(0, num_options))
                elif self.resampling_strategy == 'truncated_geometric':
                    next_obs_i = int(truncated_geometric(
                        p=self.truncated_geom_factor/num_options,
                        truncate_threshold=num_options-1,
                        size=1,
                        new_value=0
                    ))
                else:
                    raise ValueError("Invalid resampling strategy: {}".format(
                        self.resampling_strategy
                    ))
            next_obs_idxs.append(possible_next_obs_idxs[next_obs_i])
        next_obs_idxs = np.array(next_obs_idxs)
        resampled_goals = self.env.convert_obs_to_goals(
            self._next_obs[next_obs_idxs]
        )
        num_goals_are_from_rollout = int(
            batch_size * self.fraction_goals_are_rollout_goals
        )
        if num_goals_are_from_rollout > 0:
            resampled_goals[:num_goals_are_from_rollout] = self._goals[
                indices[:num_goals_are_from_rollout]
            ]
        # recompute rewards
        new_obs = self._observations[indices]
        new_next_obs = self._next_obs[indices]
        new_actions = self._actions[indices]
        new_rewards = self._rewards[indices].copy() # needs to be recomputed
        env_info_dicts = [self.rebuild_env_info_dict(idx) for idx in indices]
        random_numbers = np.random.rand(batch_size)
        for i in range(batch_size):
            if random_numbers[i] < self.fraction_resampled_goals_are_env_goals:
                resampled_goals[i, :] = self.env.sample_goal_for_rollout() # env_goals[i, :]

            new_reward = self.env.compute_her_reward_np(
                new_obs[i, :],
                new_actions[i, :],
                new_next_obs[i, :],
                resampled_goals[i, :],
                env_info_dicts[i],
            )
            new_rewards[i] = new_reward
        # new_rewards = self.env.computer_her_reward_np_batch(
        #     new_obs,
        #     new_actions,
        #     new_next_obs,
        #     resampled_goals,
        #     env_infos,
        # )

        batch = dict(
            observations=new_obs,
            actions=new_actions,
            rewards=new_rewards,
            terminals=self._terminals[indices],
            next_observations=new_next_obs,
            goals_used_for_rollout=self._goals[indices],
            resampled_goals=resampled_goals,
            num_steps_left=self._num_steps_left[indices],
            indices=np.array(indices).reshape(-1, 1),
            goals=resampled_goals,
        )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        return batch


def flatten_n(xs):
    xs = np.asarray(xs)
    return xs.reshape((xs.shape[0], -1))


def flatten_env_info(env_infos, env_info_keys):
# Turns list of env_info dicts into env_info dict of 2D np arrays
    return {
        key: flatten_n(
                [env_infos[i][key] for i in range(len(env_infos))]
        )
        for key in env_info_keys
    }
