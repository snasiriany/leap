from collections import OrderedDict

import numpy as np
import torch
from torch import optim

import railrl.torch.pytorch_util as ptu
from railrl.misc.eval_util import create_stats_ordered_dict
from railrl.state_distance.tdm import TemporalDifferenceModel
from railrl.torch.td3.td3 import TD3

from torch.nn import functional as F


class TdmTd3(TemporalDifferenceModel, TD3):
    def __init__(
            self,
            env,
            qf1,
            qf2,
            exploration_policy,
            td3_kwargs,
            tdm_kwargs,
            base_kwargs,
            policy=None,
            eval_policy=None,
            replay_buffer=None,
            optimizer_class=optim.Adam,
            use_policy_saturation_cost=False,
            **kwargs
    ):
        TD3.__init__(
            self,
            env=env,
            qf1=qf1,
            qf2=qf2,
            policy=policy,
            exploration_policy=exploration_policy,
            replay_buffer=replay_buffer,
            eval_policy=eval_policy,
            optimizer_class=optimizer_class,
            **td3_kwargs,
            **base_kwargs
        )
        super().__init__(**tdm_kwargs)
        self.use_policy_saturation_cost = use_policy_saturation_cost

    def _do_training(self):
        batch = self.get_batch()
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        goals = batch['goals']
        num_steps_left = batch['num_steps_left']

        """
        Critic operations.
        """
        next_actions = self.target_policy(
            observations=next_obs,
            goals=goals,
            num_steps_left=num_steps_left-1,
        )
        noise = torch.normal(
            torch.zeros_like(next_actions),
            self.target_policy_noise,
        )
        noise = torch.clamp(
            noise,
            -self.target_policy_noise_clip,
            self.target_policy_noise_clip
        )
        noisy_next_actions = next_actions + noise

        target_q1_values = self.target_qf1(
            observations=next_obs,
            actions=noisy_next_actions,
            goals=goals,
            num_steps_left=num_steps_left-1,
        )
        target_q2_values = self.target_qf2(
            observations=next_obs,
            actions=noisy_next_actions,
            goals=goals,
            num_steps_left=num_steps_left-1,
        )
        target_q_values = torch.min(target_q1_values, target_q2_values)
        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        q_target = q_target.detach()

        q1_pred = self.qf1(
            observations=obs,
            actions=actions,
            goals=goals,
            num_steps_left=num_steps_left,
        )
        q2_pred = self.qf2(
            observations=obs,
            actions=actions,
            goals=goals,
            num_steps_left=num_steps_left,
        )

        bellman_errors_1 = (q1_pred - q_target) ** 2
        bellman_errors_2 = (q2_pred - q_target) ** 2
        qf1_loss = bellman_errors_1.mean()
        qf2_loss = bellman_errors_2.mean()

        """
        Update Networks
        """
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        policy_actions, pre_tanh_value = self.policy(
            obs, goals, num_steps_left, return_preactivations=True,
        )
        policy_saturation_cost = F.relu(torch.abs(pre_tanh_value) - 20.0)
        q_output = self.qf1(
            observations=obs,
            actions=policy_actions,
            num_steps_left=num_steps_left,
            goals=goals,
        )

        policy_loss = - q_output.mean()
        if self.use_policy_saturation_cost:
            policy_loss = policy_loss + policy_saturation_cost.mean()

        if self._n_train_steps_total % self.policy_and_target_update_period == 0:
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            ptu.soft_update_from_to(self.policy, self.target_policy, self.tau)
            ptu.soft_update_from_to(self.qf1, self.target_qf1, self.tau)
            ptu.soft_update_from_to(self.qf2, self.target_qf2, self.tau)

        if self.need_to_update_eval_statistics:
            self.need_to_update_eval_statistics = False
            self.eval_statistics = OrderedDict()
            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Bellman1 Errors',
                ptu.get_numpy(bellman_errors_1),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Bellman2 Errors',
                ptu.get_numpy(bellman_errors_2),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy Saturation Cost',
                ptu.get_numpy(policy_saturation_cost),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy Action',
                ptu.get_numpy(policy_actions),
                exclude_abs=False,
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy Action Pre-tanh',
                ptu.get_numpy(pre_tanh_value),
                exclude_abs=False,
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy Q Output',
                ptu.get_numpy(q_output),
                exclude_abs=False,
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy Noisy Next Actions',
                ptu.get_numpy(noisy_next_actions),
                exclude_abs=False,
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Replay Buffer Actions',
                ptu.get_numpy(actions),
                exclude_abs=False,
            ))

            be_np = ptu.get_numpy(bellman_errors_1)
            num_steps_left_np = ptu.get_numpy(num_steps_left)

            ### tau == 0 ###
            idx_0 = np.argwhere(num_steps_left_np == 0)
            be_0 = 0
            if len(idx_0) > 0:
                be_0 = be_np[idx_0[:,0]]
            self.eval_statistics['QF1 Loss tau=0'] = np.mean(be_0)

            ### tau == 1 ###
            idx_1 = np.argwhere(num_steps_left_np == 1)
            be_1 = 0
            if len(idx_1) > 0:
                be_1 = be_np[idx_1[:,0]]
            self.eval_statistics['QF1 Loss tau=1'] = np.mean(be_1)

            ### tau in [2, 5) ###
            idx_2_to_5 = np.argwhere(np.logical_and(num_steps_left_np >= 2, num_steps_left_np < 5))
            be_2_to_5 = 0
            if len(idx_2_to_5) > 0:
                be_2_to_5 = be_np[idx_2_to_5[:,0]]
            self.eval_statistics['QF1 Loss tau=2_to_5'] = np.mean(be_2_to_5)

            ### tau in [5, 10) ###
            idx_5_to_10 = np.argwhere(np.logical_and(num_steps_left_np >= 5, num_steps_left_np < 10))
            be_5_to_10 = 0
            if len(idx_5_to_10) > 0:
                be_5_to_10 = be_np[idx_5_to_10[:,0]]
            self.eval_statistics['QF1 Loss tau=5_to_10'] = np.mean(be_5_to_10)

            ### tau in [10, max_tau] ###
            idx_10_to_end = np.argwhere(np.logical_and(num_steps_left_np >= 10, num_steps_left_np < self.max_tau+1))
            be_10_to_end = 0
            if len(idx_10_to_end) > 0:
                be_10_to_end = be_np[idx_10_to_end[:,0]]
            self.eval_statistics['QF1 Loss tau=10_to_end'] = np.mean(be_10_to_end)

