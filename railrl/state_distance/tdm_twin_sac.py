import numpy as np
import torch
from torch import optim

import railrl.torch.pytorch_util as ptu
from railrl.misc.eval_util import create_stats_ordered_dict
from railrl.state_distance.tdm import TemporalDifferenceModel
from railrl.torch.sac.twin_sac import TwinSAC


class TdmTwinSAC(TemporalDifferenceModel, TwinSAC):
    def __init__(
            self,
            env,
            qf1,
            qf2,
            vf,
            twin_sac_kwargs,
            tdm_kwargs,
            base_kwargs,
            policy=None,
            eval_policy=None,
            replay_buffer=None,
            dense_log_pi=True,
            optimizer_class=optim.Adam,
            **kwargs
    ):
        TwinSAC.__init__(
            self,
            env=env,
            qf1=qf1,
            qf2=qf2,
            vf=vf,
            policy=policy,
            replay_buffer=replay_buffer,
            eval_policy=eval_policy,
            optimizer_class=optimizer_class,
            **twin_sac_kwargs,
            **base_kwargs
        )
        super().__init__(**tdm_kwargs)
        self.dense_log_pi = dense_log_pi

    def _do_training(self):
        batch = self.get_batch()
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        goals = batch['goals']
        num_steps_left = batch['num_steps_left']

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
        # Make sure policy accounts for squashing functions like tanh correctly!
        policy_outputs = self.policy(obs,
                                     goals,
                                     num_steps_left,
                                     reparameterize=self.train_policy_with_reparameterization,
                                     return_log_prob=True)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
        if not self.dense_rewards and not self.dense_log_pi:
            log_pi = log_pi * terminals

        """
        QF Loss
        """
        target_v_values = self.target_vf(
            observations=next_obs,
            goals=goals,
            num_steps_left=num_steps_left-1,
        )
        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_v_values
        q_target = q_target.detach()
        bellman_errors_1 = (q1_pred - q_target) ** 2
        bellman_errors_2 = (q2_pred - q_target) ** 2
        qf1_loss = bellman_errors_1.mean()
        qf2_loss = bellman_errors_2.mean()

        if self.use_automatic_entropy_tuning:
            """
            Alpha Loss
            """
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha = 1

        """
        VF Loss
        """
        q1_new_actions = self.qf1(
            observations=obs,
            actions=new_actions,
            goals=goals,
            num_steps_left=num_steps_left,
        )
        q2_new_actions = self.qf2(
            observations=obs,
            actions=new_actions,
            goals=goals,
            num_steps_left=num_steps_left,
        )
        q_new_actions = torch.min(q1_new_actions, q2_new_actions)
        v_target = q_new_actions - alpha * log_pi
        v_pred = self.vf(
            observations=obs,
            goals=goals,
            num_steps_left=num_steps_left,
        )
        v_target = v_target.detach()
        bellman_errors = (v_pred - v_target) ** 2
        vf_loss = bellman_errors.mean()

        """
        Update networks
        """
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        """
        Policy Loss
        """
        # paper says to do + but apparently that's a typo. Do Q - V.
        if self.train_policy_with_reparameterization:
            policy_loss = (alpha * log_pi - q_new_actions).mean()
        else:
            log_policy_target = q_new_actions - v_pred
            policy_loss = (
                log_pi * (alpha * log_pi - log_policy_target).detach()
            ).mean()
        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value ** 2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        if self._n_train_steps_total % self.policy_update_period == 0:
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.vf, self.target_vf, self.soft_target_tau
            )

        """
        Save some statistics for eval
        """
        if self.need_to_update_eval_statistics:
            self.need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
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
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = ptu.get_numpy(alpha)[0]
                self.eval_statistics['Alpha Loss'] = ptu.get_numpy(alpha_loss)[0]
