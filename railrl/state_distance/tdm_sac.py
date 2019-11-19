import numpy as np

from railrl.misc.eval_util import create_stats_ordered_dict
from railrl.torch.sac.sac import SoftActorCritic
from railrl.state_distance.tdm import TemporalDifferenceModel
import railrl.torch.pytorch_util as ptu


class TdmSac(TemporalDifferenceModel, SoftActorCritic):
    def __init__(
            self,
            env,
            qf,
            vf,
            sac_kwargs,
            tdm_kwargs,
            base_kwargs,
            policy=None,
            replay_buffer=None,
            give_terminal_reward=False,
    ):
        SoftActorCritic.__init__(
            self,
            env=env,
            policy=policy,
            qf=qf,
            vf=vf,
            replay_buffer=replay_buffer,
            **sac_kwargs,
            **base_kwargs
        )
        TemporalDifferenceModel.__init__(self, **tdm_kwargs)
        action_space_diff = (
            self.env.action_space.high - self.env.action_space.low
        )

        # TODO(vitchyr): Maybe add this to the main SAC code.
        terminal_reward = 0
        for dim in range(action_space_diff.size):
            terminal_reward += (
                    -np.log(1./action_space_diff[dim])
            )
        self.terminal_bonus = float(terminal_reward)
        self.give_terminal_reward = give_terminal_reward

    def _do_training(self):
        batch = self.get_batch()
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        goals = batch['goals']
        num_steps_left = batch['num_steps_left']

        q_pred = self.qf(
            observations=obs,
            actions=actions,
            goals=goals,
            num_steps_left=num_steps_left,
        )
        v_pred = self.vf(
            observations=obs,
            goals=goals,
            num_steps_left=num_steps_left,
        )
        # Check policy accounts for squashing functions like tanh correctly!
        policy_outputs = self.policy(
            observations=obs,
            goals=goals,
            num_steps_left=num_steps_left,
            return_log_prob=True,
        )
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        """
        QF Loss
        """
        target_v_values = self.target_vf(
            next_obs,
            goals=goals,
            num_steps_left=num_steps_left-1,
        )
        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_v_values
        if self.give_terminal_reward:
            terminal_rewards = self.terminal_bonus * num_steps_left
            q_target = q_target + terminals * terminal_rewards
        qf_loss = self.qf_criterion(q_pred, q_target.detach())

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
        q_new_actions = self.qf(
            observations=obs,
            actions=new_actions,
            goals=goals,
            num_steps_left=num_steps_left,
        )
        v_target = q_new_actions - alpha * log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())

        """
        Policy Loss
        """
        if self.train_policy_with_reparameterization:
            policy_loss = (alpha * log_pi - q_new_actions).mean()
        else:
            log_policy_target = q_new_actions - v_pred
            policy_loss = (
                log_pi * (alpha * log_pi - log_policy_target).detach()
            ).mean()
        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value**2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        """
        Update networks
        """
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self._update_target_network()

        if self.need_to_update_eval_statistics:
            self.need_to_update_eval_statistics = False
            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q_pred),
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

    def evaluate(self, epoch):
        SoftActorCritic.evaluate(self, epoch)