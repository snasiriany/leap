"""
Policies to be used with a state-distance Q/V function.
"""
import math
import numpy as np
from scipy import optimize
import torch
from torch import nn

from railrl.torch import pytorch_util as ptu

from collections import OrderedDict
from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict

class SubgoalPlanner(nn.Module):
    """
    Optimize subgols using LBFGS/LBFGS-B/BFGS/SGD/RMSPROP/CEM.

    Make it sublcass nn.Module so that calls to `train` and `cuda` get
    propagated to the sub-networks
    """
    def __init__(
            self,
            env,
            qf,
            mf_policy,
            observation_key,
            desired_goal_key,
            achieved_goal_key,
            max_tau,
            max_tau_per_subprob,
            reward_scale=1.0,
            vf=None,
            vae=None,
            infinite_horizon=False,
            replan_freq=-1,
            use_double=False,
            reproject_encoding=False,
            optimize_over_states=False,
            q_input_is_raw_state=False,
            init_method='random',
            use_true_prior_for_init=True,
            do_optimization=True,
            optimizer='lbfgs',
            cem_optimizer_kwargs={},
            gradient_optimizer_kwargs={},
            bfgs_optimizer_kwargs={},
            cost_mode='min',
            cost_kwargs={},
            realistic_subgoal_weight=0.0,
            use_realistic_hard_constraint=False,
            realistic_hard_constraint_threshold=1.75,
            **kwargs
    ):
        assert cost_mode in ['sum', 'min', 'softmin', 'exp']
        assert optimizer in ['lbfgs_b', 'lbfgs', 'bfgs', 'rmsprop', 'adam', 'sgd', 'cem', 'manual']
        assert init_method in ['warm_start', 'random', 'goal', 'state', 'state_and_goal', 'state_to_goal']
        if optimize_over_states:
            assert optimizer == 'cem'

        nn.Module.__init__(self)
        self.env = env
        self.qf = qf
        self.vf = vf
        self.mf_policy = mf_policy
        self.infinite_horizon = infinite_horizon
        self.max_tau = max_tau
        self.max_tau_per_subprob = max_tau_per_subprob
        self.replan_freq = replan_freq
        self.use_double = use_double
        self.reproject_encoding = reproject_encoding

        if optimizer == 'manual':
            init_method = 'warm_start'
            do_optimization = False

        self.init_method = init_method
        self.optimize_over_states = optimize_over_states # doesn't use a VAE, subgoals are states themselves
        self.do_optimization = do_optimization
        self.optimizer = optimizer
        self.cem_optimizer_kwargs = cem_optimizer_kwargs
        self.gradient_optimizer_kwargs = gradient_optimizer_kwargs
        self.bfgs_optimizer_kwargs = bfgs_optimizer_kwargs

        self.use_true_prior_for_init = use_true_prior_for_init # probably belongs in optimizer kwargs
        self.realistic_subgoal_weight = realistic_subgoal_weight
        self.use_realistic_hard_constraint = use_realistic_hard_constraint
        self.realistic_hard_constraint_threshold = realistic_hard_constraint_threshold
        self.cost_mode = cost_mode
        self.cost_kwargs = cost_kwargs

        self.observation_key = observation_key
        self.desired_goal_key = desired_goal_key
        self.achieved_goal_key = achieved_goal_key
        
        self.q_input_is_raw_state = q_input_is_raw_state
        self.raw_state_size = len(self.env.observation_space.spaces['state_observation'].low)

        if hasattr(self.env, 'vae'):
            self.vae = self.env.vae
        else:
            self.vae = vae
        
        if self.vae:
            bounds_mu, bounds_std = self._get_vae_prior(use_true_prior=self.use_true_prior_for_init)
            self.opt_state_size = self.vae.representation_size
            self.bounds_mu, self.bounds_std = bounds_mu, bounds_std
            self.lower_bounds = bounds_mu - 2.0 * bounds_std
            self.upper_bounds = bounds_mu + 2.0 * bounds_std

            true_prior_mu_np = np.zeros(self.opt_state_size)
            true_prior_std_np = np.ones(self.opt_state_size)

            true_prior_mu = ptu.np_to_var(true_prior_mu_np, double=use_double)
            true_prior_std = ptu.np_to_var(true_prior_std_np, double=use_double)
            self.true_prior_distr = torch.distributions.Normal(true_prior_mu, true_prior_std)
        else:
            self.opt_state_size = self.raw_state_size
            self.lower_bounds = self.env.observation_space.spaces[self.achieved_goal_key].low
            self.upper_bounds = self.env.observation_space.spaces[self.achieved_goal_key].high

        self.reward_scale = reward_scale

    def reset(self):
        self.need_to_update_stats = True
        self.taus = None
        self.subgoals = None
        self.subgoals_reproj = None

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        stat_names = [
            'opt_func_loss',
            'opt_grad_norm',
            'opt_v_value',
            'opt_v_value_min',
            'opt_v_value_sum',
            'opt_realistic_subgoal_rew',

            'init_opt_func_loss',
            'init_opt_grad_norm',
            'init_opt_v_value',
            'init_opt_v_value_min',
            'init_opt_v_value_sum',
            'init_opt_realistic_subgoal_rew',

            'opt_func_calls',
            'opt_num_iters',
            'opt_stopped',
        ]

        for stat_name in stat_names:
            stat_name = stat_name
            stat = get_stat_in_paths(paths, 'agent_infos', stat_name)
            statistics.update(create_stats_ordered_dict(
                '%s%s' % (prefix, stat_name),
                stat,
                always_show_all_stats=True,
                ))
        return statistics

    def eval_np(self, *args):
        return self.mf_policy.eval_np(*args)

    def get_actions(self, *args):
        return self.mf_policy.get_actions(*args)

    def get_action(self, ob, goal, tau_high_level):
        self._update_taus(tau_high_level)

        if len(self.taus) == 1 and self.taus[0] == -1: # planner actually has more time to get to goal, make new plan
            update_subgoals = True
            self.taus = None
            self._update_taus(tau_high_level)
        elif tau_high_level == self.max_tau: # planning hasn't started
            update_subgoals = True
        elif self.taus[0] == -1: # reached subgoal; time to replan
            update_subgoals = True
            self.taus = self.taus[1:]
            self.num_subprobs = len(self.taus)
            self.num_subgoals = self.num_subprobs - 1
        elif self.replan_freq > 0 and tau_high_level % self.replan_freq == 0:
            update_subgoals = True
        else:
            update_subgoals = False

        if update_subgoals:
            if self.replan_freq == -1:
                replan = True
            elif self.replan_freq > 0:
                if tau_high_level % self.replan_freq == 0:
                    replan = True
                elif tau_high_level == self.max_tau:
                    replan = True
                else:
                    replan = False
            else:
                replan = False

            self._update_subgoals(ob, goal, replan=replan)

            if hasattr(self.env, 'update_subgoals'):
                self.env.update_subgoals(
                    subgoals=self.subgoals,
                    subgoals_reproj=self.subgoals_reproj,
                    subgoal_v_vals=self.current_opt_v_values,
                )
        info = {
            'opt_func_loss': self.opt_func_loss,
            'opt_grad_norm': self.opt_grad_norm,
            'opt_v_value': self.opt_v_value,
            'opt_v_value_min': self.opt_v_value_min,
            'opt_v_value_sum': self.opt_v_value_sum,
            'opt_realistic_subgoal_rew': self.opt_realistic_subgoal_rew,

            'init_opt_func_loss': self.init_opt_func_loss,
            'init_opt_grad_norm': self.init_opt_grad_norm,
            'init_opt_v_value': self.init_opt_v_value,
            'init_opt_v_value_min': self.init_opt_v_value_min,
            'init_opt_v_value_sum': self.init_opt_v_value_sum,
            'init_opt_realistic_subgoal_rew': self.init_opt_realistic_subgoal_rew,

            'opt_func_calls': self.opt_func_calls,
            'opt_num_iters': self.opt_num_iters,
            'opt_stopped': self.opt_stopped,

            'tau': self.taus[0],
            'current_subgoal': self.subgoals_reproj[0] if (self.subgoals_reproj is not None) else None,
        }
        if self.subgoals_reproj is not None:
            goal_to_try = self.subgoals_reproj[0]
            if self.q_input_is_raw_state:
                goal_to_try = ptu.get_numpy(self.vae.decode(ptu.np_to_var(goal_to_try)))[0]
        else:
            goal_to_try = goal
        if self.infinite_horizon:
            ob_and_goal = np.hstack((ob, goal_to_try))
            ac, _ = self.mf_policy.get_action(ob_and_goal)
        else:
            ac, _ = self.mf_policy.get_action(
                ob,
                goal_to_try,
                self.taus[0][None]
            )
        return ac, info

    def _update_taus(self, tau_high_level):
        if self.taus is None:
            self.num_subprobs = int(math.ceil((tau_high_level + 1) / (self.max_tau_per_subprob+1)))
            self.num_subgoals = self.num_subprobs - 1

            taus_np = np.ones(self.num_subprobs) * ((tau_high_level + 1) // (self.num_subprobs))
            extra_time = int(np.sum(taus_np) - (tau_high_level + 1 - self.num_subprobs))
            if extra_time > 0:
                taus_np[-extra_time:] -= np.ones(extra_time)

            self.taus = taus_np
        else:
            self.taus[0] -= 1

    def _update_subgoals(
            self,
            ob_np,
            goal_np,
            replan=True
    ):
        if self.num_subgoals == 0:
            self.subgoals = None
            self.subgoals_reproj = None

            ob, goal, taus = self._np_to_pytorch(ob_np, goal_np, self.taus)
            loss, info = self._loss_np(self.subgoals, ob, goal, taus, info=True)
            self.current_opt_v_values = info['v_vals'][0]
            return

        lower_bounds = np.tile(self.lower_bounds, self.num_subgoals)
        upper_bounds = np.tile(self.upper_bounds, self.num_subgoals)
        bounds = list(zip(lower_bounds, upper_bounds))

        if self.subgoals is None:
            if self.init_method == 'warm_start':
                initial_x = self.env.generate_expert_subgoals(self.num_subprobs)
                if initial_x is None:
                    initial_x = np.tile(goal_np, self.num_subgoals)
            elif self.init_method == 'random':
                initial_x = np.random.uniform(lower_bounds, upper_bounds)
            elif self.init_method == 'goal':
                initial_x = np.tile(goal_np, self.num_subgoals)
            elif self.init_method == 'state':
                initial_x = np.tile(ob_np, self.num_subgoals)
            elif self.init_method == 'state_and_goal':
                initial_x = np.concatenate((np.tile(ob_np, self.num_subgoals - 1), goal_np))
            elif self.init_method == 'state_to_goal':
                scalings = np.linspace(0.0, 1.0, num=self.num_subgoals)
                initial_x = np.outer((1-scalings), ob_np) + np.outer(scalings, goal_np)
            else:
                initial_x = None

        else:
            initial_x = self.subgoals[-self.num_subgoals:]

        initial_x = initial_x.flatten()

        if self.use_double:
            if self.vf is not None:
                self.vf.double()
            else:
                self.qf.double()
                self.mf_policy.double()
            if self.reproject_encoding:
                self.vae.double()

        if self.do_optimization and replan:
            do_optimization = True
        else:
            do_optimization = False

        opt_info = self._optimize(
            init_subgoals_np=initial_x,
            ob_np=ob_np,
            goal_np=goal_np,
            taus_np=self.taus,
            bounds=bounds,
            do_optimization=do_optimization,
        )

        self.subgoals = opt_info['x'].reshape(-1, self.opt_state_size)
        self.subgoals_reproj = opt_info['x_reproj'].reshape(-1, self.opt_state_size)

        if self.need_to_update_stats:
            self.need_to_update_stats = False

            ob, goal, taus = self._np_to_pytorch(ob_np, goal_np, self.taus)

            init_loss, init_info = self._loss_np(initial_x, ob, goal, taus, info=True)
            self.init_opt_func_loss = init_loss

            init_grad = self._grad_np(initial_x, ob, goal, taus)
            self.init_opt_grad_norm = np.linalg.norm(init_grad) / np.sqrt(self.subgoals.size)

            self.init_opt_v_value = init_info['v_val']
            self.init_opt_v_value_min = init_info['v_val_min']
            self.init_opt_v_value_sum = init_info['v_val_sum']
            self.init_opt_realistic_subgoal_rew = init_info['realistic_subgoal_rew']

            loss, info = self._loss_np(self.subgoals, ob, goal, taus, info=True)
            self.opt_func_loss = loss
            self.opt_grad_norm = np.linalg.norm(opt_info['jac']) / np.sqrt(self.subgoals.size)
            self.current_opt_v_values = self.opt_v_values = info['v_vals'][0]
            self.opt_v_value = info['v_val']
            self.opt_v_value_min = info['v_val_min']
            self.opt_v_value_sum = info['v_val_sum']
            self.opt_realistic_subgoal_rew = info['realistic_subgoal_rew']

            self.opt_func_calls = opt_info['nfev']
            self.opt_num_iters = opt_info['nit']
            self.opt_stopped = (opt_info['status'] == 2)
        else:
            ob, goal, taus = self._np_to_pytorch(ob_np, goal_np, self.taus)
            loss, info = self._loss_np(self.subgoals, ob, goal, taus, info=True)
            self.current_opt_v_values = info['v_vals'][0]

        if self.use_double:
            if self.vf is not None:
                self.vf.float()
            else:
                self.qf.float()
                self.mf_policy.float()
            if self.reproject_encoding:
                self.vae.float()

    def _np_to_pytorch(
            self,
            ob_np,
            goal_np,
            taus_np,
            batch_size=1
    ):
        ob_np = np.tile(ob_np, (batch_size, 1, 1))
        goal_np = np.tile(goal_np, (batch_size, 1, 1))
        taus_np = np.tile(taus_np.reshape((1, self.num_subprobs, 1)), (batch_size, 1, 1))

        ob = ptu.np_to_var(ob_np, double=self.use_double)
        goal = ptu.np_to_var(goal_np, double=self.use_double)
        taus = ptu.np_to_var(taus_np, double=self.use_double)

        return ob, goal, taus

    def _get_vae_prior(self, use_true_prior):
        if use_true_prior:
            mu_np = np.zeros(self.vae.representation_size)
            std_np = np.ones(self.vae.representation_size)
        else:
            mu_np = self.vae.dist_mu
            std_np = self.vae.dist_std

        return mu_np, std_np
    
    def _optimize(
            self,
            init_subgoals_np,
            ob_np,
            goal_np,
            taus_np,
            bounds=None,
            do_optimization=True,
    ):
        if not do_optimization:
            opt_info = {}
            opt_info['x'] = init_subgoals_np
            opt_info['x_reproj'] = init_subgoals_np
            ob, goal, taus = self._np_to_pytorch(
                ob_np,
                goal_np,
                taus_np,
                batch_size=1
            )
            opt_info['fun'] = self._grad_np(init_subgoals_np, ob, goal, taus)
            opt_info['nfev'] = opt_info['nit'] = opt_info['status'] = 0
            opt_info['jac'] = np.zeros(init_subgoals_np.shape)
            return opt_info

        opt_info = {}
        if self.optimizer in ['lbfgs_b', 'lbfgs', 'bfgs']:
            ob, goal, taus = self._np_to_pytorch(ob_np, goal_np, taus_np)
            if self.optimizer == 'lbfgs_b':
                opt_info = optimize.minimize(
                    fun=self._loss_np,
                    jac=self._grad_np,
                    x0=init_subgoals_np,
                    args=(ob, goal, taus,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    **self.bfgs_optimizer_kwargs,
                )
            elif self.optimizer == 'lbfgs':
                opt_info = optimize.minimize(
                    fun=self._loss_np,
                    jac=self._grad_np,
                    x0=init_subgoals_np,
                    args=(ob, goal, taus,),
                    method='L-BFGS-B',
                    bounds=None,
                    **self.bfgs_optimizer_kwargs,
                )
            elif self.optimizer == 'bfgs':
                opt_info = optimize.minimize(
                    fun=self._loss_np,
                    jac=self._grad_np,
                    x0=init_subgoals_np,
                    args=(ob, goal, taus,),
                    method='BFGS',
                    **self.bfgs_optimizer_kwargs,
                )

            subgoals_reproj = ptu.np_to_var(opt_info['x'], double=self.use_double, requires_grad=False)
            if self.reproject_encoding:
                subgoals_reproj = self.env.reproject_encoding(subgoals_reproj)
            opt_info['x_reproj'] = ptu.get_numpy(subgoals_reproj).astype(np.float64)

        elif self.optimizer in ['rmsprop', 'sgd', 'adam']:
            opt_info = self._optimize_sgd(init_subgoals_np, ob_np, goal_np, taus_np)
        elif self.optimizer == 'cem':
            opt_info = self._optimize_cem(init_subgoals_np, ob_np, goal_np, taus_np)
        else:
            raise NotImplementedError

        return opt_info

    def _optimize_sgd(
            self,
            init_subgoals_np,
            ob_np,
            goal_np,
            taus_np
    ):
        optimizer_kwargs = self.gradient_optimizer_kwargs
        num_iters = optimizer_kwargs.get('num_iters', 3000)
        verbose = optimizer_kwargs.get('verbose', False)
        lr = optimizer_kwargs.get('lr', 1e-3)

        ob, goal, taus = self._np_to_pytorch(ob_np, goal_np, taus_np)
        subgoals = ptu.np_to_var(init_subgoals_np, double=self.use_double, requires_grad=True)
        if self.optimizer == 'sgd':
            optimizer = torch.optim.SGD([subgoals], lr=lr)
        elif self.optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop([subgoals], lr=lr)
        elif self.optimizer == 'adam':
            optimizer = torch.optim.Adam([subgoals], lr=lr)
        else:
            raise NotImplementedError
        for i in range(num_iters):
            loss = self._loss(
                subgoals,
                ob,
                goal,
                taus,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if verbose and (i % (num_iters//10) == 0 or i == num_iters-1):
                print(i, loss.data[0])
        if verbose:
            print()
        opt_info = {}
        opt_info['x'] = ptu.get_numpy(subgoals).astype(np.float64)
        subgoals_reproj = subgoals
        if self.reproject_encoding:
            subgoals_reproj = self.env.reproject_encoding(subgoals_reproj)
        opt_info['x_reproj'] = ptu.get_numpy(subgoals_reproj).astype(np.float64)
        opt_info['nfev'] = num_iters
        opt_info['nit'] = num_iters
        opt_info['status'] = 0
        if subgoals.grad is not None:
            opt_info['jac'] = ptu.get_numpy(subgoals.grad).astype(np.float64)
        else:
            opt_info['jac'] = np.zeros(init_subgoals_np.shape)

        return opt_info

    def _optimize_cem(
            self,
            init_subgoals_np,
            ob_np,
            goal_np,
            taus_np
    ):
        optimizer_kwargs = self.cem_optimizer_kwargs
        num_iters = optimizer_kwargs.get('num_iters', 50)
        batch_size = optimizer_kwargs.get('batch_size', 30000)
        # if the batch sizes are variable over iterations
        if hasattr(batch_size, "__len__"):
            batch_sizes = [batch_size[0]]*(num_iters//2) + [batch_size[1]]*(num_iters-(num_iters//2))
        else:
            batch_sizes = [batch_size]*num_iters

        frac_top_chosen = optimizer_kwargs.get('frac_top_chosen', 0.01)
        # if the top chosen is variable over iterations
        if hasattr(frac_top_chosen, "__len__"):
            frac_top_chosens = np.array(
                [frac_top_chosen[0]]*(num_iters//2) + [frac_top_chosen[1]]*(num_iters-(num_iters//2))
            )
        else:
            frac_top_chosens = np.ones(num_iters) * frac_top_chosen

        verbose = optimizer_kwargs.get('verbose', False)
        use_init_subgoals = optimizer_kwargs.get('use_init_subgoals', False)
        use_realistic_subgoals = optimizer_kwargs.get('use_realistic_subgoals', True)
        obs, goals, taus = self._np_to_pytorch(
            ob_np,
            goal_np,
            taus_np,
            batch_size=batch_sizes[0]
        )

        if self.optimize_over_states:
            state_space = self.env.observation_space.spaces['state_observation']
            mu_subgoal = (state_space.low + state_space.high) / 2
            std_subgoal = (mu_subgoal - state_space.low) * 0.625

            mu_np = np.tile(mu_subgoal, (self.num_subgoals, 1)).flatten()
            std_np = np.tile(std_subgoal, (self.num_subgoals, 1)).flatten()
        else:
            vae_mu, vae_std = self._get_vae_prior(use_true_prior=self.use_true_prior_for_init)
            mu_np = np.tile(vae_mu, (self.num_subgoals, 1)).flatten()
            std_np = np.tile(vae_std * 1.5, (self.num_subgoals, 1)).flatten()

        if use_init_subgoals and self.subgoals is not None:
            mu_np = init_subgoals_np

        mu = ptu.np_to_var(mu_np, double=self.use_double)
        std = ptu.np_to_var(std_np, double=self.use_double)

        for i in range(num_iters):
            # if batch sized changed from prev iteration
            if i > 0 and batch_sizes[i] != batch_sizes[i-1]:
                obs, goals, taus = self._np_to_pytorch(
                    ob_np,
                    goal_np,
                    taus_np,
                    batch_size=batch_sizes[i]
                )

            samples = torch.distributions.Normal(mu, std).sample_n(batch_sizes[i])

            if self.optimize_over_states:
                state_space = self.env.observation_space.spaces['state_observation']
                samples_np = ptu.get_numpy(samples)
                realistic_indices = []
                num_invalid_samples, num_unrealistic_samples = 0, 0
                for (idx, sample) in enumerate(samples_np):
                    sample = sample.reshape((-1, len(state_space.low)))
                    realistic_sample = True
                    for subgoal in sample:
                        if not state_space.contains(subgoal):
                            num_invalid_samples += 1
                            realistic_sample = False
                            break
                        if use_realistic_subgoals and not self.env.realistic_state_np(subgoal):
                            num_unrealistic_samples += 1
                            realistic_sample = False
                            break
                    if realistic_sample:
                        realistic_indices.append(idx)
                samples_np = samples_np[realistic_indices]
                num_realistic_samples = len(realistic_indices)
                if verbose:
                    print(num_realistic_samples, num_invalid_samples, num_unrealistic_samples, self.num_subgoals)
                samples = ptu.np_to_var(samples_np, double=self.use_double)

                if 'latent' in self.observation_key: # convert states into images and then latents
                    latent_samples_mu, latent_samples_logvar = self.env.encode_states(samples_np)
                    samples_for_eval = self.env.reparameterize(
                        latent_samples_mu,
                        latent_samples_logvar,
                        noisy=self.env.noisy_encoding
                    ).reshape(num_realistic_samples, -1)
                    samples_for_eval = ptu.np_to_var(samples_for_eval, double=self.use_double)
                else:
                    samples_for_eval = samples

                losses = self._loss(
                    samples_for_eval,
                    obs[:num_realistic_samples],
                    goals[:num_realistic_samples],
                    taus[:num_realistic_samples],
                )
            else:
                samples_for_eval = samples
                losses = self._loss(samples_for_eval, obs, goals, taus)

            sorted_losses, sorted_indices = torch.sort(losses)
            if verbose and (i % 3 == 0 or i == num_iters-1):
                print(i, sorted_losses[0].data[0], torch.mean(std).data[0], self.num_subgoals)
            num_top_chosen = int(frac_top_chosens[i] * batch_sizes[i])
            elite_indices = sorted_indices[:num_top_chosen]
            elites = samples[elite_indices]
            mu = torch.mean(elites, dim=0)
            std = torch.std(elites, dim=0)
        if verbose:
            print()

        if self.optimize_over_states:
            opt_sample = opt_sample_reproj = samples_for_eval[sorted_indices[0]][0]
        else:
            opt_sample = opt_sample_reproj = elites[0]
            if self.reproject_encoding:
                opt_sample_reproj = self.env.reproject_encoding(opt_sample)

        opt_info = {}
        opt_info['x'] = ptu.get_numpy(opt_sample).astype(np.float64)
        opt_info['x_reproj'] = ptu.get_numpy(opt_sample_reproj).astype(np.float64)
        opt_info['nfev'] = np.sum(batch_sizes)
        opt_info['nit'] = num_iters
        opt_info['status'] = 0
        if self.optimize_over_states:
            opt_info['jac'] = np.zeros(init_subgoals_np.shape)
        else:
            opt_info['jac'] = self._grad_np(opt_info['x'], obs[:1], goals[:1], taus[:1])

        return opt_info

    def _loss_np(self, subgoals_np, obs, goals, taus, info=False):
        if subgoals_np is not None:
            subgoals = ptu.np_to_var(subgoals_np, double=self.use_double, requires_grad=False)
        else:
            subgoals = None
        if info:
            loss, info = self._loss(subgoals, obs, goals, taus, info=True)
            loss_np = ptu.get_numpy(loss).astype(np.float64)
            return loss_np, info
        else:
            loss = self._loss(subgoals, obs, goals, taus, info=False)
            loss_np = ptu.get_numpy(loss).astype(np.float64)
            return loss_np

    def _grad_np(self, subgoals_np, obs, goals, taus):
        subgoals = ptu.np_to_var(subgoals_np, double=self.use_double, requires_grad=True)
        loss = self._loss(subgoals, obs, goals, taus, info=False)
        try:
            loss.backward()
            if subgoals.grad is not None:
                gradient_np = ptu.get_numpy(subgoals.grad).astype(np.float64)
            else:
                gradient_np = np.zeros(subgoals_np.shape)
        except:
            gradient_np = np.zeros(subgoals_np.shape)
        return gradient_np

    def _loss(self, subgoals, obs, goals, taus, info=False):
        if self.q_input_is_raw_state:
            state_size = self.raw_state_size
        else:
            state_size = self.opt_state_size
        batch_size = int(obs.numel() / state_size)

        if self.num_subgoals > 0:
            subgoals = subgoals.view(batch_size, self.num_subgoals, self.opt_state_size)
            original_subgoals = subgoals
            if self.reproject_encoding:
                subgoals = self.env.reproject_encoding(subgoals)

            if self.q_input_is_raw_state:
                subgoals = self.vae.decode(subgoals)
                subgoals = subgoals.view(batch_size, self.num_subgoals, state_size)

        if self.num_subgoals > 0:
            path = torch.cat((obs, subgoals, goals), dim=1)
        else:
            path = torch.cat((obs, goals), dim=1)
        s = path[:, :-1].contiguous().view(-1, state_size)
        g = path[:, 1:].contiguous().view(-1, state_size)
        taus = taus.view(-1, 1)

        if self.vf is not None:
            if self.infinite_horizon:
                states_and_goals = torch.cat((s, g), dim=1)
                v_val = self.vf(states_and_goals) / self.reward_scale
            else:
                v_vals = self.vf(s, g, taus) / self.reward_scale
        else:
            n = s.shape[0]
            v_vals = None
            bs = 100000
            for i in range(0, n, bs):
                if self.infinite_horizon:
                    states_and_goals = torch.cat((s[i:i+bs], g[i:i+bs]), dim=1)
                    a = self.mf_policy(states_and_goals).detach()
                    batch_v_vals = self.qf(states_and_goals, a).detach() / self.reward_scale
                else:
                    a = self.mf_policy(s[i:i + bs], g[i:i + bs], taus[i:i + bs]).detach()
                    batch_v_vals = self.qf(s[i:i + bs], a, g[i:i + bs], taus[i:i + bs]).detach() / self.reward_scale
                if v_vals is None:
                    v_vals = batch_v_vals
                else:
                    v_vals = torch.cat((v_vals, batch_v_vals), dim=0)

        if v_vals.size()[1] > 1:
            v_vals = -torch.norm(v_vals, p=self.qf.norm_order, dim=-1)
        v_vals = v_vals.view(batch_size, self.num_subprobs)

        min_v_val, _ = torch.min(v_vals, dim=-1)
        sum_v_val = torch.sum(v_vals, dim=-1)

        if self.cost_mode == 'sum':
            v_val = sum_v_val
        elif self.cost_mode == 'min':
            v_val = min_v_val
        elif self.cost_mode == 'softmin':
            softmin_temp = self.cost_kwargs.get("softmin_temp", 1e-4)
            softmin_v_val = -self._log_sum_exp(-v_vals / softmin_temp, dim=-1) * softmin_temp
            v_val = softmin_v_val
        elif self.cost_mode == 'exp':
            exp_temp = self.cost_kwargs.get("exp_temp", 1e0)
            exp_v_val = -torch.sum(
                torch.exp(
                    torch.clamp(-v_vals / exp_temp, min=-10.0, max=10.0)
                ) - 1,
                dim=-1
            )
            v_val = exp_v_val

        if not self.optimize_over_states:
            if self.num_subgoals > 0:
                realistic_subgoal_rew = self._realistic_subgoal_reward(original_subgoals, use_double=self.use_double)
                is_outside_threshold = torch.abs(original_subgoals) > self.realistic_hard_constraint_threshold
                if self.use_double:
                    is_outside_threshold = is_outside_threshold.double()
                else:
                    is_outside_threshold = is_outside_threshold.float()
                realistic_hard_constraint_subgoal_rew = - 1e6 * is_outside_threshold

                realistic_subgoal_rew = realistic_subgoal_rew.view(batch_size, self.num_subgoals)
                realistic_subgoal_rew = torch.sum(realistic_subgoal_rew, dim=-1)

                realistic_hard_constraint_subgoal_rew = realistic_hard_constraint_subgoal_rew.view(
                    batch_size, self.num_subgoals * self.opt_state_size)
                realistic_hard_constraint_subgoal_rew = torch.sum(realistic_hard_constraint_subgoal_rew, dim=-1)
            else:
                realistic_subgoal_rew = ptu.np_to_var(np.zeros(batch_size), double=self.use_double)
                realistic_hard_constraint_subgoal_rew = ptu.np_to_var(np.zeros(batch_size), double=self.use_double)

            if self.use_realistic_hard_constraint:
                loss = - (self.realistic_subgoal_weight * realistic_subgoal_rew
                          + realistic_hard_constraint_subgoal_rew
                          + v_val).squeeze(0)
            else:
                loss = - (self.realistic_subgoal_weight * realistic_subgoal_rew + v_val).squeeze(0)
        else:
            loss = - (v_val).squeeze(0)

        if info:
            v_vals_np = ptu.get_numpy(v_vals).astype(np.float64)
            v_val_np = ptu.get_numpy(v_val).astype(np.float64)
            min_v_val_np = ptu.get_numpy(min_v_val).astype(np.float64)
            sum_v_val_np = ptu.get_numpy(sum_v_val).astype(np.float64)
            if not self.optimize_over_states:
                realistic_subgoal_rew_np = ptu.get_numpy(realistic_subgoal_rew).astype(np.float64)
            else:
                realistic_subgoal_rew_np = 0.0
            return loss, {
                'v_vals': v_vals_np,
                'v_val': v_val_np,
                'v_val_min': min_v_val_np,
                'v_val_sum': sum_v_val_np,
                'realistic_subgoal_rew': realistic_subgoal_rew_np
            }
        else:
            return loss

    def _log_sum_exp(self, value, dim=None, keepdim=False):
        """Numerically stable implementation of the operation

        value.exp().sum(dim, keepdim).log()
        """
        # TODO: torch.max(value, dim=None) threw an error at time of writing
        if dim is not None:
            m, _ = torch.max(value, dim=dim, keepdim=True)
            value0 = value - m
            if keepdim is False:
                m = m.squeeze(dim)
            return m + torch.log(torch.sum(torch.exp(value0),
                                           dim=dim, keepdim=keepdim))
        else:
            m = torch.max(value)
            sum_exp = torch.sum(torch.exp(value - m))
            return m + torch.log(sum_exp)

    def _realistic_subgoal_reward(self, subgoals, use_double=True):
        if type(subgoals) is np.ndarray:
            subgoals = ptu.np_to_var(subgoals, double=use_double)

        if hasattr(self, "true_prior_distr"):
            log_prob = self.true_prior_distr.log_prob(subgoals)
            log_prob = torch.sum(log_prob, dim=-1)
            return log_prob
        else:
            return ptu.np_to_var(np.zeros(subgoals.shape[:-1]))

class InfiniteHorizonSubgoalPlanner(SubgoalPlanner):
    def __init__(self, *args, **kwargs):
        SubgoalPlanner.__init__(
            self,
            *args,
            infinite_horizon=True,
            **kwargs
        )

    def get_action(self, ob_and_goal):
        ob = ob_and_goal[:len(ob_and_goal) // 2]
        goal = ob_and_goal[len(ob_and_goal) // 2:]

        ac, info = super().get_action(ob, goal, self.tau_high_level)
        self.tau_high_level -= 1

        return ac, info

    def reset(self):
        self.tau_high_level = self.max_tau
        super().reset()



