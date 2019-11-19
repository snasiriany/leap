import numpy as np
from railrl.state_distance.policies import UniversalPolicy

def tau_sampling_tdm_rollout(*args, tau_sampler=None, **kwargs):
    init_tau = tau_sampler()
    return tdm_rollout(*args, init_tau=init_tau, **kwargs)

def create_rollout_function(rollout_function, **initial_kwargs):
    """
    initial_kwargs for
        rollout_function=tdm_rollout_function may contain:
            init_tau,
            decrement_tau,
            cycle_tau,
            get_action_kwargs,
            observation_key,
            desired_goal_key,
        rollout_function=multitask_rollout may contain:
            observation_key,
            desired_goal_key,
    """
    def wrapped_rollout_func(*args, **dynamic_kwargs):
        combined_args = {
            **initial_kwargs,
            **dynamic_kwargs
        }
        return rollout_function(*args, **combined_args)
    return wrapped_rollout_func

def multitask_rollout(
    env,
    agent,
    max_path_length=np.inf,
    animated=False,
    observation_key=None,
    desired_goal_key=None,
    vis_list=list(),
    qf=None,
    vf=None,
    dont_terminate=False,
    epoch=None,
    rollout_num=None,
    **kwargs
):
    full_observations = []
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    next_observations = []
    path_length = 0
    agent.reset()
    o = env.reset()
    if animated:
        env.render()
    goal = env.get_goal()
    agent_goal = goal
    if desired_goal_key:
        agent_goal = agent_goal[desired_goal_key]
    while path_length < max_path_length:
        full_observations.append(o)
        agent_o = o
        if observation_key:
            agent_o = agent_o[observation_key]
        new_obs = np.hstack((agent_o, agent_goal))
        a, agent_info = agent.get_action(new_obs)
        next_o, r, d, env_info = env.step(a)
        if animated:
            env.render()

        if 'latent' in observation_key:
            key = 'image_observation'
        else:
            key = observation_key

        update_next_obs(
            next_o, o, key, env, agent, qf, vf, vis_list, epoch, rollout_num, path_length,
        )

        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        next_observations.append(next_o)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d and not dont_terminate:
            break
        o = next_o
    full_observations.append(o)
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        goals=np.repeat(agent_goal[None], path_length, 0),
        full_observations=full_observations,
    )

def rollout(env, agent, max_path_length=np.inf, animated=False):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos

    :param env:
    :param agent:
    :param max_path_length:
    :param animated:
    :return:
    """
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    next_o = None
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )

def tdm_rollout(
    env,
    agent: UniversalPolicy,
    qf=None,
    vf=None,
    max_path_length=np.inf,
    animated=False,
    init_tau=0.0,
    decrement_tau=False,
    cycle_tau=False,
    get_action_kwargs=None,
    observation_key=None,
    desired_goal_key=None,
    vis_list=list(),
    dont_terminate=False,
    epoch=None,
    rollout_num=None,
):
    full_observations = []
    from railrl.state_distance.rollout_util import _expand_goal
    if get_action_kwargs is None:
        get_action_kwargs = {}
    observations = []
    next_observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    taus = []
    agent.reset()
    path_length = 0
    if animated:
        env.render()

    tau = np.array([init_tau])
    o = env.reset()

    goal = env.get_goal()
    agent_goal = goal
    if desired_goal_key:
        agent_goal = agent_goal[desired_goal_key]

    while path_length < max_path_length:
        full_observations.append(o)
        agent_o = o
        if observation_key:
            agent_o = agent_o[observation_key]

        a, agent_info = agent.get_action(agent_o, agent_goal, tau, **get_action_kwargs)
        if animated:
            env.render()
        if hasattr(env, 'set_tau'):
            if 'tau' in agent_info:
                env.set_tau(agent_info['tau'])
            else:
                env.set_tau(tau)
        next_o, r, d, env_info = env.step(a)

        if 'latent' in observation_key:
            key = 'image_observation'
        else:
            key = observation_key

        update_next_obs(
            next_o, o, key, env, agent, qf, vf, vis_list, epoch, rollout_num, path_length,
            agent_info.get('tau', tau),
        )

        next_observations.append(next_o)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        taus.append(tau.copy())
        path_length += 1
        if decrement_tau:
            tau -= 1
        if tau < 0:
            if cycle_tau:
                if init_tau > max_path_length - path_length - 1:
                    tau = np.array([max_path_length - path_length - 1])
                else:
                    tau = np.array([init_tau])
            else:
                tau = np.array([0])
        if d and not dont_terminate:
            break
        o = next_o
    full_observations.append(o)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)

    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=np.array(agent_infos),
        env_infos=np.array(env_infos),
        num_steps_left=np.array(taus),
        goals=_expand_goal(agent_goal, len(terminals)),
        full_observations=full_observations,
    )

def update_next_obs(next_o, o, key, env, agent, qf, vf, vis_list, epoch, rollout_num, path_length, *args):
    if 'plt' in vis_list:
        next_o['image_plt'] = env.transform_image(
            env.get_image_plt(vals=False, imsize=env.imsize, draw_state=True, draw_goal=True, draw_subgoals=True)
        )
    if 'v' in vis_list:
        next_o['image_v'] = env.transform_image(
            env.get_image_v(agent, qf, vf, o[key], *args, imsize=env.imsize)
        )
    if 'v_noisy_state_and_goal' in vis_list:
        next_o['image_v_noisy_state_and_goal'] = env.transform_image(
            env.get_image_v(agent, qf, vf, o[key], *args, noisy='state_and_goal')
        )
    if 'v_noisy_state' in vis_list:
        next_o['image_v_noisy_state'] = env.transform_image(
            env.get_image_v(agent, qf, vf, o[key], *args, noisy='state')
        )
    if 'v_noisy_goal' in vis_list:
        next_o['image_v_noisy_goal'] = env.transform_image(
            env.get_image_v(agent, qf, vf, o[key], *args, noisy='goal')
        )
    if 'rew' in vis_list:
        next_o['image_rew'] = env.transform_image(
            env.get_image_rew(o[key])
        )
    if 'rew_mahalanobis' in vis_list:
        next_o['image_rew_mahalanobis'] = env.transform_image(
            env.get_image_rew(o[key], reward_type='mahalanobis')
        )
    if 'rew_logp' in vis_list:
        next_o['image_rew_logp'] = env.transform_image(
            env.get_image_rew(o[key], reward_type='logp')
        )
    if 'rew_kl' in vis_list:
        next_o['image_rew_kl'] = env.transform_image(
            env.get_image_rew(o[key], reward_type='kl')
        )
    if 'rew_kl_rev' in vis_list:
        next_o['image_rew_kl_rev'] = env.transform_image(
            env.get_image_rew(o[key], reward_type='kl_rev')
        )
    if 'v_latent' in vis_list:
        next_o['image_v_latent'] = env.transform_image(
            env.get_image_v_latent(agent, qf, vf, o['latent_observation'], *args)
        )
    if 'latent_histogram_2d' in vis_list:
        next_o['image_latent_histogram_2d'] = env.transform_image(
            env.get_image_latent_histogram_2d(noisy=True)
        )
    if 'latent_histogram_mu_2d' in vis_list:
        next_o['image_latent_histogram_mu_2d'] = env.transform_image(
            env.get_image_latent_histogram_2d()
        )
    if 'latent_histogram' in vis_list and rollout_num == 0 and path_length == 0:
        env.dump_latent_histogram(epoch, noisy=True, draw_dots=True)
    if 'latent_histogram_mu' in vis_list and rollout_num == 0 and path_length == 0:
        env.dump_latent_histogram(epoch, noisy=False, draw_dots=True)
    if 'latent_histogram_reproj' in vis_list and rollout_num == 0 and path_length == 0:
        env.dump_latent_histogram(epoch, reproj=True, draw_dots=True)

