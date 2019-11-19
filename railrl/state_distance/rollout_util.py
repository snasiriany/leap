import numpy as np

from railrl.samplers.rollout_functions import tdm_rollout

class MultigoalSimplePathSampler(object):
    def __init__(
            self,
            env,
            policy,
            max_samples,
            max_path_length,
            tau_sampling_function,
            qf=None,
            cycle_taus_for_rollout=True,
            render=False,
            observation_key=None,
            desired_goal_key=None,
    ):
        self.env = env
        self.policy = policy
        self.qf = qf
        self.max_samples = max_samples
        self.max_path_length = max_path_length
        self.tau_sampling_function = tau_sampling_function
        self.cycle_taus_for_rollout = cycle_taus_for_rollout
        self.render = render
        self.observation_key = observation_key
        self.desired_goal_key = desired_goal_key

    def obtain_samples(self):
        paths = []
        for i in range(self.max_samples // self.max_path_length):
            tau = self.tau_sampling_function()
            path = multitask_rollout(
                self.env,
                self.policy,
                self.qf,
                init_tau=tau,
                max_path_length=self.max_path_length,
                decrement_tau=self.cycle_taus_for_rollout,
                cycle_tau=self.cycle_taus_for_rollout,
                animated=self.render,
                observation_key=self.observation_key,
                desired_goal_key=self.desired_goal_key,
            )
            paths.append(path)
        return paths

ax1 = None
ax2 = None


def debug(env, obs, agent_info):
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        print("could not import matplotlib")
    global ax1
    global ax2
    if ax1 is None:
        _, (ax1, ax2) = plt.subplots(1, 2)

    subgoal_seq = agent_info['subgoal_seq']
    planned_action_seq = agent_info['planned_action_seq']
    real_obs_seq = env.true_states(
        obs, planned_action_seq
    )
    ax1.clear()
    env.plot_trajectory(
        ax1,
        np.array(subgoal_seq),
        np.array(planned_action_seq),
        goal=env._target_position,
    )
    ax1.set_title("imagined")
    ax2.clear()
    env.plot_trajectory(
        ax2,
        np.array(real_obs_seq),
        np.array(planned_action_seq),
        goal=env._target_position,
    )
    ax2.set_title("real")
    plt.draw()
    plt.pause(0.001)

def multitask_rollout(*args, **kwargs):
    # TODO Steven: remove pointer
    return tdm_rollout(*args, **kwargs)

def _expand_goal(goal, path_length):
    return np.repeat(
        np.expand_dims(goal, 0),
        path_length,
        0,
    )
