import numpy as np
import torch
import railrl.torch.pytorch_util as ptu

from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.classic_mujoco.ant_maze import AntMazeEnv

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def compute_sampled_latents(vae_env):
    vae_env.num_active_dims = 0
    for std in vae_env.vae.dist_std:
        if std > 0.15:
            vae_env.num_active_dims += 1

    vae_env.active_dims = vae_env.vae.dist_std.argsort()[-vae_env.num_active_dims:][::-1]
    vae_env.inactive_dims = vae_env.vae.dist_std.argsort()[:-vae_env.num_active_dims][::-1]

    if vae_env.use_vae_dataset and vae_env.vae_dataset_path is not None:
        from multiworld.core.image_env import normalize_image
        from railrl.misc.asset_loader import local_path_from_s3_or_local_path
        filename = local_path_from_s3_or_local_path(vae_env.vae_dataset_path)
        dataset = np.load(filename).item()
        vae_env.num_samples_for_latent_histogram = min(dataset['next_obs'].shape[0], vae_env.num_samples_for_latent_histogram)
        sampled_idx = np.random.choice(dataset['next_obs'].shape[0], vae_env.num_samples_for_latent_histogram)
        if vae_env.vae_input_key_prefix == 'state':
            vae_dataset_samples = dataset['next_obs'][sampled_idx]
        else:
            vae_dataset_samples = normalize_image(dataset['next_obs'][sampled_idx])
        del dataset
    else:
        vae_dataset_samples = None

    n = vae_env.num_samples_for_latent_histogram

    if vae_dataset_samples is not None:
        imgs = vae_dataset_samples
    else:
        if vae_env.vae_input_key_prefix == 'state':
            imgs = vae_env.wrapped_env.wrapped_env.sample_goals(n)['state_desired_goal']
        else:
            imgs = vae_env.wrapped_env.sample_goals(n)['image_desired_goal']

    batch_size = 2500
    latents, latents_noisy, latents_reproj = None, None, None
    for i in range(0, n, batch_size):
        batch_latents_mean, batch_latents_logvar = vae_env.encode_imgs(imgs[i:i + batch_size], clip_std=False)
        batch_latents_noisy = vae_env.reparameterize(batch_latents_mean, batch_latents_logvar, noisy=True)
        if vae_env.use_reprojection_network:
            batch_latents_reproj = ptu.get_numpy(vae_env.reproject_encoding(ptu.np_to_var(batch_latents_noisy)))
        if latents is None:
            latents = batch_latents_mean
            latents_noisy = batch_latents_noisy
            if vae_env.use_reprojection_network:
                latents_reproj = batch_latents_reproj
        else:
            latents = np.concatenate((latents, batch_latents_mean), axis=0)
            latents_noisy = np.concatenate((latents_noisy, batch_latents_noisy), axis=0)
            if vae_env.use_reprojection_network:
                latents_reproj = np.concatenate((latents_reproj, batch_latents_reproj), axis=0)

    vae_env.sampled_latents = latents
    vae_env.sampled_latents_noisy = latents_noisy
    vae_env.sampled_latents_reproj = latents_reproj

def compute_latent_histogram(vae_env):
    vae_env.compute_sampled_latents()
    num_dims = min(len(vae_env.vae.dist_std), 10)

    vae_env.latent_histogram = [[] for _ in range(num_dims)]
    vae_env.latent_histogram_noisy = [[] for _ in range(num_dims)]
    vae_env.latent_histogram_reproj = [[] for _ in range(num_dims)]

    if vae_env.num_samples_for_latent_histogram >= 50000:
        nx, ny = 65, 65
    elif vae_env.num_samples_for_latent_histogram >= 10000:
        nx, ny = 50, 50
    else:
        nx, ny = 35, 35

    sorted_dims = vae_env.vae.dist_std.argsort()[::-1]
    for i in range(num_dims):
        for j in range(num_dims):
            if i < j:
                histogram = np.zeros((nx, ny))
                dims = [sorted_dims[i], sorted_dims[j]]
                latents = vae_env.sampled_latents[:, dims]
                mu = vae_env.vae.dist_mu[dims]
                std = vae_env.vae.dist_std[dims]
                latents = (latents - mu) / std

                lower_bounds = np.array([-2.5, -2.5])
                upper_bounds = np.array([2.5, 2.5])
                diffs = (upper_bounds - lower_bounds) / (nx, ny)
                latent_indices = ((latents - lower_bounds) / diffs).astype(int)

                for idx in latent_indices:
                    if np.any(idx < 0) or np.any(idx > (nx - 1, ny - 1)):
                        continue
                    histogram[tuple(idx)] += 1
            else:
                histogram = np.array([[0.0]])
            vae_env.latent_histogram[i].append(histogram)

    for i in range(num_dims):
        for j in range(num_dims):
            if i < j:
                histogram = np.zeros((nx, ny))
                dims = [sorted_dims[i], sorted_dims[j]]
                latents = vae_env.sampled_latents_noisy[:, dims]
                mu = np.zeros(len(dims))
                std = np.ones(len(dims))
                latents = (latents - mu) / std

                lower_bounds = np.array([-2.5, -2.5])
                upper_bounds = np.array([2.5, 2.5])
                diffs = (upper_bounds - lower_bounds) / (nx, ny)
                latent_indices = ((latents - lower_bounds) / diffs).astype(int)

                for idx in latent_indices:
                    if np.any(idx < 0) or np.any(idx > (nx - 1, ny - 1)):
                        continue
                    histogram[tuple(idx)] += 1
            else:
                histogram = np.array([[0.0]])
            vae_env.latent_histogram_noisy[i].append(histogram)

    if vae_env.use_reprojection_network:
        for i in range(num_dims):
            for j in range(num_dims):
                if i < j:
                    histogram = np.zeros((nx, ny))
                    dims = [sorted_dims[i], sorted_dims[j]]
                    latents = vae_env.sampled_latents_reproj[:, dims]
                    mu = vae_env.vae.dist_mu[dims]
                    std = vae_env.vae.dist_std[dims]
                    latents = (latents - mu) / std

                    lower_bounds = np.array([-2.5, -2.5])
                    upper_bounds = np.array([2.5, 2.5])
                    diffs = (upper_bounds - lower_bounds) / (nx, ny)
                    latent_indices = ((latents - lower_bounds) / diffs).astype(int)

                    for idx in latent_indices:
                        if np.any(idx < 0) or np.any(idx > (nx - 1, ny - 1)):
                            continue
                        histogram[tuple(idx)] += 1
                else:
                    histogram = np.array([[0.0]])
                vae_env.latent_histogram_reproj[i].append(histogram)

def compute_goal_encodings(vae_env):
    nx, ny = (vae_env.vis_granularity, vae_env.vis_granularity)
    if getattr(vae_env, "get_states_sweep", None) is not None:
        states_sweep = vae_env.get_states_sweep(nx, ny)
        vae_env.sweep_goal_mu, vae_env.sweep_goal_logvar = vae_env.encode_states(states_sweep)
    else:
        vae_env.sweep_goal_mu, vae_env.sweep_goal_logvar = None, None

def get_image_v(vae_env, agent, qf, vf, obs, tau=None, noisy='none', imsize=None):
    wrapped_env = vae_env.wrapped_env
    if isinstance(wrapped_env, ImageEnv):
        wrapped_env = wrapped_env.wrapped_env

    if isinstance(wrapped_env, AntMazeEnv):
        return wrapped_env.get_image_v(agent, qf, vf, obs, tau=tau, imsize=vae_env.imsize)

    nx, ny = (vae_env.vis_granularity, vae_env.vis_granularity)
    if len(obs) == len(vae_env.wrapped_env.observation_space.spaces['state_observation'].low):
        obs_mu, obs_logvar = vae_env.encode_states(obs.reshape((1, -1)))
    else:
        obs_mu, obs_logvar = vae_env.encode_imgs(obs.reshape((1, -1)))
    sweep_obs_mu = np.tile(obs_mu.reshape((1, -1)), (nx * ny, 1))
    sweep_obs_logvar = np.tile(obs_logvar.reshape((1, -1)), (nx * ny, 1))
    if vae_env.sweep_goal_mu is None:
        vae_env.compute_goal_encodings()

    if noisy != 'none':
        num_samples = 10
    else:
        num_samples = 1

    sweep_obs, sweep_goal, sweep_tau = None, None, None
    for i in range(num_samples):
        sweep_obs_batch = vae_env.reparameterize(
            sweep_obs_mu,
            sweep_obs_logvar,
            (noisy == 'state') or (noisy == 'state_and_goal')
        )
        sweep_goal_batch = vae_env.reparameterize(vae_env.sweep_goal_mu,
                                               vae_env.sweep_goal_logvar,
                                               (noisy == 'goal') or (noisy == 'state_and_goal'))
        if tau is not None:
            sweep_tau_batch = np.tile(tau, (nx * ny, 1))

        if sweep_obs is None:
            sweep_obs = sweep_obs_batch
            sweep_goal = sweep_goal_batch
            if tau is not None:
                sweep_tau = sweep_tau_batch
        else:
            sweep_obs = np.concatenate((sweep_obs, sweep_obs_batch), axis=0)
            sweep_goal = np.concatenate((sweep_goal, sweep_goal_batch), axis=0)
            if tau is not None:
                sweep_tau = np.concatenate((sweep_tau, sweep_tau_batch), axis=0)

    if vf is not None:
        if tau is not None:
            v_vals = vf.eval_np(sweep_obs, sweep_goal, sweep_tau)
        else:
            sweep_obs_goal = np.hstack((sweep_obs, sweep_goal))
            v_vals = vf.eval_np(sweep_obs_goal)
    else:
        if tau is not None:
            sweep_actions = agent.eval_np(sweep_obs, sweep_goal, sweep_tau)
            v_vals = qf.eval_np(sweep_obs, sweep_actions, sweep_goal, sweep_tau)
        else:
            sweep_obs_goal = np.hstack((sweep_obs, sweep_goal))
            sweep_actions = agent.eval_np(sweep_obs_goal)
            v_vals = qf.eval_np(sweep_obs_goal, sweep_actions)

    v_vals = -np.linalg.norm(v_vals, ord=vae_env.norm_order, axis=1)
    v_vals = v_vals.reshape((num_samples, nx, ny))
    v_vals = np.mean(v_vals, axis=0)

    if vae_env.v_func_heatmap_bounds is not None:
        vmin = vae_env.v_func_heatmap_bounds[0]
        vmax = vae_env.v_func_heatmap_bounds[1]
    else:
        vmin, vmax = None, None

    if vae_env._use_vae_goals:
        draw_goal = False
    else:
        draw_goal = True

    return vae_env.get_image_plt(
        v_vals,
        imsize=vae_env.imsize,
        vmin=vmin,
        vmax=vmax,
        draw_goal=draw_goal
    )

def get_image_rew(vae_env, obs, reward_type='none'):
    nx, ny = (vae_env.vis_granularity, vae_env.vis_granularity)
    if len(obs) == len(vae_env.wrapped_env.observation_space.spaces['state_observation'].low):
        obs_mu, obs_logvar = vae_env.encode_states(obs.reshape((1, -1)))
    else:
        obs_mu, obs_logvar = vae_env.encode_imgs(obs.reshape((1, -1)))
    obs = obs_mu
    sweep_obs = np.tile(obs.reshape((1, -1)), (nx * ny, 1))
    if vae_env.sweep_goal_mu is None:
        vae_env.compute_goal_encodings()
    sweep_goal = vae_env.sweep_goal_mu

    if reward_type == 'none':
        if 'mahalanobis' in vae_env.reward_type:
            reward_type = 'mahalanobis'
        elif 'logp' in vae_env.reward_type:
            reward_type = 'logp'
        elif 'sparse' in vae_env.reward_type:
            reward_type = 'sparse'
        elif 'exponential' in vae_env.reward_type:
            reward_type = 'exponential'
        else:
            reward_type = 'euclidean'
    if reward_type == 'euclidean':
        rew_vals = -np.linalg.norm(sweep_obs - sweep_goal, ord=vae_env.norm_order, axis=-1)
    elif reward_type == 'sparse':
        dist = np.linalg.norm(sweep_obs - sweep_goal, ord=vae_env.norm_order, axis=1)
        rew_vals = (dist >= vae_env.epsilon) * -1.0
    elif reward_type == 'exponential':
        dist = np.linalg.norm(sweep_obs - sweep_goal, ord=vae_env.norm_order, axis=1)
        rew_vals = np.exp(-dist / vae_env.temperature) - 1.0
    elif reward_type == 'mahalanobis':
        std = np.exp(0.5 * obs_logvar)
        scaling = 1 / std
        rew_vals = -np.linalg.norm((sweep_obs - sweep_goal) * scaling, ord=vae_env.norm_order, axis=-1)
    elif reward_type == 'logp':
        std = np.exp(0.5 * obs_logvar)
        logstd = np.log(std)
        rew_vals = -logstd - (sweep_obs - sweep_goal) ** 2 / (2 * std ** 2)
        rew_vals = np.sum(rew_vals, axis=-1)
    elif reward_type == 'kl':
        sweep_obs_mu = np.tile(obs_mu.reshape((1, -1)), (nx * ny, 1))
        sweep_obs_logvar = np.tile(obs_logvar.reshape((1, -1)), (nx * ny, 1))

        sweep_goal_mu = vae_env.sweep_goal_mu
        sweep_goal_logvar = vae_env.sweep_goal_logvar

        mu1, logvar1 = sweep_obs_mu, sweep_obs_logvar
        mu2, logvar2 = sweep_goal_mu, sweep_goal_logvar

        kl_vals = 0.5 * np.sum(logvar2 - logvar1 - 1
                               + np.exp(logvar1) / np.exp(logvar2)
                               + (mu2 - mu1) ** 2 / np.exp(logvar2),
                               axis=-1)
        rew_vals = -kl_vals
    elif reward_type == 'kl_rev':
        sweep_obs_mu = np.tile(obs_mu.reshape((1, -1)), (nx * ny, 1))
        sweep_obs_logvar = np.tile(obs_logvar.reshape((1, -1)), (nx * ny, 1))

        sweep_goal_mu = vae_env.sweep_goal_mu
        sweep_goal_logvar = vae_env.sweep_goal_logvar

        mu2, logvar2 = sweep_obs_mu, sweep_obs_logvar
        mu1, logvar1 = sweep_goal_mu, sweep_goal_logvar

        kl_vals = 0.5 * np.sum(logvar2 - logvar1 - 1
                               + np.exp(logvar1) / np.exp(logvar2)
                               + (mu2 - mu1) ** 2 / np.exp(logvar2),
                               axis=-1)
        rew_vals = -kl_vals
    if reward_type == 'sparse':
        small_markers = True
    else:
        small_markers = False

    rew_vals = rew_vals.reshape((nx, ny))

    if vae_env._use_vae_goals:
        draw_goal = False
    else:
        draw_goal = True

    return vae_env.get_image_plt(
        rew_vals,
        vmin=None,
        vmax=None,
        imsize=vae_env.imsize,
        small_markers=small_markers,
        draw_goal=draw_goal,
    )

def get_image_latent_histogram_2d(vae_env, noisy=False):
    if vae_env.latent_histogram is None:
        vae_env.compute_latent_histogram()

    vmin = 0
    if vae_env.num_samples_for_latent_histogram >= 50000:
        vmax = 15
    elif vae_env.num_samples_for_latent_histogram >= 10000:
        vmax = 5
    else:
        vmax = 3

    if noisy:
        histogram = vae_env.latent_histogram_noisy[0][1]
        use_true_prior = True
    else:
        histogram = vae_env.latent_histogram[0][1]
        use_true_prior = False

    return vae_env.get_image_latent_plt(
        np.transpose(histogram),
        extent=[-2.5, 2.5, -2.5, 2.5],
        vmin=vmin, vmax=vmax,
        draw_goal=True,
        draw_state=True,
        draw_subgoals=True,
        use_true_prior=use_true_prior,
    )

def get_image_latent_histogram(vae_env, noisy=False, reproj=False, draw_dots=True, use_true_prior=None):
    if vae_env.latent_histogram is None:
        vae_env.compute_latent_histogram()

    num_dims = len(vae_env.latent_histogram)
    images = []
    images_active_dims = []

    if reproj:
        latent_histogram = vae_env.latent_histogram_reproj
        if len(latent_histogram[0]) == 0:
            return None
    elif noisy:
        latent_histogram = vae_env.latent_histogram_noisy
    else:
        latent_histogram = vae_env.latent_histogram

    if use_true_prior is None:
        use_true_prior = noisy

    vmin = 0
    if vae_env.num_samples_for_latent_histogram >= 50000:
        vmax = 15
    elif vae_env.num_samples_for_latent_histogram >= 10000:
        vmax = 5
    else:
        vmax = 3
    for i in range(num_dims):
        for j in range(num_dims):
            histogram = latent_histogram[i][j]
            image = vae_env.transform_image(vae_env.get_image_latent_plt(
                np.transpose(histogram),
                extent=[-2.5, 2.5, -2.5, 2.5],
                vmin=vmin, vmax=vmax,
                top_dim_axis=(i, j),
                draw_goal=(draw_dots and histogram.size > 1),
                draw_state=(draw_dots and histogram.size > 1),
                draw_subgoals=(draw_dots and histogram.size > 1),
                imsize=84,
                use_true_prior=use_true_prior,
            ))
            images.append(image)
            if i < vae_env.num_active_dims and j < vae_env.num_active_dims:
                images_active_dims.append(image)

    images = np.array(images)
    return images.reshape((num_dims ** 2, -1, 84, 84))

def get_image_v_latent(vae_env, agent, qf, vf, obs, tau=None):
    nx, ny = (vae_env.vis_granularity, vae_env.vis_granularity)

    if vae_env.sweep_goal_latents is None:
        top_latent_indices = vae_env.vae.dist_std.argsort()[-2:][::-1]
        bottom_latent_indices = vae_env.vae.dist_std.argsort()[:-2][::-1]
        mu = vae_env.vae.dist_mu[top_latent_indices]
        std = vae_env.vae.dist_std[top_latent_indices]

        lower_bounds = np.array([-2.5, -2.5]) * std + mu
        upper_bounds = np.array([2.5, 2.5]) * std + mu

        x = np.linspace(lower_bounds[0], upper_bounds[0], nx)
        y = np.linspace(lower_bounds[1], upper_bounds[1], ny)
        xv, yv = np.meshgrid(x, y)

        sweep_goal_top_latents = np.stack((xv, yv), axis=2).reshape((-1, 2))
        sweep_goal = np.zeros((nx * ny, vae_env.representation_size))
        sweep_goal[:, top_latent_indices] = sweep_goal_top_latents
        sweep_goal[:, bottom_latent_indices] = vae_env.vae.dist_mu[bottom_latent_indices]
        vae_env.sweep_goal_latents = sweep_goal

    sweep_obs = np.tile(obs.reshape((1, -1)), (nx * ny, 1))
    if tau is not None:
        sweep_tau = np.tile(tau, (nx * ny, 1))
    if vf is not None:
        if tau is not None:
            v_vals = vf.eval_np(sweep_obs, vae_env.sweep_goal_latents, sweep_tau)
        else:
            sweep_obs_goal = np.hstack((sweep_obs, vae_env.sweep_goal_latents))
            v_vals = vf.eval_np(sweep_obs_goal)
    else:
        if tau is not None:
            sweep_actions = agent.eval_np(sweep_obs, vae_env.sweep_goal_latents, sweep_tau)
            v_vals = qf.eval_np(sweep_obs, sweep_actions, vae_env.sweep_goal_latents, sweep_tau)
        else:
            sweep_obs_goal = np.hstack((sweep_obs, vae_env.sweep_goal_latents))
            sweep_actions = agent.eval_np(sweep_obs_goal)
            v_vals = qf.eval_np(sweep_obs_goal, sweep_actions)

    v_vals = -np.linalg.norm(v_vals, ord=vae_env.norm_order, axis=1)
    v_vals = v_vals.reshape((nx, ny))
    if vae_env.reward_type == 'latent_sparse':
        vmin = -1 * agent.reward_scale
        vmax = 0
    else:
        vmin = None
        vmax = None
    return vae_env.get_image_latent_plt(
        v_vals,
        draw_goal=True,
        draw_state=True,
        draw_subgoals=True,
        vmin=vmin, vmax=vmax,
        cmap='plasma'
    )

def get_image_latent_plt(
        vae_env,
        vals,
        vmin=None, vmax=None,
        extent=[-2.5, 2.5, -2.5, 2.5],
        cmap='Greys',
        top_dim_axis=None,
        draw_state=False, draw_goal=False, draw_subgoals=False,
        imsize=None,
        use_true_prior=False
):
    fig, ax = plt.subplots()
    ax.set_ylim(extent[2:4])
    ax.set_xlim(extent[0:2])
    ax.set_ylim(ax.get_ylim()[::-1])
    DPI = fig.get_dpi()
    if imsize is None:
        imsize = vae_env.imsize
    fig.set_size_inches(imsize / float(DPI), imsize / float(DPI))

    if top_dim_axis is None:
        top_latent_indices = vae_env.vae.dist_std.argsort()[-2:][::-1]
    else:
        sorted_dims = vae_env.vae.dist_std.argsort()[::-1]
        top_latent_indices = sorted_dims[np.array(top_dim_axis)]

    if use_true_prior:
        mu = np.zeros(len(top_latent_indices))
        std = np.ones(len(top_latent_indices))
    else:
        mu = vae_env.vae.dist_mu[top_latent_indices]
        std = vae_env.vae.dist_std[top_latent_indices]

    if draw_state:
        obs_pos = vae_env.latent_obs[top_latent_indices]
        obs_pos = (obs_pos - mu) / std

        ball = plt.Circle(obs_pos, 0.12, color='dodgerblue')
        ax.add_artist(ball)
    if draw_goal:
        goal_pos = vae_env.desired_goal['latent_desired_goal'][top_latent_indices]
        goal_pos = (goal_pos - mu) / std

        goal = plt.Circle(goal_pos, 0.12, color='lime')
        ax.add_artist(goal)
    if draw_subgoals and vae_env.latent_subgoals is not None:
        if use_true_prior:
            latent_subgoals = vae_env.latent_subgoals[:, top_latent_indices]
        else:
            latent_subgoals = vae_env.latent_subgoals_reproj[:, top_latent_indices]
        for subgoal in latent_subgoals:
            latent_pos = (subgoal - mu) / std
            sg = plt.Circle(latent_pos, 0.12, color='red')
            ax.add_artist(sg)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.subplots_adjust(bottom=0)
    fig.subplots_adjust(top=1)
    fig.subplots_adjust(right=1)
    fig.subplots_adjust(left=0)
    ax.axis('off')

    ax.imshow(
        vals,
        extent=extent,
        cmap=plt.get_cmap(cmap),
        interpolation='nearest',
        vmax=vmax,
        vmin=vmin,
        origin='bottom',
    )

    return plt_to_numpy(fig)

def plt_to_numpy(fig):
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data

def dump_latent_histogram(vae_env, epoch, noisy=False, reproj=False, use_true_prior=None, draw_dots=False):
    from railrl.core import logger
    import os.path as osp
    from torchvision.utils import save_image

    images = vae_env.get_image_latent_histogram(
        noisy=noisy, reproj=reproj, draw_dots=draw_dots, use_true_prior=use_true_prior
    )
    if noisy:
        prefix = 'h'
    elif reproj:
        prefix = 'h_r'
    else:
        prefix = 'h_mu'

    if epoch is None:
        save_dir = osp.join(logger.get_snapshot_dir(), prefix + '.png')
    else:
        save_dir = osp.join(logger.get_snapshot_dir(), prefix + '_%d.png' % epoch)
    save_image(
        ptu.FloatTensor(ptu.from_numpy(images)),
        save_dir,
        nrow=int(np.sqrt(images.shape[0])),
    )

def dump_samples(vae_env, epoch, n_samples=64):
    from railrl.core import logger
    from torchvision.utils import save_image
    import os.path as osp
    vae_env.vae.eval()
    sample = ptu.Variable(torch.randn(n_samples, vae_env.representation_size))
    sample = vae_env.vae.decode(sample).cpu()
    if vae_env.vae_input_key_prefix == 'state':
        sample = ptu.np_to_var(vae_env.wrapped_env.states_to_images(ptu.get_numpy(sample)))
        if sample is None:
            return
    if epoch is not None:
        save_dir = osp.join(logger.get_snapshot_dir(), 's_%d.png' % epoch)
    else:
        save_dir = osp.join(logger.get_snapshot_dir(), 's.png')
    save_image(
        sample.data.view(n_samples, -1, vae_env.wrapped_env.imsize, vae_env.wrapped_env.imsize),
        save_dir,
        nrow=int(np.sqrt(n_samples))
    )

def dump_reconstructions(vae_env, epoch, n_recon=16):
    from railrl.core import logger
    import os.path as osp
    from torchvision.utils import save_image

    if vae_env.use_vae_dataset and vae_env.vae_dataset_path is not None:
        from multiworld.core.image_env import normalize_image
        from railrl.misc.asset_loader import local_path_from_s3_or_local_path
        filename = local_path_from_s3_or_local_path(vae_env.vae_dataset_path)
        dataset = np.load(filename).item()
        sampled_idx = np.random.choice(dataset['next_obs'].shape[0], n_recon)
        if vae_env.vae_input_key_prefix == 'state':
            states = dataset['next_obs'][sampled_idx]
            imgs = ptu.np_to_var(
                vae_env.wrapped_env.states_to_images(states)
            )
            recon_samples, _, _ = vae_env.vae(ptu.np_to_var(states))
            recon_imgs = ptu.np_to_var(
                vae_env.wrapped_env.states_to_images(ptu.get_numpy(recon_samples))
            )
        else:
            imgs = ptu.np_to_var(
                normalize_image(dataset['next_obs'][sampled_idx])
            )
            recon_imgs, _, _, _ = vae_env.vae(imgs)
        del dataset
    else:
        return

    comparison = torch.cat([
        imgs.narrow(start=0, length=vae_env.wrapped_env.image_length, dimension=1).contiguous().view(
            -1,
            vae_env.wrapped_env.channels,
            vae_env.wrapped_env.imsize,
            vae_env.wrapped_env.imsize
        ),
        recon_imgs.contiguous().view(
            n_recon,
            vae_env.wrapped_env.channels,
            vae_env.wrapped_env.imsize,
            vae_env.wrapped_env.imsize
        )[:n_recon]
    ])

    if epoch is not None:
        save_dir = osp.join(logger.get_snapshot_dir(), 'r_%d.png' % epoch)
    else:
        save_dir = osp.join(logger.get_snapshot_dir(), 'r.png')
    save_image(comparison.data.cpu(), save_dir, nrow=n_recon)

def dump_latent_plots(vae_env, epoch):
    from railrl.core import logger
    import os.path as osp
    from torchvision.utils import save_image

    if getattr(vae_env, "get_states_sweep", None) is None:
        return

    nx, ny = (vae_env.vis_granularity, vae_env.vis_granularity)
    states_sweep = vae_env.get_states_sweep(nx, ny)
    sweep_latents_mu, sweep_latents_logvar = vae_env.encode_states(states_sweep, clip_std=False)

    sweep_latents_std = np.exp(0.5*sweep_latents_logvar)
    sweep_latents_sample = vae_env.reparameterize(sweep_latents_mu, sweep_latents_logvar, noisy=True)
    images_mu_sc, images_std_sc, images_sample_sc = [], [], []
    imsize = 84
    for i in range(sweep_latents_mu.shape[1]):
        image_mu_sc = vae_env.transform_image(vae_env.get_image_plt(
            sweep_latents_mu[:,i].reshape((nx, ny)),
            vmin=-2.0, vmax=2.0,
            draw_state=False, imsize=imsize))
        images_mu_sc.append(image_mu_sc)

        image_std_sc = vae_env.transform_image(vae_env.get_image_plt(
            sweep_latents_std[:,i].reshape((nx, ny)),
            vmin=0.0, vmax=2.0,
            draw_state=False, imsize=imsize))
        images_std_sc.append(image_std_sc)

        image_sample_sc = vae_env.transform_image(vae_env.get_image_plt(
            sweep_latents_sample[:,i].reshape((nx, ny)),
            vmin=-3.0, vmax=3.0,
            draw_state=False, imsize=imsize))
        images_sample_sc.append(image_sample_sc)

    images = images_mu_sc + images_std_sc + images_sample_sc
    images = np.array(images)

    if vae_env.representation_size > 16:
        nrow = 16
    else:
        nrow = vae_env.representation_size

    if epoch is not None:
        save_dir = osp.join(logger.get_snapshot_dir(), 'z_%d.png' % epoch)
    else:
        save_dir = osp.join(logger.get_snapshot_dir(), 'z.png')
    save_image(
        ptu.FloatTensor(
            ptu.from_numpy(
                images.reshape(
                    (vae_env.representation_size*3, -1, imsize, imsize)
                ))),
        save_dir,
        nrow=nrow,
    )