import os.path as osp
import time

import numpy as np

def train_vae(variant):
    from railrl.misc.ml_util import PiecewiseLinearSchedule
    from railrl.torch.vae.conv_vae import ConvVAE
    from railrl.torch.vae.conv_vae_trainer import ConvVAETrainer
    from railrl.core import logger
    import railrl.torch.pytorch_util as ptu
    from multiworld.core.image_env import ImageEnv
    from railrl.envs.vae_wrappers import VAEWrappedEnv
    from railrl.misc.asset_loader import local_path_from_s3_or_local_path

    logger.remove_tabular_output(
        'progress.csv', relative_to_snapshot_dir=True
    )
    logger.add_tabular_output(
        'vae_progress.csv', relative_to_snapshot_dir=True
    )

    env_id = variant['generate_vae_dataset_kwargs'].get('env_id', None)
    if env_id is not None:
        import gym
        env = gym.make(env_id)
    else:
        env_class = variant['generate_vae_dataset_kwargs']['env_class']
        env_kwargs = variant['generate_vae_dataset_kwargs']['env_kwargs']
        env = env_class(**env_kwargs)

    representation_size = variant["representation_size"]
    beta = variant["beta"]
    if 'beta_schedule_kwargs' in variant:
        beta_schedule = PiecewiseLinearSchedule(
            **variant['beta_schedule_kwargs'])
    else:
        beta_schedule = None

    # obtrain training and testing data
    dataset_path = variant['generate_vae_dataset_kwargs'].get('dataset_path', None)
    test_p = variant['generate_vae_dataset_kwargs'].get('test_p', 0.9)
    filename = local_path_from_s3_or_local_path(dataset_path)
    dataset = np.load(filename).item()
    N = dataset['obs'].shape[0]
    n = int(N * test_p)
    train_data = {}
    test_data = {}
    for k in dataset.keys():
        train_data[k] = dataset[k][:n, :]
        test_data[k] = dataset[k][n:, :]

    # setup vae
    variant['vae_kwargs']['action_dim'] = train_data['actions'].shape[1]
    if variant.get('vae_type', None) == "VAE-state":
        from railrl.torch.vae.vae import VAE
        input_size = train_data['obs'].shape[1]
        variant['vae_kwargs']['input_size'] = input_size
        m = VAE(representation_size, **variant['vae_kwargs'])
    elif variant.get('vae_type', None) == "VAE2":
        from railrl.torch.vae.conv_vae2 import ConvVAE2
        variant['vae_kwargs']['imsize'] = variant['imsize']
        m = ConvVAE2(representation_size, **variant['vae_kwargs'])
    else:
        variant['vae_kwargs']['imsize'] = variant['imsize']
        m = ConvVAE(representation_size, **variant['vae_kwargs'])
    if ptu.gpu_enabled():
        m.cuda()

    # setup vae trainer
    if variant.get('vae_type', None) == "VAE-state":
        from railrl.torch.vae.vae_trainer import VAETrainer
        t = VAETrainer(train_data, test_data, m, beta=beta, beta_schedule=beta_schedule, **variant['algo_kwargs'])
    else:
        t = ConvVAETrainer(train_data, test_data, m, beta=beta, beta_schedule=beta_schedule, **variant['algo_kwargs'])

    # visualization
    vis_variant = variant.get('vis_kwargs', {})
    save_video = vis_variant.get('save_video', False)
    if isinstance(env, ImageEnv):
        image_env = env
    else:
        image_env = ImageEnv(
            env,
            variant['generate_vae_dataset_kwargs'].get('imsize'),
            init_camera=variant['generate_vae_dataset_kwargs'].get('init_camera'),
            transpose=True,
            normalize=True,
        )
    render = variant.get('render', False)
    reward_params = variant.get("reward_params", dict())
    vae_env = VAEWrappedEnv(
        image_env,
        m,
        imsize=image_env.imsize,
        decode_goals=render,
        render_goals=render,
        render_rollouts=render,
        reward_params=reward_params,
        **variant.get('vae_wrapped_env_kwargs', {})
    )
    vae_env.reset()
    vae_env.add_mode("video_env", 'video_env')
    vae_env.add_mode("video_vae", 'video_vae')
    if save_video:
        import railrl.samplers.rollout_functions as rf
        from railrl.policies.simple import RandomPolicy
        random_policy = RandomPolicy(vae_env.action_space)
        rollout_function = rf.create_rollout_function(
            rf.multitask_rollout,
            max_path_length=100,
            observation_key='latent_observation',
            desired_goal_key='latent_desired_goal',
            vis_list=vis_variant.get('vis_list', []),
            dont_terminate=True,
        )

        dump_video_kwargs = variant.get("dump_video_kwargs", dict())
        dump_video_kwargs['imsize'] = vae_env.imsize
        dump_video_kwargs['vis_list'] = [
            'image_observation',
            'reconstr_image_observation',
            'image_latent_histogram_2d',
            'image_latent_histogram_mu_2d',
            'image_plt',
            'image_rew',
            'image_rew_euclidean',
            'image_rew_mahalanobis',
            'image_rew_logp',
            'image_rew_kl',
            'image_rew_kl_rev',
        ]
    def visualization_post_processing(save_vis, save_video, epoch):
        vis_list = vis_variant.get('vis_list', [])

        if save_vis:
            if vae_env.vae_input_key_prefix == 'state':
                vae_env.dump_reconstructions(epoch, n_recon=vis_variant.get('n_recon', 16))
            vae_env.dump_samples(epoch, n_samples=vis_variant.get('n_samples', 64))
            if 'latent_representation' in vis_list:
                vae_env.dump_latent_plots(epoch)
            if any(elem in vis_list for elem in [
                'latent_histogram', 'latent_histogram_mu',
                'latent_histogram_2d', 'latent_histogram_mu_2d']):
                vae_env.compute_latent_histogram()
            if not save_video and ('latent_histogram' in vis_list):
                vae_env.dump_latent_histogram(epoch=epoch, noisy=True, use_true_prior=True)
            if not save_video and ('latent_histogram_mu' in vis_list):
                vae_env.dump_latent_histogram(epoch=epoch, noisy=False, use_true_prior=True)

        if save_video and save_vis:
            from railrl.envs.vae_wrappers import temporary_mode
            from railrl.misc.video_gen import dump_video
            from railrl.core import logger

            vae_env.compute_goal_encodings()

            logdir = logger.get_snapshot_dir()
            filename = osp.join(logdir, 'video_{epoch}.mp4'.format(epoch=epoch))
            variant['dump_video_kwargs']['epoch'] = epoch
            temporary_mode(
                vae_env,
                mode='video_env',
                func=dump_video,
                args=(vae_env, random_policy, filename, rollout_function),
                kwargs=variant['dump_video_kwargs']
            )
            if not vis_variant.get('save_video_env_only', True):
                filename = osp.join(logdir, 'video_{epoch}_vae.mp4'.format(epoch=epoch))
                temporary_mode(
                    vae_env,
                    mode='video_vae',
                    func=dump_video,
                    args=(vae_env, random_policy, filename, rollout_function),
                    kwargs=variant['dump_video_kwargs']
                )

    # train vae
    for epoch in range(variant['num_epochs']):
        save_vis = (epoch % vis_variant['save_period'] == 0 or epoch == variant['num_epochs'] - 1)
        save_vae = (epoch % variant['snapshot_gap'] == 0 or epoch == variant['num_epochs'] - 1)
        
        t.train_epoch(epoch)
        t.test_epoch(
            epoch,
            save_reconstruction=save_vis,
            save_interpolation=save_vis,
            save_vae=save_vae,
        )

        visualization_post_processing(save_vis, save_video, epoch)

    logger.save_extra_data(m, 'vae.pkl', mode='pickle')
    logger.remove_tabular_output(
        'vae_progress.csv',
        relative_to_snapshot_dir=True,
    )
    logger.add_tabular_output(
        'progress.csv',
        relative_to_snapshot_dir=True,
    )

    return m

def generate_vae_dataset(variant):
    import cv2

    env_class = variant.get('env_class', None)
    env_kwargs = variant.get('env_kwargs',None)
    env_id = variant.get('env_id', None)
    N = variant.get('N', 10000)

    use_images = variant.get('use_images', True)

    imsize = variant.get('imsize', 84)
    show = variant.get('show', False)
    init_camera = variant.get('init_camera', None)
    oracle_dataset = variant.get('oracle_dataset', False)
    if 'n_random_steps' in variant:
        n_random_steps = variant['n_random_steps']
    else:
        if oracle_dataset:
            n_random_steps = 3
        else:
            n_random_steps = 100
    vae_dataset_specific_env_kwargs = variant.get('vae_dataset_specific_env_kwargs', None)
    non_presampled_goal_img_is_garbage = variant.get('non_presampled_goal_img_is_garbage', None)
    from multiworld.core.image_env import ImageEnv, unormalize_image
    info = {}

    from railrl.core import logger
    logdir = logger.get_snapshot_dir()
    filename = osp.join(logdir, "vae_dataset.npy")

    now = time.time()

    if env_id is not None:
        import gym
        env = gym.make(env_id)
    else:
        if vae_dataset_specific_env_kwargs is None:
            vae_dataset_specific_env_kwargs = {}
        for key, val in env_kwargs.items():
            if key not in vae_dataset_specific_env_kwargs:
                vae_dataset_specific_env_kwargs[key] = val
        env = env_class(**vae_dataset_specific_env_kwargs)
    if not isinstance(env, ImageEnv):
        env = ImageEnv(
            env,
            imsize,
            init_camera=init_camera,
            transpose=True,
            normalize=True,
            non_presampled_goal_img_is_garbage=non_presampled_goal_img_is_garbage,
        )
    else:
        imsize = env.imsize
        env.non_presampled_goal_img_is_garbage = non_presampled_goal_img_is_garbage
    env.reset()
    info['env'] = env

    if use_images:
        data_size = len(env.observation_space.spaces['image_observation'].low)
        dtype = np.uint8
    else:
        data_size = len(env.observation_space.spaces['state_observation'].low)
        dtype = np.float32

    state_size = len(env.observation_space.spaces['state_observation'].low)

    dataset = {
        'obs': np.zeros((N, data_size), dtype=dtype),
        'actions': np.zeros((N, len(env.action_space.low)), dtype=np.float32),
        'next_obs': np.zeros((N, data_size), dtype=dtype),

        'obs_state': np.zeros((N, state_size), dtype=np.float32),
        'next_obs_state': np.zeros((N, state_size), dtype=np.float32),
    }

    for i in range(N):
        if i % (N/50) == 0:
            print(i)
        if oracle_dataset:
            if i % 100 == 0:
                env.reset()
            goal = env.sample_goal()
            env.set_to_goal(goal)
            for _ in range(n_random_steps):
                env.step(env.action_space.sample())
        else:
            env.reset()
            for _ in range(n_random_steps):
                env.step(env.action_space.sample())

        obs = env._get_obs()
        if use_images:
            dataset['obs'][i, :] = unormalize_image(obs['image_observation'])
        else:
            dataset['obs'][i, :] = obs['state_observation']
        dataset['obs_state'][i, :] = obs['state_observation']

        action = env.action_space.sample()
        dataset['actions'][i, :] = action

        obs = env.step(action)[0]
        img = obs['image_observation']
        if use_images:
            dataset['next_obs'][i, :] = unormalize_image(img)
        else:
            dataset['next_obs'][i, :] = obs['state_observation']
        dataset['next_obs_state'][i, :] = obs['state_observation']
        if show:
            img = img.reshape(3, imsize, imsize).transpose((1, 2, 0))
            img = img[::, :, ::-1]
            cv2.imshow('img', img)
            cv2.waitKey(1000)

    print("keys and shapes:")
    for k in dataset.keys():
        print(k, dataset[k].shape)
    print("done making training data", filename, time.time() - now)
    np.save(filename, dataset)

def train_reprojection_network_and_update_variant(variant):
    from railrl.core import logger
    from railrl.misc.asset_loader import load_local_or_remote_file
    import railrl.torch.pytorch_util as ptu

    rl_variant = variant.get("rl_variant", {})
    vae_wrapped_env_kwargs = rl_variant.get('vae_wrapped_env_kwargs', {})
    if vae_wrapped_env_kwargs.get("use_reprojection_network", False):
        train_reprojection_network_variant = variant.get("train_reprojection_network_variant", {})

        if train_reprojection_network_variant.get("use_cached_network", False):
            vae_path = rl_variant.get("vae_path", None)
            reprojection_network = load_local_or_remote_file(osp.join(vae_path, 'reproj_network.pkl'))
            from railrl.core import logger
            logger.save_extra_data(reprojection_network, 'reproj_network.pkl', mode='pickle')

            if ptu.gpu_enabled():
                reprojection_network.cuda()

            vae_wrapped_env_kwargs['reprojection_network'] = reprojection_network
        else:
            logger.remove_tabular_output(
                'progress.csv', relative_to_snapshot_dir=True
            )
            logger.add_tabular_output(
                'reproj_progress.csv', relative_to_snapshot_dir=True
            )

            vae_path = rl_variant.get("vae_path", None)
            ckpt = rl_variant.get("ckpt", None)

            if type(ckpt) is str:
                vae = load_local_or_remote_file(osp.join(ckpt, 'vae.pkl'))
                from railrl.core import logger

                logger.save_extra_data(vae, 'vae.pkl', mode='pickle')
            elif type(vae_path) is str:
                vae = load_local_or_remote_file(osp.join(vae_path, 'vae_params.pkl'))
                from railrl.core import logger

                logger.save_extra_data(vae, 'vae.pkl', mode='pickle')
            else:
                vae = vae_path

            if type(vae) is str:
                vae = load_local_or_remote_file(vae)
            else:
                vae = vae

            if ptu.gpu_enabled():
                vae.cuda()

            train_reprojection_network_variant['vae'] = vae
            reprojection_network = train_reprojection_network(train_reprojection_network_variant)
            vae_wrapped_env_kwargs['reprojection_network'] = reprojection_network

def train_reprojection_network(variant):
    from railrl.torch.vae.reprojection_network import (
        ReprojectionNetwork,
        ReprojectionNetworkTrainer,
    )
    from railrl.core import logger
    import railrl.torch.pytorch_util as ptu

    logger.get_snapshot_dir()

    vae = variant['vae']

    generate_reprojection_network_dataset_kwargs = variant.get("generate_reprojection_network_dataset_kwargs", {})
    generate_reprojection_network_dataset_kwargs['vae'] = vae
    train_data, test_data = generate_reprojection_network_dataset(generate_reprojection_network_dataset_kwargs)

    reprojection_network_kwargs = variant.get("reprojection_network_kwargs", {})
    m = ReprojectionNetwork(vae, **reprojection_network_kwargs)
    if ptu.gpu_enabled():
        m.cuda()

    algo_kwargs = variant.get("algo_kwargs", {})
    t = ReprojectionNetworkTrainer(train_data, test_data, m, **algo_kwargs)

    num_epochs = variant.get('num_epochs', 5000)
    for epoch in range(num_epochs):
        should_save_network = (epoch % 250 == 0 or epoch == num_epochs - 1)
        t.train_epoch(epoch)
        t.test_epoch(
            epoch,
            save_network=should_save_network,
        )
    logger.save_extra_data(m, 'reproj_network.pkl', mode='pickle')
    return m

def generate_reprojection_network_dataset(variant):
    import railrl.torch.pytorch_util as ptu

    vae = variant.get("vae")
    N = variant.get("N", 10000000)
    test_p = variant.get("test_p", 0.9)

    dataset = {
        'z': np.zeros((N, vae.representation_size), dtype=np.float32),
        'z_proj': np.zeros((N, vae.representation_size), dtype=np.float32),
    }

    mu, sigma = vae.dist_mu, vae.dist_std
    n = np.random.randn(N, vae.representation_size)
    z = sigma * n + mu

    batch_size = 1000
    z_proj = None
    t = time.time()
    print("generating reprojection network dataset...")
    for i in range(0, N, batch_size):
        batch_imgs = vae.decode(ptu.np_to_var(z[i:i + batch_size]))
        batch_encoding =  ptu.get_numpy(vae.encode(batch_imgs)[0])
        if z_proj is None:
            z_proj = batch_encoding
        else:
            z_proj = np.concatenate((z_proj, batch_encoding), axis=0)

    print("time spent (s):", time.time() - t)

    dataset['z'] = z.astype(np.float32)
    dataset['z_proj'] = z_proj.astype(np.float32)

    n = int(N * test_p)
    train_dataset = {}
    test_dataset = {}
    for k in dataset.keys():
        train_dataset[k] = dataset[k][:n, :]
        test_dataset[k] = dataset[k][n:, :]
    return train_dataset, test_dataset