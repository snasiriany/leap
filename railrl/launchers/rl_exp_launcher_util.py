import os.path as osp

### TD3 ###

def ih_td3_experiment(variant):
    import railrl.samplers.rollout_functions as rf
    import railrl.torch.pytorch_util as ptu
    from railrl.data_management.obs_dict_replay_buffer import \
        ObsDictRelabelingBuffer
    from railrl.exploration_strategies.base import (
        PolicyWrappedWithExplorationStrategy
    )
    from railrl.misc.asset_loader import local_path_from_s3_or_local_path
    import joblib
    from railrl.torch.her.her_td3 import HerTd3
    from railrl.torch.networks import FlattenMlp, TanhMlpPolicy
    from railrl.state_distance.subgoal_planner import InfiniteHorizonSubgoalPlanner

    preprocess_rl_variant(variant)
    env = get_envs(variant)
    es = get_exploration_strategy(variant, env)

    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")

    vectorized = 'vectorized' in env.reward_type
    variant['replay_buffer_kwargs']['vectorized'] = vectorized

    if 'ckpt' in variant:
        if 'ckpt_epoch' in variant:
            epoch = variant['ckpt_epoch']
            filename = local_path_from_s3_or_local_path(osp.join(variant['ckpt'], 'itr_%d.pkl' % epoch))
        else:
            filename = local_path_from_s3_or_local_path(osp.join(variant['ckpt'], 'params.pkl'))
        print("Loading ckpt from", filename)
        data = joblib.load(filename)
        qf1 = data['qf1']
        qf2 = data['qf2']
        policy = data['policy']
    else:
        obs_dim = (
                env.observation_space.spaces[observation_key].low.size
                + env.observation_space.spaces[desired_goal_key].low.size
        )
        action_dim = env.action_space.low.size

        env.reset()
        _, rew, _, _ = env.step(env.action_space.sample())
        if hasattr(rew, "__len__"):
            output_size = len(rew)
        else:
            output_size = 1

        qf1 = FlattenMlp(
            input_size=obs_dim + action_dim,
            output_size=output_size,
            **variant['qf_kwargs']
        )
        qf2 = FlattenMlp(
            input_size=obs_dim + action_dim,
            output_size=output_size,
            **variant['qf_kwargs']
        )
        policy = TanhMlpPolicy(
            input_size=obs_dim,
            output_size=action_dim,
            **variant['policy_kwargs']
        )
        policy.reward_scale = variant['algo_kwargs']['base_kwargs'].get('reward_scale', 1.0)

    eval_policy = None
    if variant.get('eval_policy', None) == 'SubgoalPlanner':
        eval_policy = InfiniteHorizonSubgoalPlanner(
            env,
            qf1,
            policy,
            observation_key=observation_key,
            desired_goal_key=desired_goal_key,
            achieved_goal_key=achieved_goal_key,
            state_based=variant.get("do_state_exp", False),
            max_tau=variant['algo_kwargs']['base_kwargs']['max_path_length'] - 1,
            reward_scale=variant['algo_kwargs']['base_kwargs'].get('reward_scale', 1.0),
            **variant['SubgoalPlanner_kwargs']
        )

    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )

    replay_buffer = ObsDictRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )

    algo_kwargs = variant['algo_kwargs']
    algo_kwargs['replay_buffer'] = replay_buffer
    base_kwargs = algo_kwargs['base_kwargs']
    base_kwargs['training_env'] = env
    base_kwargs['render'] = variant.get("render", False)
    base_kwargs['render_during_eval'] = variant.get("render_during_eval", False)
    her_kwargs = algo_kwargs['her_kwargs']
    her_kwargs['observation_key'] = observation_key
    her_kwargs['desired_goal_key'] = desired_goal_key
    algorithm = HerTd3(
        env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        exploration_policy=exploration_policy,
        eval_policy=eval_policy,
        **variant['algo_kwargs']
    )

    if variant.get("test_ckpt", False):
        algorithm.post_epoch_funcs.append(get_update_networks_func(variant))

    vis_variant = variant.get('vis_kwargs', {})
    vis_list = vis_variant.get('vis_list', [])
    if vis_variant.get("save_video", True):
        rollout_function = rf.create_rollout_function(
            rf.multitask_rollout,
            max_path_length=algorithm.max_path_length,
            observation_key=algorithm.observation_key,
            desired_goal_key=algorithm.desired_goal_key,
            vis_list=vis_list,
            dont_terminate=True,
        )
        video_func = get_video_save_func(
            rollout_function,
            env,
            variant,
        )
        algorithm.post_epoch_funcs.append(video_func)

    if ptu.gpu_enabled():
        print("using GPU")
        algorithm.cuda()
        if not variant.get("do_state_exp", False):
            env.vae.cuda()

    env.reset()
    if not variant.get("do_state_exp", False):
        env.dump_samples(epoch=None)
        env.dump_latent_plots(epoch=None)
        env.dump_latent_plots(epoch=None)

    algorithm.train()

def tdm_td3_experiment(variant):
    import railrl.samplers.rollout_functions as rf
    import railrl.torch.pytorch_util as ptu
    from railrl.data_management.obs_dict_replay_buffer import \
        ObsDictRelabelingBuffer
    from railrl.exploration_strategies.base import (
        PolicyWrappedWithExplorationStrategy
    )
    from railrl.state_distance.tdm_networks import TdmQf, TdmPolicy
    from railrl.state_distance.tdm_td3 import TdmTd3
    from railrl.state_distance.subgoal_planner import SubgoalPlanner
    from railrl.misc.asset_loader import local_path_from_s3_or_local_path
    import joblib

    preprocess_rl_variant(variant)
    env = get_envs(variant)
    es = get_exploration_strategy(variant, env)

    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")

    vectorized = 'vectorized' in env.reward_type
    variant['algo_kwargs']['tdm_kwargs']['vectorized'] = vectorized
    variant['replay_buffer_kwargs']['vectorized'] = vectorized

    if 'ckpt' in variant:
        if 'ckpt_epoch' in variant:
            epoch = variant['ckpt_epoch']
            filename = local_path_from_s3_or_local_path(osp.join(variant['ckpt'], 'itr_%d.pkl' % epoch))
        else:
            filename = local_path_from_s3_or_local_path(osp.join(variant['ckpt'], 'params.pkl'))
        print("Loading ckpt from", filename)
        data = joblib.load(filename)
        qf1 = data['qf1']
        qf2 = data['qf2']
        policy = data['policy']
        variant['algo_kwargs']['base_kwargs']['reward_scale'] = policy.reward_scale
    else:
        obs_dim = (
            env.observation_space.spaces[observation_key].low.size
        )
        goal_dim = (
            env.observation_space.spaces[desired_goal_key].low.size
        )
        action_dim = env.action_space.low.size

        variant['qf_kwargs']['vectorized'] = vectorized
        norm_order = env.norm_order
        variant['qf_kwargs']['norm_order'] = norm_order
        env.reset()
        _, rew, _, _ = env.step(env.action_space.sample())
        if hasattr(rew, "__len__"):
            variant['qf_kwargs']['output_dim'] = len(rew)
        qf1 = TdmQf(
            env=env,
            observation_dim=obs_dim,
            goal_dim=goal_dim,
            action_dim=action_dim,
            **variant['qf_kwargs']
        )
        qf2 = TdmQf(
            env=env,
            observation_dim=obs_dim,
            goal_dim=goal_dim,
            action_dim=action_dim,
            **variant['qf_kwargs']
        )
        policy = TdmPolicy(
            env=env,
            observation_dim=obs_dim,
            goal_dim=goal_dim,
            action_dim=action_dim,
            reward_scale=variant['algo_kwargs']['base_kwargs'].get('reward_scale', 1.0),
            **variant['policy_kwargs']
        )

    eval_policy = None
    if variant.get('eval_policy', None) == 'SubgoalPlanner':
        eval_policy = SubgoalPlanner(
            env,
            qf1,
            policy,
            observation_key=observation_key,
            desired_goal_key=desired_goal_key,
            achieved_goal_key=achieved_goal_key,
            state_based=variant.get("do_state_exp", False),
            max_tau=variant['algo_kwargs']['tdm_kwargs']['max_tau'],
            reward_scale=variant['algo_kwargs']['base_kwargs'].get('reward_scale', 1.0),
            **variant['SubgoalPlanner_kwargs']
        )

    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )

    replay_buffer = ObsDictRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )

    algo_kwargs = variant['algo_kwargs']
    algo_kwargs['replay_buffer'] = replay_buffer
    base_kwargs = algo_kwargs['base_kwargs']
    base_kwargs['training_env'] = env
    base_kwargs['render'] = variant.get("render", False)
    base_kwargs['render_during_eval'] = variant.get("render_during_eval", False)
    tdm_kwargs = algo_kwargs['tdm_kwargs']
    tdm_kwargs['observation_key'] = observation_key
    tdm_kwargs['desired_goal_key'] = desired_goal_key
    algorithm = TdmTd3(
        env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        exploration_policy=exploration_policy,
        eval_policy=eval_policy,
        **variant['algo_kwargs']
    )

    if variant.get("test_ckpt", False):
        algorithm.post_epoch_funcs.append(get_update_networks_func(variant))

    vis_variant = variant.get('vis_kwargs', {})
    vis_list = vis_variant.get('vis_list', [])
    if vis_variant.get("save_video", True):
        rollout_function = rf.create_rollout_function(
            rf.tdm_rollout,
            init_tau=algorithm._sample_max_tau_for_rollout(),
            decrement_tau=algorithm.cycle_taus_for_rollout,
            cycle_tau=algorithm.cycle_taus_for_rollout,
            max_path_length=algorithm.max_path_length,
            observation_key=algorithm.observation_key,
            desired_goal_key=algorithm.desired_goal_key,
            vis_list=vis_list,
            dont_terminate=True,
        )
        video_func = get_video_save_func(
            rollout_function,
            env,
            variant,
        )
        algorithm.post_epoch_funcs.append(video_func)

    if ptu.gpu_enabled():
        print("using GPU")
        algorithm.cuda()
        if not variant.get("do_state_exp", False):
            env.vae.cuda()

    env.reset()
    if not variant.get("do_state_exp", False):
        env.dump_samples(epoch=None)
        env.dump_reconstructions(epoch=None)
        env.dump_latent_plots(epoch=None)

    algorithm.train()

### SAC ###

def ih_twin_sac_experiment(variant):
    assert NotImplementedError

def tdm_twin_sac_experiment(variant):
    assert NotImplementedError

def get_envs(variant):
    from multiworld.core.image_env import ImageEnv
    from railrl.envs.vae_wrappers import VAEWrappedEnv
    from railrl.misc.asset_loader import load_local_or_remote_file

    render = variant.get('render', False)
    vae_path = variant.get("vae_path", None)
    reproj_vae_path = variant.get("reproj_vae_path", None)
    ckpt = variant.get("ckpt", None)
    reward_params = variant.get("reward_params", dict())
    init_camera = variant.get("init_camera", None)
    do_state_exp = variant.get("do_state_exp", False)

    presample_goals = variant.get('presample_goals', False)
    presample_image_goals_only = variant.get('presample_image_goals_only', False)
    presampled_goals_path = variant.get('presampled_goals_path', None)

    if not do_state_exp and type(ckpt) is str:
        vae = load_local_or_remote_file(osp.join(ckpt, 'vae.pkl'))
        if vae is not None:
            from railrl.core import logger
            logger.save_extra_data(vae, 'vae.pkl', mode='pickle')
    else:
        vae = None

    if vae is None and type(vae_path) is str:
        vae = load_local_or_remote_file(osp.join(vae_path, 'vae_params.pkl'))
        from railrl.core import logger

        logger.save_extra_data(vae, 'vae.pkl', mode='pickle')
    elif vae is None:
        vae = vae_path

    if type(vae) is str:
        vae = load_local_or_remote_file(vae)
    else:
        vae = vae

    if type(reproj_vae_path) is str:
        reproj_vae = load_local_or_remote_file(osp.join(reproj_vae_path, 'vae_params.pkl'))
    else:
        reproj_vae = None

    if 'env_id' in variant:
        import gym
        # trigger registration
        env = gym.make(variant['env_id'])
    else:
        env = variant["env_class"](**variant['env_kwargs'])
    if not do_state_exp:
        if isinstance(env, ImageEnv):
            image_env = env
        else:
            image_env = ImageEnv(
                env,
                variant.get('imsize'),
                init_camera=init_camera,
                transpose=True,
                normalize=True,
            )
        vae_env = VAEWrappedEnv(
            image_env,
            vae,
            imsize=image_env.imsize,
            decode_goals=render,
            render_goals=render,
            render_rollouts=render,
            reward_params=reward_params,
            reproj_vae=reproj_vae,
            **variant.get('vae_wrapped_env_kwargs', {})
        )
        if presample_goals:
            """
            This will fail for online-parallel as presampled_goals will not be
            serialized. Also don't use this for online-vae.
            """
            if presampled_goals_path is None:
                image_env.non_presampled_goal_img_is_garbage = True
                presampled_goals = variant['generate_goal_dataset_fctn'](
                    image_env=image_env,
                    **variant['goal_generation_kwargs']
                )
            else:
                presampled_goals = load_local_or_remote_file(
                    presampled_goals_path
                ).item()
                presampled_goals = {
                    'state_desired_goal': presampled_goals['next_obs_state'],
                    'image_desired_goal': presampled_goals['next_obs'],
                }

            image_env.set_presampled_goals(presampled_goals)
            vae_env.set_presampled_goals(presampled_goals)
            print("Presampling all goals")
        else:
            if presample_image_goals_only:
                presampled_goals = variant['generate_goal_dataset_fctn'](
                    image_env=vae_env.wrapped_env,
                    **variant['goal_generation_kwargs']
                )
                image_env.set_presampled_goals(presampled_goals)
                print("Presampling image goals only")
            else:
                print("Not using presampled goals")

        env = vae_env

    if not do_state_exp:
        training_mode = variant.get("training_mode", "train")
        testing_mode = variant.get("testing_mode", "test")
        env.add_mode('eval', testing_mode)
        env.add_mode('train', training_mode)
        env.add_mode('relabeling', training_mode)
        # relabeling_env.disable_render()
        env.add_mode("video_vae", 'video_vae')
        env.add_mode("video_env", 'video_env')
    return env

def get_exploration_strategy(variant, env):
    from railrl.exploration_strategies.epsilon_greedy import EpsilonGreedy
    from railrl.exploration_strategies.gaussian_strategy import GaussianStrategy
    from railrl.exploration_strategies.ou_strategy import OUStrategy
    exploration_type = variant.get('exploration_type', 'epsilon')
    exploration_noise = variant.get('exploration_noise', 0.1)
    if exploration_type == 'ou':
        es = OUStrategy(
            action_space=env.action_space,
            max_sigma=exploration_noise,
            min_sigma=exploration_noise,  # Constant sigma
        )
    elif exploration_type == 'gaussian':
        es = GaussianStrategy(
            action_space=env.action_space,
            max_sigma=exploration_noise,
            min_sigma=exploration_noise,  # Constant sigma
        )
    elif exploration_type == 'epsilon':
        es = EpsilonGreedy(
            action_space=env.action_space,
            prob_random_action=exploration_noise,
        )
    else:
        raise Exception("Invalid type: " + exploration_type)
    return es

def preprocess_rl_variant(variant):
    if variant.get("do_state_exp", False):
        if 'observation_key' not in variant:
            variant['observation_key'] = 'state_observation'
        if 'desired_goal_key' not in variant:
            variant['desired_goal_key'] = 'state_desired_goal'
        if 'achieved_goal_key' not in variant:
            variant['achieved_goal_key'] = 'state_acheived_goal'
    else:
        if 'observation_key' not in variant:
            variant['observation_key'] = 'latent_observation'
        if 'desired_goal_key' not in variant:
            variant['desired_goal_key'] = 'latent_desired_goal'
        if 'achieved_goal_key' not in variant:
            variant['achieved_goal_key'] = 'latent_acheived_goal'

def get_video_save_func(
        rollout_function,
        env,
        variant,
):
    from multiworld.core.image_env import ImageEnv
    from railrl.core import logger
    from railrl.envs.vae_wrappers import temporary_mode
    from railrl.misc.video_gen import dump_video
    logdir = logger.get_snapshot_dir()

    vis_variant = variant.get('vis_kwargs', {})
    save_period = vis_variant.get('save_period', 50)
    do_state_exp = variant.get("do_state_exp", False)
    dump_video_kwargs = variant.get("dump_video_kwargs", dict())

    vis_variant = variant.get('vis_kwargs', {})
    vis_blacklist = vis_variant.get('vis_blacklist', [])
    dump_video_kwargs['vis_blacklist'] = vis_blacklist

    if do_state_exp:
        imsize = variant.get('imsize')
        dump_video_kwargs['imsize'] = imsize
        image_env = ImageEnv(
            env,
            imsize,
            init_camera=variant.get('init_camera', None),
            transpose=True,
            normalize=True,
        )

        if 'pick' in env.__module__:
            from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place import get_image_presampled_goals
            num_goals_presampled = vis_variant.get('num_goals_presampled', 100)
            image_goals = get_image_presampled_goals(image_env, num_goals_presampled)
            image_env.set_presampled_goals(image_goals)

        def save_video(algo, epoch):
            dump_video_kwargs["epoch"] = epoch
            if hasattr(algo, "qf1"):
                dump_video_kwargs['qf'] = algo.qf1
            if hasattr(algo, "vf"):
                dump_video_kwargs['vf'] = algo.vf

            if epoch % save_period == 0 or epoch == algo.num_epochs - 1:
                filename = osp.join(logdir, 'video_{epoch}.mp4'.format(epoch=epoch))
                dump_video(image_env, algo.eval_policy, filename, rollout_function,
                           **dump_video_kwargs)

                if vis_variant.get('save_video_exp_policy', False):
                    filename = osp.join(logdir, 'video_{epoch}_exp.mp4'.format(epoch=epoch))
                    dump_video(image_env, algo.exploration_policy, filename, rollout_function,
                               **dump_video_kwargs)
    else:
        image_env = env
        dump_video_kwargs['imsize'] = env.imsize

        def save_video(algo, epoch):
            dump_video_kwargs["epoch"] = epoch
            if hasattr(algo, "qf1"):
                dump_video_kwargs['qf'] = algo.qf1
            if hasattr(algo, "vf"):
                dump_video_kwargs['vf'] = algo.vf

            if epoch % save_period == 0 or epoch == algo.num_epochs - 1:
                filename = osp.join(logdir, 'video_{epoch}.mp4'.format(epoch=epoch))
                temporary_mode(
                    image_env,
                    mode='video_env',
                    func=dump_video,
                    args=(image_env, algo.eval_policy, filename, rollout_function),
                    kwargs=dump_video_kwargs
                )

                if vis_variant.get('save_video_exp_policy', False):
                    filename = osp.join(logdir, 'video_{epoch}_exp.mp4'.format(epoch=epoch))
                    temporary_mode(
                        image_env,
                        mode='video_env',
                        func=dump_video,
                        args=(image_env, algo.exploration_policy, filename, rollout_function),
                        kwargs=dump_video_kwargs
                    )


                if not vis_variant.get('save_video_env_only', True):
                    filename = osp.join(logdir, 'video_{epoch}_vae.mp4'.format(epoch=epoch))
                    temporary_mode(
                        image_env,
                        mode='video_vae',
                        func=dump_video,
                        args=(image_env, algo.eval_policy, filename, rollout_function),
                        kwargs=dump_video_kwargs
                    )
    return save_video

def get_update_networks_func(variant):
    import railrl.torch.pytorch_util as ptu
    from railrl.misc.asset_loader import local_path_from_s3_or_local_path
    import joblib
    from railrl.state_distance.subgoal_planner import SubgoalPlanner

    def update_networks_func(algo, epoch):
        if epoch % algo.epoch_freq != 0 and epoch != algo.num_epochs - 1:
            exit()
        if epoch == algo.num_epochs - 1:
            filename = local_path_from_s3_or_local_path(osp.join(variant['ckpt'], 'params.pkl'))
        else:
            filename = local_path_from_s3_or_local_path(osp.join(variant['ckpt'], 'itr_%d.pkl' % epoch))
        print("updating networks from {}".format(filename))
        data = joblib.load(filename)
        assert (data["epoch"] == epoch)
        algo.qf1 = data['qf1']
        algo.qf2 = data['qf2']
        algo.policy = data['trained_policy']
        algo.target_policy = data["target_policy"]
        algo.exploration_policy = data["exploration_policy"]

        if 'n_env_steps_total' in data:
            algo._n_env_steps_total = data["n_env_steps_total"]

        if isinstance(algo.eval_policy, SubgoalPlanner):
            algo.eval_policy.qf = algo.qf1
            algo.eval_policy.mf_policy = algo.policy
        else:
            algo.eval_policy = data["eval_policy"]

        if ptu.gpu_enabled():
            algo.cuda()
        if hasattr(algo, "update_sampler_and_rollout_function"):
            algo.update_sampler_and_rollout_function()

    return update_networks_func