from railrl.misc.exp_util import *
from railrl.launchers.exp_launcher import tdm_experiment
import railrl.misc.hyperparameter as hyp
from railrl.config.base_exp_config import variant as base_exp_variant

from multiworld.envs.mujoco.cameras import *
from multiworld.core.image_env import get_image_presampled_goals as image_env_presampled_goals_func

variant = deep_update(base_exp_variant, dict(
    rl_variant=dict(
        do_state_exp=False,
        algo_kwargs=dict(
            tdm_kwargs=dict(
                max_tau=25,
            ),
        ),
        presample_goals=True,
        generate_goal_dataset_fctn=image_env_presampled_goals_func,
        goal_generation_kwargs=dict(
            num_goals_presampled=10000,
        ),
        replay_buffer_kwargs=dict(
            store_distributions=True,
        ),
        use_state_reward=True,
        vae_wrapped_env_kwargs=dict(),
        reward_params=dict(
            type='vectorized_state_distance',
        ),
        train_algo='mf-tdm',
        snapshot_gap=25,
    ),
    observation_modality='latent',
    tag='',
))

env_params = {
    'pm': {
        'env_id': ['Image48PointmassUWallTrainEnvBig-v1'],

        # 'rl_variant.vae_base_path': [
        #     'your-base-path-here',
        # ],
        'rl_variant.vae_path': [
            'your-path-here',
        ],

        'rl_variant.algo_kwargs.base_kwargs.max_path_length': [100],
        'rl_variant.algo_kwargs.tdm_kwargs.max_tau': [25],
        'rl_variant.algo_kwargs.base_kwargs.num_epochs': [200],
        'rl_variant.exploration_type': ['epsilon'],
        'rl_variant.exploration_noise': [0.1],
        'rl_variant.algo_kwargs.base_kwargs.reward_scale': [1e0],

        'rl_variant.snapshot_gap': [10],
        'rl_variant.vis_kwargs.save_period': [20],
        'rl_variant.vis_kwargs.vis_list': [[
            'v',
        ]],
        'rl_variant.vis_kwargs.vis_blacklist': [[
            'reconstr_image_reproj_observation',
        ]],
        'rl_variant.vae_wrapped_env_kwargs.v_func_heatmap_bounds': [(-1.5, 0.0)],
    },
    'pnr': {
        'env_id': ['Image84SawyerPushAndReachArenaTrainEnvBig-v0'],

        # 'rl_variant.vae_base_path': [
        #     'your-base-path-here',
        # ],
        'rl_variant.vae_path': [
            'your-path-here',
        ],

        'rl_variant.algo_kwargs.base_kwargs.max_path_length': [100],
        'rl_variant.algo_kwargs.tdm_kwargs.max_tau': [25],
        'rl_variant.algo_kwargs.base_kwargs.batch_size': [2048],
        'rl_variant.algo_kwargs.base_kwargs.num_epochs': [500],
        'rl_variant.exploration_type': ['ou'],
        'rl_variant.exploration_noise': [0.3],
        'rl_variant.algo_kwargs.base_kwargs.reward_scale': [1e1],

        'rl_variant.snapshot_gap': [25],
        'rl_variant.vis_kwargs.save_period': [50],
        'rl_variant.vis_kwargs.vis_list': [[
            'plt',
        ]],
    },
}

def process_variant(variant):
    rl_variant = variant['rl_variant']

    if args.debug:
        rl_variant['algo_kwargs']['base_kwargs']['num_epochs'] = 4
        rl_variant['algo_kwargs']['base_kwargs']['batch_size'] = 128
        rl_variant['vis_kwargs']['num_samples_for_video'] = 2
        rl_variant['vae_wrapped_env_kwargs']['num_samples_for_latent_histogram'] = 100

        if 'env_kwargs' in variant and 'num_goals_presampled' in variant['env_kwargs']:
            variant['env_kwargs']['num_goals_presampled'] = 10
        if 'goal_generation_kwargs' in rl_variant and \
                'num_goals_presampled' in rl_variant['goal_generation_kwargs']:
            rl_variant['goal_generation_kwargs']['num_goals_presampled'] = 10

    if rl_variant['use_state_reward']:
        assert 'latent' not in rl_variant['reward_params']['type']

        rl_variant['training_mode'] = 'train_env_goals'
        rl_variant['vis_kwargs']['save_video_env_only'] = True
        rl_variant['qf_kwargs']['structure'] = 'none'
        rl_variant['vf_kwargs']['structure'] = 'none'
        rl_variant['replay_buffer_kwargs']['ob_keys_to_save'] = [
            'state_observation', 'state_desired_goal', 'state_achieved_goal',
            'latent_observation', 'latent_desired_goal', 'latent_achieved_goal',
        ]
        rl_variant['replay_buffer_kwargs']['goal_keys'] = ['state_desired_goal', 'latent_desired_goal']
        rl_variant['replay_buffer_kwargs']['desired_goal_keys'] = ['state_desired_goal', 'latent_desired_goal']

    variant['tag'] = 'max-tau-' + str(rl_variant['algo_kwargs']['tdm_kwargs']['max_tau'])
    rl_variant['train_algo'] = rl_variant['train_algo'] + "-" + variant['tag']
    variant['train_algo'] = rl_variant['train_algo']

if __name__ == "__main__":
    args = parse_args()
    preprocess_args(args)
    search_space = env_params[args.env]
    load_vae_base_path_meta_data(search_space)
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters(print_info=False)):
        process_variant(variant)
        run_experiment(
            exp_function=tdm_experiment,
            variant=variant,
            args=args,
            exp_id=exp_id,
            exp_type='train-tdm',
        )
