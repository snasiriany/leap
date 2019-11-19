from railrl.misc.exp_util import *
from railrl.launchers.exp_launcher import tdm_experiment
import railrl.misc.hyperparameter as hyp
from railrl.misc.asset_loader import local_path_from_s3_or_local_path
from railrl.config.base_exp_config import variant as base_exp_variant

from multiworld.envs.mujoco.cameras import *

import json
import os.path as osp

variant = deep_update(base_exp_variant, dict(
    rl_variant=dict(
        do_state_exp=False,
        algo_kwargs=dict(
            base_kwargs=dict(
                num_epochs=1,
                num_rollouts_per_eval=30,
                do_training=False,
            ),
            tdm_kwargs=dict(),
        ),
        replay_buffer_kwargs=dict(
            max_size=int(1E2),
        ),
        vae_wrapped_env_kwargs=dict(),
        snapshot_gap=25,
        vis_kwargs=dict(
            save_period=1,
        ),
    ),
    observation_modality='latent',
))

env_params = {
    'pm': {
        # 'rl_variant.ckpt_base_path': [
        #     'your-base-path-here',
        # ],
        'rl_variant.ckpt': [
            'your-path-here',
        ],

        'rl_variant.eval_algo': ['mb-tdm'],
        'rl_variant.SubgoalPlanner_kwargs.realistic_subgoal_weight': [0.1],

        # 'rl_variant.ckpt_epoch': [490],
        'rl_variant.test_ckpt': [True],
        'rl_variant.algo_kwargs.base_kwargs.num_epochs': [400],
        'rl_variant.algo_kwargs.base_kwargs.epoch_freq': [50],

        'rl_variant.vis_kwargs.vis_list': [[
            'latent_histogram',
            'v',
            'reconstr_image_reproj_observation',
        ]],
        'rl_variant.vis_kwargs.vis_blacklist': [[
            'image_desired_goal',
            'image_observation',
            'reconstr_image_observation',
            'image_desired_subgoal',
        ]],
    },
    'pnr': {
        # 'rl_variant.ckpt_base_path': [
        #     'your-base-path-here',
        # ],
        'rl_variant.ckpt': [
            'your-path-here',
        ],

        'rl_variant.eval_algo': ['mf-tdm'],
        'rl_variant.SubgoalPlanner_kwargs.realistic_subgoal_weight': [0.001],

        # 'rl_variant.ckpt_epoch': [1950],
        'rl_variant.test_ckpt': [True],
        'rl_variant.algo_kwargs.base_kwargs.num_epochs': [1000],
        'rl_variant.algo_kwargs.base_kwargs.epoch_freq': [250],

        'rl_variant.vis_kwargs.vis_list': [[
            'latent_histogram',
            'plt',
        ]],
        'rl_variant.vis_kwargs.vis_blacklist': [[
            'image_desired_subgoal',
        ]],
    },
}

def process_variant(variant):
    rl_variant = variant['rl_variant']

    if args.debug:
        rl_variant['algo_kwargs']['base_kwargs']['num_rollouts_per_eval'] = 1
        rl_variant['vis_kwargs']['num_samples_for_video'] = 2
        rl_variant['vae_wrapped_env_kwargs']['num_samples_for_latent_histogram'] = 100
        variant['train_reprojection_network_variant']['num_epochs'] = 1
        variant['train_reprojection_network_variant']['generate_reprojection_network_dataset_kwargs']['N'] = int(2 ** 8)

        if 'env_kwargs' in variant and 'num_goals_presampled' in variant['env_kwargs']:
            variant['env_kwargs']['num_goals_presampled'] = 10
        if 'goal_generation_kwargs' in rl_variant and \
                'num_goals_presampled' in rl_variant['goal_generation_kwargs']:
            rl_variant['goal_generation_kwargs']['num_goals_presampled'] = 10

    assert rl_variant['eval_algo'] in [
        'mb-tdm',
        'mf-tdm',
    ]

    update_variant_from_ckpt(variant)

    ckpt_path = local_path_from_s3_or_local_path(osp.join(rl_variant['ckpt'], 'variant.json'))
    with open(ckpt_path) as f:
        ckpt_variant = json.load(f)
    if 'rl_variant' in ckpt_variant:
        ckpt_rl_variant = ckpt_variant['rl_variant']
    else:
        ckpt_rl_variant = ckpt_variant['grill_variant'] # backwards compatibility
    if 'mb' in rl_variant['eval_algo']:
        if 'max_tau' not in rl_variant['algo_kwargs']['tdm_kwargs']:
            rl_variant['algo_kwargs']['tdm_kwargs']['max_tau'] = \
                rl_variant['algo_kwargs']['base_kwargs']['max_path_length'] - 1
            if 'extra_time' in rl_variant['SubgoalPlanner_kwargs']:
                rl_variant['algo_kwargs']['tdm_kwargs']['max_tau'] -= \
                    rl_variant['SubgoalPlanner_kwargs']['extra_time']
        if 'max_tau_per_subprob' not in rl_variant['SubgoalPlanner_kwargs']:
            rl_variant['SubgoalPlanner_kwargs']['max_tau_per_subprob'] = \
                ckpt_rl_variant['algo_kwargs']['tdm_kwargs']['max_tau']
    else:
        if 'max_tau' not in rl_variant['algo_kwargs']['tdm_kwargs']:
            rl_variant['algo_kwargs']['tdm_kwargs']['max_tau'] = \
                ckpt_rl_variant['algo_kwargs']['tdm_kwargs']['max_tau']

    eval_algo = rl_variant['eval_algo']
    if eval_algo == 'mb-tdm':
        rl_variant['eval_policy'] = 'SubgoalPlanner'
        rl_variant['SubgoalPlanner_kwargs']['reproject_encoding'] = True
    elif eval_algo == 'mf-tdm':
        pass

    rl_variant['eval_algo_base'] = eval_algo
    rl_variant['eval_algo_tag'] = 'mt=' + str(rl_variant['algo_kwargs']['tdm_kwargs']['max_tau'])
    if 'mb' in rl_variant['eval_algo']:
        rl_variant['eval_algo_tag'] = '-'.join([
            rl_variant['eval_algo_tag'],
            'mtps=' + str(rl_variant['SubgoalPlanner_kwargs']['max_tau_per_subprob'])
        ])
    rl_variant['eval_algo'] = '-'.join([
        rl_variant['eval_algo_base'],
        rl_variant['eval_algo_tag']
    ])
    variant['eval_algo_base'] = rl_variant['eval_algo_base']
    variant['eval_algo_tag'] = rl_variant['eval_algo_tag']
    variant['eval_algo'] = rl_variant['eval_algo']


if __name__ == "__main__":
    args = parse_args()
    preprocess_args(args)
    search_space = env_params[args.env]
    load_ckpt_base_path_meta_data(search_space)
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
            exp_type='test-tdm',
        )
