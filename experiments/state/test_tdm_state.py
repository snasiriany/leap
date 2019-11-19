from railrl.misc.exp_util import *
import railrl.misc.hyperparameter as hyp
from railrl.launchers.exp_launcher import tdm_experiment
from railrl.misc.asset_loader import local_path_from_s3_or_local_path
from railrl.config.base_exp_config import variant as base_exp_variant

from multiworld.envs.mujoco.cameras import *

import json
import os.path as osp

variant = deep_update(base_exp_variant, dict(
    rl_variant=dict(
        do_state_exp=True,
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
        snapshot_gap=25,
        vis_kwargs=dict(
            save_period=1,
        ),
        vae_wrapped_env_kwargs=dict(
            vae_input_key_prefix='state',
        ),
        observation_key='state_observation',
        desired_goal_key='state_desired_goal',
        achieved_goal_key='state_acheived_goal',
    ),
    observation_modality='state',
))

env_params = {
    'ant': {
        'env_id': ['AntULongTestEnv-v0'],
        'init_camera': [ant_u_long_camera],

        # 'rl_variant.ckpt_base_path': [
        #     'your-base-path-here',
        # ],
        # 'rl_variant.vae_base_path': [
        #     'your-base-path-here',
        # ],
        # 'rl_variant.num_rl_seeds_per_vae': [
        #     3,
        # ],

        'rl_variant.ckpt': [
            'your-path-here',
        ],
        'rl_variant.vae_path': [
            'your-path-here',
        ],

        'rl_variant.eval_algo': [
            'mb-tdm',
            # 'mf-tdm',
        ],
        'rl_variant.SubgoalPlanner_kwargs.q_input_is_raw_state': [True],
        'rl_variant.SubgoalPlanner_kwargs.cem_optimizer_kwargs.batch_size': [10000],
        'rl_variant.SubgoalPlanner_kwargs.cem_optimizer_kwargs.num_iters': [50],
        'rl_variant.SubgoalPlanner_kwargs.cem_optimizer_kwargs.frac_top_chosen': [[0.25, 0.01]],
        'rl_variant.SubgoalPlanner_kwargs.realistic_subgoal_weight': [1e-1],

        # 'rl_variant.ckpt_epoch': [350],
        'rl_variant.test_ckpt': [True],
        'rl_variant.algo_kwargs.base_kwargs.epoch_freq': [20],

        'rl_variant.vis_kwargs.num_samples_for_video': [6],
        'imsize': [150],
        'rl_variant.vis_kwargs.vis_list': [[
            'v',
        ]],
        'rl_variant.vis_kwargs.vis_blacklist': [[
            'image_desired_goal',
            'reconstr_image_observation',
            'image_desired_subgoal_reproj',
        ]],
    },
}

def process_variant(variant):
    rl_variant = variant['rl_variant']

    if args.debug:
        rl_variant['algo_kwargs']['base_kwargs']['num_rollouts_per_eval'] = 1
        rl_variant['vis_kwargs']['num_samples_for_video'] = 2
        rl_variant['vae_wrapped_env_kwargs']['num_samples_for_latent_histogram'] = 100

    assert rl_variant['eval_algo'] in [
        'mf-tdm',
        'mb-tdm',
    ]

    if 'ckpt_and_vae_path' in rl_variant:
        rl_variant['ckpt'] = rl_variant['ckpt_and_vae_path'][0]
        rl_variant['vae_path'] = rl_variant['ckpt_and_vae_path'][1]
        del rl_variant['ckpt_and_vae_path']

    update_variant_from_ckpt(variant)
    update_variant_from_vae(variant)

    local_path = local_path_from_s3_or_local_path(osp.join(rl_variant['ckpt'], 'variant.json'))
    with open(local_path) as f:
        ckpt_variant = json.load(f)
    ckpt_rl_variant = ckpt_variant.get('rl_variant', ckpt_variant)
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
        rl_variant['do_state_exp'] = False
    elif eval_algo == 'mf-tdm':
        pass

    variant['eval_algo_base'] = rl_variant['eval_algo']
    variant['eval_algo_tag'] = 'mt=' + str(rl_variant['algo_kwargs']['tdm_kwargs']['max_tau'])
    if 'mb' in rl_variant['eval_algo']:
        variant['eval_algo_tag'] = '-'.join([
            variant['eval_algo_tag'],
            'mtps=' + str(rl_variant['SubgoalPlanner_kwargs']['max_tau_per_subprob'])
        ])
    variant['eval_algo'] = '-'.join([
        variant['eval_algo_base'],
        variant['eval_algo_tag']
    ])

if __name__ == '__main__':
    args = parse_args()
    preprocess_args(args)
    search_space = env_params[args.env]
    load_ckpt_base_path_meta_data(search_space)
    load_vae_base_path_meta_data(search_space)
    match_ckpts_and_vaes(search_space)
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