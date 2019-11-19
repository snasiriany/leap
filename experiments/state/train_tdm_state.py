from railrl.misc.exp_util import *
import railrl.misc.hyperparameter as hyp
from railrl.launchers.exp_launcher import tdm_experiment
from railrl.config.base_exp_config import variant as base_exp_variant

from multiworld.envs.mujoco.cameras import *

variant = deep_update(base_exp_variant, dict(
    rl_variant=dict(
        do_state_exp=True,
        algo_kwargs=dict(
            tdm_kwargs=dict(
                max_tau=25,
            ),
        ),
        train_algo='mf-tdm',
        snapshot_gap=25,
    ),
    observation_modality='state',
    tag='',
))

env_params = {
    'ant': {
        'env_id': ['AntULongTrainEnv-v0'],
        'init_camera': [ant_u_long_camera],

        'rl_variant.algo_kwargs.base_kwargs.max_path_length': [600],
        'rl_variant.algo_kwargs.use_policy_saturation_cost': [True],
        'rl_variant.algo_kwargs.base_kwargs.num_epochs': [500],
        'rl_variant.algo_kwargs.base_kwargs.batch_size': [2048],
        'rl_variant.algo_kwargs.base_kwargs.reward_scale': [10.0],
        'rl_variant.algo_kwargs.tdm_kwargs.max_tau': [
            50,
            # 599,
        ],
        'rl_variant.qf_kwargs.structure': ['none'],
        'rl_variant.exploration_type': ['ou'],
        'rl_variant.exploration_noise': [0.3],

        'rl_variant.snapshot_gap': [30],
        'rl_variant.vis_kwargs.save_period': [60],
        'rl_variant.vis_kwargs.num_samples_for_video': [6],
        'imsize': [100],
        'rl_variant.vis_kwargs.vis_list': [[
            'v',
        ]],
    },
}

def process_variant(variant):
    rl_variant = variant['rl_variant']

    if args.debug:
        rl_variant['algo_kwargs']['base_kwargs']['num_epochs'] = 4
        rl_variant['algo_kwargs']['base_kwargs']['batch_size'] = 128
        rl_variant['vis_kwargs']['num_samples_for_video'] = 2
        rl_variant['vis_kwargs']['save_period'] = 2

        if 'env_kwargs' in variant and 'num_goals_presampled' in variant['env_kwargs']:
            variant['env_kwargs']['num_goals_presampled'] = 10
        if 'vis_kwargs' in rl_variant and 'num_goals_presampled' in rl_variant['vis_kwargs']:
            rl_variant['vis_kwargs']['num_goals_presampled'] = 10

    variant['tag'] = 'max-tau-' + str(rl_variant['algo_kwargs']['tdm_kwargs']['max_tau'])
    rl_variant['train_algo'] = rl_variant['train_algo'] + "-" + variant['tag']
    variant['train_algo'] = rl_variant['train_algo']

if __name__ == "__main__":
    args = parse_args()
    preprocess_args(args)
    search_space = env_params[args.env]
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
