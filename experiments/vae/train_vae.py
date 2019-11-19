from railrl.misc.exp_util import *
from railrl.launchers.exp_launcher import vae_experiment
import railrl.misc.hyperparameter as hyp

from multiworld.envs.mujoco.cameras import *

variant = dict(
    env_kwargs=dict(),
    imsize=84,
    rl_variant=dict(
        observation_key='latent_observation',
        desired_goal_key='latent_desired_goal',
        vae_wrapped_env_kwargs=dict(),
        vis_kwargs=dict(
            save_video=True,
            num_samples_for_video=10,
        ),
    ),
    vae_variant=dict(
        representation_size=16,
        beta=2.5,
        num_epochs=2500,
        generate_vae_dataset_kwargs=dict(
            test_p=.9,
        ),
        vae_kwargs=dict(
            input_channels=3,
        ),
        algo_kwargs=dict(
            lr=1e-3,
            train_data_workers=0,
            use_linear_dynamics=True,
            linearity_weight=0.0,
        ),
        vis_kwargs=dict(
            num_samples_for_video=10,
            save_video=True,
            save_video_env_only=True,
        ),
        snapshot_gap=1000,
    ),
)

env_params = {
    'pm': {
        'env_id': ['Image48PointmassUWallTrainEnvBig-v0'],
        'imsize': [48],

        'vae_variant.beta': [7.5],
        'vae_variant.generate_vae_dataset_kwargs.dataset_path': [
            'your-npy-path-here',
        ],
        'vae_variant.vae_type': ['VAE'],
        'vae_variant.vae_kwargs.gaussian_decoder': [False],
        'vae_variant.vae_kwargs.use_sigmoid_for_decoder': [True],

        'vae_variant.vis_kwargs.save_period': [500],
        'vae_variant.vis_kwargs.vis_list': [[
            'latent_histogram_2d',
            'latent_histogram',
            'rew',
            'latent_representation',
        ]],
    },
    'pnr': {
        'env_id': ['Image84SawyerPushAndReachArenaTrainEnvBig-v0'],

        'vae_variant.beta': [2.5],
        'vae_variant.generate_vae_dataset_kwargs.dataset_path': [
            'your-npy-path-here',
        ],
        'vae_variant.vae_type': ['VAE2'],
        'vae_variant.vae_kwargs.gaussian_decoder': [True],
        'vae_variant.vae_kwargs.use_sigmoid_for_decoder': [False],
        'vae_variant.vae_kwargs.num_filters': [8],

        'vae_variant.vis_kwargs.save_period': [500],
        'vae_variant.vis_kwargs.vis_list': [[
            'latent_histogram_2d',
            'latent_histogram',
            'plt',
        ]],
    },
}

def process_variant(variant):
    if args.debug:
        variant['vae_variant']['num_epochs'] = 10
        variant['vae_variant']['vis_kwargs']['save_period'] = 2
        variant['vae_variant']['vis_kwargs']['num_samples_for_video'] = 2
        variant['rl_variant']['vae_wrapped_env_kwargs']['num_samples_for_latent_histogram'] = 100

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
            exp_function=vae_experiment,
            variant=variant,
            args=args,
            exp_id=exp_id,
            exp_type='train-vae',
        )
