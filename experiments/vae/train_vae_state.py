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
        vae_type="VAE-state",
        representation_size=16,
        beta=2.5,
        num_epochs=2500,
        generate_vae_dataset_kwargs=dict(
            test_p=.9,
        ),
        vae_kwargs=dict(
            input_channels=3,
            hidden_sizes=[64, 128, 64],
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
    'ant': {
        'env_id': ['AntULongVAEEnv-v0'],
        'init_camera': [ant_u_long_camera],

        'vae_variant.representation_size': [8],
        'vae_variant.beta': [5e-3],
        'vae_variant.generate_vae_dataset_kwargs.dataset_path': [
            'your-npy-path-here',
        ],
        'vae_variant.vae_kwargs.normalize': [True],
        'vae_variant.algo_kwargs.extra_recon_logging.MSE_xy': [list(range(2))],
        'vae_variant.algo_kwargs.extra_recon_logging.MSE_epos': [list(range(17))],
        'vae_variant.algo_kwargs.extra_recon_logging.MSE_qvel': [list(range(17, 31))],
        'vae_variant.algo_kwargs.recon_weights': [
            [10] * 2 + [5] * 15 + [1] * 14
        ],

        'imsize': [200],
        'vae_variant.vis_kwargs.save_video': [False],
        'vae_variant.vis_kwargs.n_recon': [8],
        'vae_variant.vis_kwargs.n_samples': [16],
        'vae_variant.vis_kwargs.save_period': [250],
        'vae_variant.vis_kwargs.vis_list': [[
            'latent_histogram',
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
