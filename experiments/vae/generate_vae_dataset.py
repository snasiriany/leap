from railrl.misc.exp_util import *
from railrl.launchers.exp_launcher import vae_dataset_experiment
import railrl.misc.hyperparameter as hyp

from multiworld.envs.mujoco.cameras import *

variant = dict(
    env_kwargs=dict(),
    imsize=84,
    rl_variant=dict(
        vis_kwargs=dict(),
    ),
    vae_variant=dict(
        generate_vae_dataset_kwargs=dict(
            N=50000,
            oracle_dataset=True,
        ),
        vis_kwargs=dict(),
    ),
)

env_params = {
    'pm': {
        'env_id': ['Image48PointmassUWallTrainEnvBig-v0'],
    },
    'pnr': {
        'env_id': ['Image84SawyerPushAndReachArenaTrainEnvBig-v0'],
    },
    'ant': {
        'env_id': ['AntULongVAEEnv-v0'],
        'init_camera': [ant_u_long_camera],

        'vae_variant.generate_vae_dataset_kwargs.use_images': [False],
        'vae_variant.generate_vae_dataset_kwargs.n_random_steps': [0],
    },
}

def process_variant(variant):
    if args.debug:
        variant['vae_variant']['generate_vae_dataset_kwargs']['N'] = 100

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
            exp_function=vae_dataset_experiment,
            variant=variant,
            args=args,
            exp_id=exp_id,
            exp_type='generate-vae-dataset',
        )
