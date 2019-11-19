import argparse
import collections
import json
from railrl.launchers.launcher_util import run_experiment as exp_launcher_function
from railrl.misc.asset_loader import local_path_from_s3_or_local_path
import os.path as osp

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str)
    parser.add_argument('--mode', type=str, default='local_docker')
    parser.add_argument('--label', type=str, default='')
    parser.add_argument('--num_seeds', type=int, default=1)
    parser.add_argument('--no_gpu', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--first_variant_only', action='store_true')
    parser.add_argument('--max_exps_per_instance', type=int, default=2)
    parser.add_argument('--no_video',  action='store_true')
    return parser.parse_args()

def deep_update(source, overrides):
    '''
    Update a nested dictionary or similar mapping.
    Modify ``source`` in place.

    Copied from: https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    '''
    for key, value in overrides.items():
        if isinstance(value, collections.Mapping) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source

def run_experiment(
        exp_function,
        variant,
        args,
        exp_id,
        exp_type,
):
    snapshot_gap = update_snapshot_gap_and_save_period(variant)
    load_vae_meta_data(variant)
    update_video_logging(variant, args)

    exp_prefix = get_exp_prefix(args, variant, type=exp_type)
    num_exps_for_instances = get_num_exps_for_instances(args)
    for num_exps in num_exps_for_instances:
        run_experiment_kwargs = get_instance_kwargs(args, num_exps, variant)
        exp_launcher_function(
            exp_function,
            variant=variant,
            exp_folder=args.env,
            exp_prefix=exp_prefix,
            exp_id=exp_id,
            snapshot_gap=snapshot_gap,
            snapshot_mode='gap_and_last',

            **run_experiment_kwargs
        )

    if args.first_variant_only:
        exit()

def preprocess_args(args):
    if args.mode == 'local' and args.label == '':
        args.label = 'local'

def get_instance_kwargs(args, num_exps, variant):
    mode = args.mode
    ssh_host = None
    gpu_id = args.gpu_id

    if mode == 'local_docker':
        interactive_docker = True
    else:
        interactive_docker = False

    instance_kwargs = dict(
        mode=mode,
        ssh_host=ssh_host,
        use_gpu=(not args.no_gpu),
        gpu_id=gpu_id,
        num_exps_per_instance=int(num_exps),
        interactive_docker=interactive_docker,
    )

    variant['instance_kwargs'] = instance_kwargs
    return instance_kwargs

def get_exp_prefix(args, variant, type='train-tdm'):
    if 'vae_variant' in variant:
        if 'state' in variant['vae_variant'].get('vae_type', 'VAE'):
            data_type = 'full-state'
        else:
            data_type = None
    else:
        data_type = 'full-state'

    prefix_list = [type, data_type, args.label]

    while None in prefix_list: prefix_list.remove(None)
    while '' in prefix_list: prefix_list.remove('')
    exp_prefix = '-'.join(prefix_list)

    return exp_prefix

def get_num_exps_for_instances(args):
    import numpy as np
    import math

    if args.mode == 'ec2' and (not args.no_gpu):
        max_exps_per_instance = args.max_exps_per_instance
    else:
        max_exps_per_instance = 1

    num_exps_for_instances = np.ones(int(math.ceil(args.num_seeds / max_exps_per_instance)), dtype=np.int32) \
                             * max_exps_per_instance
    num_exps_for_instances[-1] -= (np.sum(num_exps_for_instances) - args.num_seeds)

    return num_exps_for_instances

def load_vae_meta_data(variant):
    from railrl.misc.asset_loader import local_path_from_s3_or_local_path
    import os.path as osp
    import json

    rl_variant = variant['rl_variant']
    if 'vae_path' in rl_variant:
        local_path = local_path_from_s3_or_local_path(osp.join(rl_variant['vae_path'], 'variant.json'))
        with open(local_path) as f:
            data = json.load(f)
            variant['vae_exp_prefix'] = data['exp_prefix']
            variant['vae_exp_id'] = data['exp_id']
            variant['vae_seed'] = data['seed']
            if 'vae_variant' in data:
                variant['vae_variant'] = data['vae_variant']
            else:
                variant['vae_variant'] = data['train_vae_variant']
    if 'reproj_vae_path' in rl_variant:
        local_path = local_path_from_s3_or_local_path(osp.join(rl_variant['reproj_vae_path'], 'variant.json'))
        with open(local_path) as f:
            data = json.load(f)
            variant['reproj_vae_exp_prefix'] = data['exp_prefix']
            variant['reproj_vae_exp_id'] = data['exp_id']
            variant['reproj_vae_seed'] = data['seed']

def update_snapshot_gap_and_save_period(variant):
    import math

    rl_variant = variant['rl_variant']
    if 'algo_kwargs' not in rl_variant:
        return 0

    if 'vae_variant' in variant:
        if 'snapshot_gap' not in variant['vae_variant']:

            variant['vae_variant']['snapshot_gap'] = \
                int(math.ceil(variant['vae_variant']['num_epochs'] / 10))

    if 'snapshot_gap' not in rl_variant:
        rl_variant['snapshot_gap'] = int(math.ceil(rl_variant['algo_kwargs']['base_kwargs']['num_epochs'] / 10))

    if 'save_period' not in rl_variant['vis_kwargs']:
        rl_variant['vis_kwargs']['save_period'] = \
            int(math.ceil(rl_variant['algo_kwargs']['base_kwargs']['num_epochs'] / 10))

    return rl_variant['snapshot_gap']

def update_video_logging(variant, args):
    if not args.no_video:
        return
    if 'rl_variant' in variant:
        variant['rl_variant']['vis_kwargs']['save_video'] = False
        variant['rl_variant']['vis_kwargs']['vis_list'] = []
    if 'vae_variant' in variant:
        variant['vae_variant']['vis_kwargs']['save_video'] = False
        variant['vae_variant']['vis_kwargs']['vis_list'] = []

def load_ckpt_base_path_meta_data(search_space):
    base_data_directory = '/home/soroush/data/s3'
    import os

    if search_space.get('rl_variant.ckpt_base_path', None) is not None \
            and len(search_space['rl_variant.ckpt_base_path']) > 0:
        search_space['rl_variant.ckpt'] = []
        for base_path in search_space['rl_variant.ckpt_base_path']:
            for exp_path in os.listdir(os.path.join(base_data_directory, base_path)):
                if os.path.isdir(os.path.join(base_data_directory, base_path, exp_path)):
                    id_and_seed = exp_path.split('id')[-1]
                    id = int(id_and_seed[:3])
                    seed = int(id_and_seed.split('s')[-1])
                    path = os.path.join(base_path, exp_path)
                    if path not in search_space.get('rl_variant.ckpt_exclude_path', []) and \
                            id not in search_space.get('rl_variant.ckpt_exclude_id', []):
                        search_space['rl_variant.ckpt'].append(path)

    if 'rl_variant.ckpt_base_path' in search_space:
        del search_space['rl_variant.ckpt_base_path']
    if 'rl_variant.ckpt_exclude_path' in search_space:
        del search_space['rl_variant.ckpt_exclude_path']
    if 'rl_variant.ckpt_exclude_id' in search_space:
        del search_space['rl_variant.ckpt_exclude_id']

def load_vae_base_path_meta_data(search_space):
    base_data_directory = '/home/soroush/data/s3'
    import os

    if search_space.get('rl_variant.vae_base_path', None) is not None \
            and len(search_space['rl_variant.vae_base_path']) > 0:
        search_space['rl_variant.vae_path'] = []
        for path in search_space['rl_variant.vae_base_path']:
            search_space['rl_variant.vae_path'] += [
                os.path.join(path, o)#, 'params.pkl')
                for o in os.listdir(os.path.join(base_data_directory, path))
                if os.path.isdir(os.path.join(base_data_directory, path, o))
            ]
    if 'rl_variant.vae_base_path' in search_space:
        del search_space['rl_variant.vae_base_path']
        
def match_ckpts_and_vaes(search_space):
    if search_space.get('rl_variant.num_rl_seeds_per_vae', None) is not None:
        num_rl_seeds_per_vae = search_space['rl_variant.num_rl_seeds_per_vae'][0]
        search_space['rl_variant.ckpt_and_vae_path'] = []
        ckpt_counter = 0
        for vae_path in search_space['rl_variant.vae_path']:
            for i in range(num_rl_seeds_per_vae):
                search_space['rl_variant.ckpt_and_vae_path'].append(
                    [search_space['rl_variant.ckpt'][ckpt_counter], vae_path]
                )
                ckpt_counter += 1

    if 'rl_variant.ckpt_and_vae_path' in search_space:
        del search_space['rl_variant.ckpt']
        del search_space['rl_variant.vae_path']

def update_variant_from_ckpt(variant):
    rl_variant = variant['rl_variant']
    local_path = local_path_from_s3_or_local_path(osp.join(rl_variant['ckpt'], 'variant.json'))
    with open(local_path) as f:
        ckpt_variant = json.load(f)
    ckpt_rl_variant = ckpt_variant.get('rl_variant', None)
    if ckpt_rl_variant is None:
        ckpt_rl_variant = ckpt_variant.get('grill_variant', ckpt_variant) # backwards compatibility

    env_kwargs = ckpt_variant['env_kwargs']
    env_kwargs.update(variant['env_kwargs'])
    variant['env_kwargs'] = env_kwargs

    rl_variant['algorithm'] = ckpt_rl_variant['algorithm']
    variant['ckpt_exp_prefix'] = ckpt_variant['exp_prefix']
    variant['ckpt_exp_id'] = ckpt_variant['exp_id']
    variant['ckpt_seed'] = ckpt_variant['seed']

    if 'vae_path' in ckpt_rl_variant:
        rl_variant['vae_path'] = ckpt_rl_variant['vae_path']

    if 'vae_variant' in ckpt_variant:
        variant['vae_variant'] = ckpt_variant['vae_variant']
    elif 'train_vae_variant' in ckpt_variant:  # backwards compatibility
        variant['vae_variant'] = ckpt_variant['train_vae_variant']

    if 'num_updates_per_env_step' in ckpt_rl_variant['algo_kwargs']['base_kwargs']:
        rl_variant['algo_kwargs']['base_kwargs']['num_updates_per_env_step'] = \
            ckpt_rl_variant['algo_kwargs']['base_kwargs']['num_updates_per_env_step']

    if 'max_path_length' not in rl_variant['algo_kwargs']['base_kwargs']:
        rl_variant['algo_kwargs']['base_kwargs']['max_path_length'] = \
            ckpt_rl_variant['algo_kwargs']['base_kwargs']['max_path_length']

    if rl_variant.get('test_ckpt', False) and rl_variant['algo_kwargs']['base_kwargs']['num_epochs'] == 1:
        rl_variant['algo_kwargs']['base_kwargs']['num_epochs'] = \
            ckpt_rl_variant['algo_kwargs']['base_kwargs']['num_epochs']

    rl_variant['exploration_type'] = ckpt_rl_variant['exploration_type']
    rl_variant['exploration_noise'] = ckpt_rl_variant['exploration_noise']

    if 'reward_params' in ckpt_rl_variant:
        rl_variant['reward_params'] = ckpt_rl_variant['reward_params']
    if 'vae_wrapped_env_kwargs' in ckpt_rl_variant:
        for k in ckpt_rl_variant['vae_wrapped_env_kwargs']:
            if k in ['test_noisy_encoding', 'num_samples_for_latent_histogram'] \
                    and k in rl_variant['vae_wrapped_env_kwargs']:
                pass
            else:
                rl_variant['vae_wrapped_env_kwargs'][k] = \
                    ckpt_rl_variant['vae_wrapped_env_kwargs'][k]

    rl_variant['algo_kwargs']['base_kwargs']['reward_scale'] = \
        ckpt_rl_variant['algo_kwargs']['base_kwargs'].get('reward_scale', 1.0)

    if 'env_class' not in variant and 'env_id' not in variant and 'env_id' in ckpt_variant:
        variant['env_id'] = ckpt_variant['env_id'].replace('Train', 'Test')

def update_variant_from_vae(variant):
    rl_variant = variant['rl_variant']
    if 'vae_path' in rl_variant:
        local_path = local_path_from_s3_or_local_path(osp.join(rl_variant['vae_path'], 'variant.json'))
        with open(local_path) as f:
            data = json.load(f)
            variant['vae_exp_prefix'] = data['exp_prefix']
            variant['vae_exp_id'] = data['exp_id']
            variant['vae_seed'] = data['seed']
            if 'vae_variant' in data:
                data_vae_variant = data['vae_variant']
            else:
                data_vae_variant = data['train_vae_variant'] # backwards compatibility
            variant['vae_variant'] = data_vae_variant
            vae_wrapped_env_kwargs = rl_variant['vae_wrapped_env_kwargs']
            vae_wrapped_env_kwargs['vae_dataset_path'] = \
                data_vae_variant['generate_vae_dataset_kwargs']['dataset_path']
