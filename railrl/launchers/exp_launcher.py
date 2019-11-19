from railrl.launchers.vae_exp_launcher_util import (
    train_vae,
    train_reprojection_network_and_update_variant,
)

from railrl.launchers.rl_exp_launcher_util import (
    tdm_td3_experiment,
    tdm_twin_sac_experiment,
    ih_td3_experiment,
    ih_twin_sac_experiment,
)

def tdm_experiment(variant):
    experiment_variant_preprocess(variant)
    rl_variant = variant['rl_variant']
    if 'vae_variant' in variant:
        if not variant['rl_variant'].get('do_state_exp', False):
            train_reprojection_network_and_update_variant(variant)
    if 'sac' in rl_variant['algorithm'].lower():
        tdm_twin_sac_experiment(rl_variant)
    else:
        tdm_td3_experiment(rl_variant)

def ih_experiment(variant):
    experiment_variant_preprocess(variant)
    rl_variant = variant['rl_variant']
    if 'vae_variant' in variant:
        if not variant['rl_variant'].get('do_state_exp', False):
            train_reprojection_network_and_update_variant(variant)
    if 'sac' in rl_variant['algorithm'].lower():
        ih_twin_sac_experiment(rl_variant)
    else:
        ih_td3_experiment(rl_variant)

def vae_experiment(variant):
    experiment_variant_preprocess(variant)
    train_vae(variant["vae_variant"])

def vae_dataset_experiment(variant):
    experiment_variant_preprocess(variant)

    from railrl.launchers.vae_exp_launcher_util import generate_vae_dataset
    from inspect import signature

    vae_variant = variant['vae_variant']
    generate_vae_dataset_fctn = vae_variant.get('generate_vae_data_fctn', generate_vae_dataset)
    sig = signature(generate_vae_dataset_fctn)
    if len(sig.parameters) > 1:
        generate_vae_dataset_fctn(**vae_variant['generate_vae_dataset_kwargs'])
    else:
        generate_vae_dataset_fctn(vae_variant['generate_vae_dataset_kwargs'])

def reproj_experiment(variant):
    experiment_variant_preprocess(variant)
    train_vae(variant["vae_variant"])
    train_reprojection_network_and_update_variant(variant)

def experiment_variant_preprocess(variant):
    rl_variant = variant['rl_variant']
    vae_variant = variant.get('vae_variant', None)
    if 'env_id' in variant:
        assert 'env_class' not in variant
        env_id = variant['env_id']
        rl_variant['env_id'] = env_id
        if vae_variant:
            vae_variant['generate_vae_dataset_kwargs']['env_id'] = env_id
    else:
        env_class = variant['env_class']
        env_kwargs = variant['env_kwargs']
        rl_variant['env_class'] = env_class
        rl_variant['env_kwargs'] = env_kwargs
        if vae_variant:
            vae_variant['generate_vae_dataset_kwargs']['env_class'] = env_class
            vae_variant['generate_vae_dataset_kwargs']['env_kwargs'] = env_kwargs
    init_camera = variant.get('init_camera', None)
    imsize = variant.get('imsize', 84)
    rl_variant['imsize'] = imsize
    rl_variant['init_camera'] = init_camera
    if vae_variant:
        vae_variant['generate_vae_dataset_kwargs']['init_camera'] = init_camera
        vae_variant['generate_vae_dataset_kwargs']['imsize'] = imsize
        vae_variant['imsize'] = imsize
        if vae_variant.get('vae_type', None) == "VAE-state":
            vae_wrapped_env_kwargs = rl_variant.get('vae_wrapped_env_kwargs', {})
            vae_wrapped_env_kwargs['vae_input_key_prefix'] = "state"
        import copy
        vae_variant['vae_wrapped_env_kwargs'] = copy.deepcopy(rl_variant.get('vae_wrapped_env_kwargs', {}))
        if 'vis_kwargs' in vae_variant and 'granularity' in vae_variant['vis_kwargs']:
            vae_variant['vae_wrapped_env_kwargs']['vis_granularity'] = vae_variant['vis_kwargs']['granularity']
        if 'vis_kwargs' in rl_variant and 'granularity' in rl_variant['vis_kwargs']:
            rl_variant['vae_wrapped_env_kwargs']['vis_granularity'] = rl_variant['vis_kwargs']['granularity']
        if 'generate_vae_dataset_kwargs' in vae_variant \
                and 'dataset_path' in vae_variant['generate_vae_dataset_kwargs']:
            vae_variant['vae_wrapped_env_kwargs']['vae_dataset_path'] = \
                vae_variant['generate_vae_dataset_kwargs']['dataset_path']
            rl_variant['vae_wrapped_env_kwargs']['vae_dataset_path'] = \
                vae_variant['generate_vae_dataset_kwargs']['dataset_path']

    dump_video_kwargs = variant.get(
        'dump_video_kwargs',
        dict(
            rows=1,
            pad_length=1,
            pad_color=0,
        ),
    )
    rl_variant['dump_video_kwargs'] = dump_video_kwargs
    rl_variant['dump_video_kwargs']['columns'] = rl_variant['vis_kwargs'].get('num_samples_for_video', 10)
    if vae_variant:
        vae_variant['dump_video_kwargs'] = copy.deepcopy(dump_video_kwargs)
        vae_variant['dump_video_kwargs']['columns'] = vae_variant['vis_kwargs'].get('num_samples_for_video', 10)