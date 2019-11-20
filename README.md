# LEAP

This is the codebase for Latent Embeddings for Abstracted Planning (LEAP), from the following paper:

**Planning with Goal Conditioned Policies** 
<br> Soroush Nasiriany*, Vitchyr Pong*, Steven Lin, Sergey Levine 
<br> Neural Information Processing Systems 2019
<br> [Arxiv](https://arxiv.org/abs/1911.08453) | [Website](https://sites.google.com/view/goal-planning/)

This guide contains information about (1) [Installation](#installation), (2) [Experiments](#experiments), and (3) [Setting up Your Own Environments](#setting-up-your-own-environments).

## Installation
### Download Code
- [multiworld](https://github.com/vitchyr/multiworld/tree/leap) (contains environments):```git clone -b leap https://github.com/vitchyr/multiworld```
- [doodad](https://github.com/vitchyr/doodad/tree/leap) (for launching experiments):```git clone -b leap https://github.com/vitchyr/doodad```
  - follow [instructions](https://github.com/vitchyr/doodad/blob/leap/README.md) to setup repo
- [viskit](https://github.com/vitchyr/viskit/tree/leap) (for plotting experiments):```git clone -b leap https://github.com/vitchyr/viskit```
  - follow [instructions](https://github.com/vitchyr/viskit/blob/leap/README.md) to setup repo
- Current codebase: ```git clone https://github.com/snasiriany/leap```
  - install dependencies: `pip install -r requirements.txt`

### Add paths
```
export PYTHONPATH=$PYTHONPATH:/path/to/multiworld/repo
export PYTHONPATH=$PYTHONPATH:/path/to/doodad/repo
export PYTHONPATH=$PYTHONPATH:/path/to/viskit/repo
export PYTHONPATH=$PYTHONPATH:/path/to/leap/repo
```

### Setup Docker Image
You will need to install docker to run experiments. We have provided a dockerfile with all relevant packages. You will use this dockerfile to build your own docker image.

Before setting up the docker image, you will need to obtain a MuJoCo [license](https://www.roboti.us/license.html) to run experiments with the MuJoCo simulator. Obtain the license file `mjkey.txt` and save it for reference.

Set up the docker image with the following steps:
```
cd docker
<add mjkey.txt to current directory>
docker build -t <your-dockerhub-uname>/leap .
docker login --username=<your-dockerhub-uname> --email=<your-email>
docker push <your-dockerhub-uname>/leap
```
### Setup Config File
You must setup the config file for launching experiments, providing paths to your code and data directories.
Inside `railrl/config/launcher_config.py`, fill in the appropriate paths. You can use `railrl/config/launcher_config_template.py` as an example reference.

## Experiments
All experiment files are located in ```experiments```. Each file conforms to the following structure:
```
variant = dict(
  # defualt hyperparam settings for all envs
)

env_params = {
  '<env1>' : {
    # add/override default hyperparam settings for specific env
    # each setting is specified as a dictionary address (key),
    # followed by list of possible options (value).
    # Example in following line:
    # 'rl_variant.algo_kwargs.tdm_kwargs.max_tau': [10, 25, 100],
  },
  '<env2>' : {
    ...
  },
}
```
### Running Experiments
You will need to follow four sequential stages to train and evaluate LEAP:

#### Stage 1: Generate VAE Dataset
```
python vae/generate_vae_dataset.py --env <env-name>
```
#### Stage 2: Train VAE
Train the VAE. There are two variants, image based (for pm and pnr) and state based (for ant):
```
python vae/train_vae.py --env <env-name>
python vae/train_vae_state.py --env <env-name>
```
Before running: locate the corresponding `.npy` file from the previous stage. The `.npy` file contains the VAE dataset. Place the path in your config settings for your env inside the script: 
```
'vae_variant.generate_vae_dataset_kwargs.dataset_path': ['your-npy-path-here'],
```
#### Stage 3: Train RL
Train the RL model. There are two variants (as described in previous stage):
```
python image/train_tdm.py --env <env-name>
python state/train_tdm_state.py --env <env-name>
```
Before running: locate the trained VAE model from the previous stage. Place the path in your config settings for your env inside the script. Complete one of the following options:
```
'rl_variant.vae_base_path': ['your-base-path-here'], # folder of vaes
'rl_variant.vae_path': ['your-path-here'], # one vae
```
#### Stage 4: Test RL
Test the RL model. There are two variants (as described in previous stage):
```
python image/test_tdm.py --env <env-name>
python state/test_tdm_state.py --env <env-name>
```
Before running: located the trained RL model from the previous stage. Place the path in your config settings for your env inside the script. Complete one of the following options:
```
'rl_variant.ckpt_base_path': ['your-base-path-here'], # folder of RL models
'rl_variant.ckpt': ['your-path-here'], # one RL model
```

### Experiment Options
See the `parse_args` function in `railrl/misc/exp_util.py` for the complete list of options. Some important options:
- `env`: the env to run (ant, pnr, pm)
- `label`: name for experiment
- `num_seeds`: number of seeds to run
- `debug`: run with light options for debugging

### Plotting Experiment Results
During training, the results will be saved to a file called under
```
LOCAL_LOG_DIR/<env>/<exp_prefix>/<foldername>
```
 - `LOCAL_LOG_DIR` is the directory set by `railrl.config.launcher_config.LOCAL_LOG_DIR`
 - `<exp_prefix>` is given either to `setup_logger`.
 - `<foldername>` is auto-generated and based off of `exp_prefix`.
 - inside this folder, you should see a file called `progress.csv`. 

Inside the viskit codebase, run:

```
python viskit/frontend.py LOCAL_LOG_DIR/<env>/<exp_prefix>/
```
If visualizing VAE results, add `--dname='vae_progress.csv'` as an option.

## Setting up Your Own Environments
You will need to follow the multiworld template for creating your own environments. You will need to register your environment. For Mujoco envs for example, follow the examples in `multiworld/envs/mujoco/__init__.py` for reference.

## Credit
Much of the coding infrastructure is based on [RLkit](https://github.com/vitchyr/rlkit), which itself is based on [rllab](https://github.com/rll/rllab).
