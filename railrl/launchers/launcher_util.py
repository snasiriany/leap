import json
import os
import os.path as osp
import pickle
import random
import sys
import time
import uuid
from collections import namedtuple

import __main__ as main
import datetime
import dateutil.tz
import numpy as np

import railrl.pythonplusplus as ppp
from railrl.core import logger as default_logger
from railrl.config import launcher_config

GitInfo = namedtuple(
    'GitInfo',
    [
        'directory',
        'code_diff',
        'code_diff_staged',
        'commit_hash',
        'branch_name',
    ],
)


ec2_okayed = False
gpu_ec2_okayed = False
first_sss_launch = True

try:
    import doodad.mount as mount
    from doodad.utils import REPO_DIR
    CODE_MOUNTS = [
        mount.MountLocal(local_dir=REPO_DIR, pythonpath=True),
    ]
    for code_dir in launcher_config.CODE_DIRS_TO_MOUNT:
        CODE_MOUNTS.append(mount.MountLocal(local_dir=code_dir, pythonpath=True))

    NON_CODE_MOUNTS = []
    for non_code_mapping in launcher_config.DIR_AND_MOUNT_POINT_MAPPINGS:
        NON_CODE_MOUNTS.append(mount.MountLocal(**non_code_mapping))

    SSS_CODE_MOUNTS = []
    SSS_NON_CODE_MOUNTS = []
    if hasattr(launcher_config, 'SSS_DIR_AND_MOUNT_POINT_MAPPINGS'):
        for non_code_mapping in launcher_config.SSS_DIR_AND_MOUNT_POINT_MAPPINGS:
            SSS_NON_CODE_MOUNTS.append(mount.MountLocal(**non_code_mapping))
    if hasattr(launcher_config, 'SSS_CODE_DIRS_TO_MOUNT'):
        for code_dir in launcher_config.SSS_CODE_DIRS_TO_MOUNT:
            SSS_CODE_MOUNTS.append(
                mount.MountLocal(local_dir=code_dir, pythonpath=True)
            )
except ImportError:
    print("doodad not detected")

target_mount = None


def run_experiment(
        method_call,
        mode='local',
        exp_folder=None,
        exp_prefix='default',
        seed=None,
        variant=None,
        exp_id=0,
        prepend_date_to_exp_prefix=True,
        use_gpu=False,
        gpu_id=None,
        snapshot_mode='last',
        snapshot_gap=1,
        base_log_dir=None,
        local_input_dir_to_mount_point_dict=None,
        # local settings
        skip_wait=False,
        logger=default_logger,
        verbose=False,
        trial_dir_suffix=None,
        num_exps_per_instance=1,
        # ssh settings
        ssh_host=None,
        interactive_docker=False,
):
    """
    Usage:
    ```
    def foo(variant):
        x = variant['x']
        y = variant['y']
        logger.log("sum", x+y)
    variant = {
        'x': 4,
        'y': 3,
    }
    run_experiment(foo, variant, exp_prefix="my-experiment")
    ```
    Results are saved to
    `base_log_dir/<date>-my-experiment/<date>-my-experiment-<unique-id>`
    By default, the base_log_dir is determined by
    `config.LOCAL_LOG_DIR/`
    :param method_call: a function that takes in a dictionary as argument
    :param mode: A string:
     - 'local'
     - 'local_docker'
     - 'here_no_doodad': Run without doodad call
    :param exp_prefix: name of experiment
    :param seed: Seed for this specific trial.
    :param variant: Dictionary
    :param exp_id: One experiment = one variant setting + multiple seeds
    :param prepend_date_to_exp_prefix: If False, do not prepend the date to
    the experiment directory.
    :param use_gpu:
    :param snapshot_mode: See rllab.logger
    :param snapshot_gap: See rllab.logger
    :param base_log_dir: Will over
    :param local_input_dir_to_mount_point_dict: Dictionary for doodad.
    :param ssh_host: the name of the host you want to ssh onto, should correspond to an entry in
    launcher_config.py of the following form:
    SSH_HOSTS=dict(
        ssh_host=dict(
            username='username',
            hostname='hostname/ip address',
        )
    )
    - if ssh_host is set to None, you will use ssh_host specified by
    config.SSH_DEFAULT_HOST
    :return:
    """
    try:
        import doodad
        import doodad.mode
        import doodad.ssh
    except ImportError:
        print("Doodad not set up! Running experiment here.")
        mode = 'here_no_doodad'
    global ec2_okayed
    global gpu_ec2_okayed
    global target_mount
    global first_sss_launch

    """
    Sanitize inputs as needed
    """
    if seed is None:
        seed = random.randint(0, 100000)
    if variant is None:
        variant = {}
    if base_log_dir is None:
        if mode == 'ssh':
            base_log_dir = launcher_config.SSH_LOG_DIR
        else:
            base_log_dir = launcher_config.LOCAL_LOG_DIR

    if exp_folder is not None:
        base_log_dir = os.path.join(base_log_dir, exp_folder)

    for key, value in ppp.recursive_items(variant):
        # This check isn't really necessary, but it's to prevent myself from
        # forgetting to pass a variant through dot_map_dict_to_nested_dict.
        if isinstance(key, str) and "." in key:
            raise Exception(
                "Variants should not have periods in keys. Did you mean to "
                "convert {} into a nested dictionary?".format(key)
            )
    if prepend_date_to_exp_prefix:
        exp_prefix = time.strftime("%m-%d") + "-" + exp_prefix
    variant['seed'] = str(seed)
    variant['exp_id'] = str(exp_id)
    variant['exp_prefix'] = str(exp_prefix)

    try:
        import git
        doodad_path = osp.abspath(osp.join(
            osp.dirname(doodad.__file__),
            os.pardir
        ))
        dirs = launcher_config.CODE_DIRS_TO_MOUNT + [doodad_path]

        git_infos = []
        for directory in dirs:
            # Idk how to query these things, so I'm just doing try-catch
            try:
                repo = git.Repo(directory)
                try:
                    branch_name = repo.active_branch.name
                except TypeError:
                    branch_name = '[DETACHED]'
                git_infos.append(GitInfo(
                    directory=directory,
                    code_diff=repo.git.diff(None),
                    code_diff_staged=repo.git.diff('--staged'),
                    commit_hash=repo.head.commit.hexsha,
                    branch_name=branch_name,
                ))
            except git.exc.InvalidGitRepositoryError:
                pass
    except ImportError:
        git_infos = None
    run_experiment_kwargs = dict(
        exp_prefix=exp_prefix,
        variant=variant,
        exp_id=exp_id,
        seed=seed,
        use_gpu=use_gpu,
        snapshot_mode=snapshot_mode,
        snapshot_gap=snapshot_gap,
        git_infos=git_infos,
        script_name=main.__file__,
        logger=logger,
        trial_dir_suffix=trial_dir_suffix,
    )
    if mode == 'here_no_doodad':
        run_experiment_kwargs['base_log_dir'] = base_log_dir
        return run_experiment_here(
            method_call,
            **run_experiment_kwargs
        )

    """
    GPU vs normal configs
    """
    if use_gpu:
        docker_image = launcher_config.GPU_DOODAD_DOCKER_IMAGE_SSH
    else:
        docker_image = launcher_config.DOODAD_DOCKER_IMAGE_SSH

    """
    Create mode
    """
    if mode == 'local':
        dmode = doodad.mode.Local()
    elif mode == 'local_docker':
        dmode = doodad.mode.LocalDocker(
            image=docker_image,
            gpu=use_gpu,
        )
    elif mode == 'ssh':
        if ssh_host == None:
            ssh_dict = launcher_config.SSH_HOSTS[launcher_config.SSH_DEFAULT_HOST]
        else:
            ssh_dict = launcher_config.SSH_HOSTS[ssh_host]
        credentials = doodad.ssh.credentials.SSHCredentials(
            username=ssh_dict['username'],
            hostname=ssh_dict['hostname'],
            identity_file=launcher_config.SSH_PRIVATE_KEY
        )
        dmode = doodad.mode.SSHDocker(
            tmp_dir=launcher_config.SSH_TMP_DIR,
            credentials=credentials,
            image=docker_image,
            gpu=use_gpu,
        )
    else:
        raise NotImplementedError("Mode not supported: {}".format(mode))

    """
    Get the mounts
    """
    mounts = create_mounts(
        base_log_dir=base_log_dir,
        mode=mode,
        local_input_dir_to_mount_point_dict=local_input_dir_to_mount_point_dict,
        exp_folder=exp_folder,
    )

    """
    Get the outputs
    """
    mode_specific_kwargs = {}
    launch_locally = None
    target = launcher_config.RUN_DOODAD_EXPERIMENT_SCRIPT_PATH
    if mode == 'local':
        base_log_dir_for_script = base_log_dir
        # The snapshot dir will be automatically created
        snapshot_dir_for_script = None
        mode_specific_kwargs['skip_wait'] = skip_wait
    elif mode == 'local_docker':
        base_log_dir_for_script = base_log_dir
        # The snapshot dir will be automatically created
        snapshot_dir_for_script = None
        mode_specific_kwargs['interactive_docker'] = interactive_docker
    elif mode == 'ssh':
        base_log_dir_for_script = base_log_dir
        if exp_prefix is not None:
            base_log_dir_for_script = osp.join(base_log_dir_for_script, exp_folder)

        # The snapshot dir will be automatically created
        snapshot_dir_for_script = None
        mode_specific_kwargs['interactive_docker'] = interactive_docker
    elif mode == 'here_no_doodad':
        base_log_dir_for_script = base_log_dir
        # The snapshot dir will be automatically created
        snapshot_dir_for_script = None
    else:
        raise NotImplementedError("Mode not supported: {}".format(mode))
    run_experiment_kwargs['base_log_dir'] = base_log_dir_for_script
    target_mount = doodad.launch_python(
        target=target,
        mode=dmode,
        mount_points=mounts,
        args={
            'method_call': method_call,
            'output_dir': snapshot_dir_for_script,
            'run_experiment_kwargs': run_experiment_kwargs,
            'mode': mode,
        },
        use_cloudpickle=True,
        target_mount=target_mount,
        verbose=verbose,
        launch_locally=launch_locally,
        gpu_id=gpu_id,
        **mode_specific_kwargs
    )


def create_mounts(
        mode,
        base_log_dir,
        local_input_dir_to_mount_point_dict=None,
        exp_folder=None,
):
    code_mounts = CODE_MOUNTS
    non_code_mounts = NON_CODE_MOUNTS

    if local_input_dir_to_mount_point_dict is None:
        local_input_dir_to_mount_point_dict = {}
    else:
        raise NotImplementedError("TODO(vitchyr): Implement this")

    mounts = [m for m in code_mounts]
    for dir, mount_point in local_input_dir_to_mount_point_dict.items():
        mounts.append(mount.MountLocal(
            local_dir=dir,
            mount_point=mount_point,
            pythonpath=False,
        ))

    if mode != 'local':
        for m in non_code_mounts:
            mounts.append(m)

    if mode in ['local', 'local_singularity', 'slurm_singularity', 'sss']:
        # To save directly to local files (singularity does this), skip mounting
        output_mount = mount.MountLocal(
            local_dir=base_log_dir,
            mount_point=None,
            output=True,
        )
    elif mode == 'local_docker':
        output_mount = mount.MountLocal(
            local_dir=launcher_config.LOCAL_LOG_DIR,
            mount_point=launcher_config.LOCAL_LOG_DIR,
            output=True,
        )
    else:
        raise NotImplementedError("Mode not supported: {}".format(mode))
    mounts.append(output_mount)

    return mounts


def save_experiment_data(dictionary, log_dir):
    with open(log_dir + '/experiment.pkl', 'wb') as handle:
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

def run_experiment_here(
        experiment_function,
        variant=None,
        exp_id=0,
        seed=0,
        use_gpu=True,
        # Logger params:
        exp_prefix="default",
        snapshot_mode='last',
        snapshot_gap=1,
        git_infos=None,
        script_name=None,
        logger=default_logger,
        trial_dir_suffix=None,
        randomize_seed=False,
        **setup_logger_kwargs
):
    """
    Run an experiment locally without any serialization.
    :param experiment_function: Function. `variant` will be passed in as its
    only argument.
    :param exp_prefix: Experiment prefix for the save file.
    :param variant: Dictionary passed in to `experiment_function`.
    :param exp_id: Experiment ID. Should be unique across all
    experiments. Note that one experiment may correspond to multiple seeds,.
    :param seed: Seed used for this experiment.
    :param use_gpu: Run with GPU. By default False.
    :param script_name: Name of the running script
    :param log_dir: If set, set the log directory to this. Otherwise,
    the directory will be auto-generated based on the exp_prefix.
    :return:
    """
    if variant is None:
        variant = {}
    variant['exp_id'] = str(exp_id)

    if randomize_seed or (seed is None and 'seed' not in variant):
        seed = random.randint(0, 100000)
        variant['seed'] = str(seed)
    reset_execution_environment(logger=logger)

    actual_log_dir = setup_logger(
        exp_prefix=exp_prefix,
        variant=variant,
        exp_id=exp_id,
        seed=seed,
        snapshot_mode=snapshot_mode,
        snapshot_gap=snapshot_gap,
        git_infos=git_infos,
        script_name=script_name,
        logger=logger,
        trial_dir_suffix=trial_dir_suffix,
        **setup_logger_kwargs
    )

    set_seed(seed)
    from railrl.torch.pytorch_util import set_gpu_mode
    set_gpu_mode(use_gpu)

    run_experiment_here_kwargs = dict(
        variant=variant,
        exp_id=exp_id,
        seed=seed,
        use_gpu=use_gpu,
        exp_prefix=exp_prefix,
        snapshot_mode=snapshot_mode,
        snapshot_gap=snapshot_gap,
        git_infos=git_infos,
        script_name=script_name,
        **setup_logger_kwargs
    )
    save_experiment_data(
        dict(
            run_experiment_here_kwargs=run_experiment_here_kwargs
        ),
        actual_log_dir
    )
    return experiment_function(variant)


def create_trial_name(exp_prefix, exp_id=0, seed=0):
    """
    Create a semi-unique experiment name that has a timestamp
    :param exp_prefix:
    :param exp_id:
    :return:
    """
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    return "%s_%s_id%03d--s%d" % (exp_prefix, timestamp, exp_id, seed)


def create_log_dir(
        exp_prefix,
        exp_id=0,
        seed=0,
        base_log_dir=None,
        variant=None,
        trial_dir_suffix=None,
        include_exp_prefix_sub_dir=True,
):
    """
    Creates and returns a unique log directory.
    :param exp_prefix: All experiments with this prefix will have log
    directories be under this directory.
    :param exp_id: Different exp_ids will be in different directories.
    :return:
    """
    if variant and "run_id" in variant and variant["run_id"] is not None:
        run_id, exp_id = variant["run_id"], variant["exp_id"]
        trial_name = "run{}/id{}".format(run_id, exp_id)
    else:
        trial_name = create_trial_name(exp_prefix, exp_id=exp_id,
                                       seed=seed)
    if trial_dir_suffix is not None:
        trial_name = "{}-{}".format(trial_name, trial_dir_suffix)
    if base_log_dir is None:
        base_log_dir = launcher_config.LOCAL_LOG_DIR
    if include_exp_prefix_sub_dir:
        log_dir = osp.join(base_log_dir, exp_prefix.replace("_", "-"), trial_name)
    else:
        log_dir = osp.join(base_log_dir, trial_name)
    if osp.exists(log_dir):
        print("WARNING: Log directory already exists {}".format(log_dir))
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def setup_logger(
        exp_prefix="default",
        variant=None,
        text_log_file="debug.log",
        variant_log_file="variant.json",
        tabular_log_file="progress.csv",
        snapshot_mode="last",
        snapshot_gap=1,
        log_tabular_only=False,
        log_dir=None,
        git_infos=None,
        script_name=None,
        logger=default_logger,
        **create_log_dir_kwargs
):
    """
    Set up logger to have some reasonable default settings.
    Will save log output to
        based_log_dir/exp_prefix/exp_name.
    exp_name will be auto-generated to be unique.
    If log_dir is specified, then that directory is used as the output dir.
    :param exp_prefix: The sub-directory for this specific experiment.
    :param exp_id: The number of the specific experiment run within this
    experiment.
    :param variant:
    :param base_log_dir: The directory where all log should be saved.
    :param text_log_file:
    :param variant_log_file:
    :param tabular_log_file:
    :param snapshot_mode:
    :param log_tabular_only:
    :param snapshot_gap:
    :param log_dir:
    :param git_infos:
    :param script_name: If set, save the script name to this.
    :return:
    """
    first_time = log_dir is None
    if first_time:
        log_dir = create_log_dir(
            exp_prefix,
            variant=variant,
            **create_log_dir_kwargs
        )

    if variant is not None:
        if 'unique_id' not in variant:
            variant['unique_id'] = str(uuid.uuid4())
        logger.log("Variant:")
        logger.log(
            json.dumps(ppp.dict_to_safe_json(variant, sort=True), indent=2)
        )
        variant_log_path = osp.join(log_dir, variant_log_file)
        logger.log_variant(variant_log_path, variant)

    tabular_log_path = osp.join(log_dir, tabular_log_file)
    text_log_path = osp.join(log_dir, text_log_file)

    logger.add_text_output(text_log_path)
    if first_time:
        logger.add_tabular_output(tabular_log_path)
    else:
        logger._add_output(tabular_log_path, logger._tabular_outputs,
                           logger._tabular_fds, mode='a')
        for tabular_fd in logger._tabular_fds:
            logger._tabular_header_written.add(tabular_fd)
    logger.set_snapshot_dir(log_dir)
    logger.set_snapshot_mode(snapshot_mode)
    logger.set_snapshot_gap(snapshot_gap)
    logger.set_log_tabular_only(log_tabular_only)
    exp_name = log_dir.split("/")[-1]
    logger.push_prefix("[%s] " % exp_name)

    if git_infos is not None:
        for (
            directory, code_diff, code_diff_staged, commit_hash, branch_name
        ) in git_infos:
            if directory[-1] == '/':
                diff_file_name = directory[1:-1].replace("/", "-") + ".patch"
                diff_staged_file_name = (
                    directory[1:-1].replace("/", "-") + "_staged.patch"
                )
            else:
                diff_file_name = directory[1:].replace("/", "-") + ".patch"
                diff_staged_file_name = (
                    directory[1:].replace("/", "-") + "_staged.patch"
                )
            if code_diff is not None and len(code_diff) > 0:
                with open(osp.join(log_dir, diff_file_name), "w") as f:
                    f.write(code_diff + '\n')
            if code_diff_staged is not None and len(code_diff_staged) > 0:
                with open(osp.join(log_dir, diff_staged_file_name), "w") as f:
                    f.write(code_diff_staged + '\n')
            with open(osp.join(log_dir, "git_infos.txt"), "a") as f:
                f.write("directory: {}".format(directory))
                f.write('\n')
                f.write("git hash: {}".format(commit_hash))
                f.write('\n')
                f.write("git branch name: {}".format(branch_name))
                f.write('\n\n')
    if script_name is not None:
        with open(osp.join(log_dir, "script_name.txt"), "w") as f:
            f.write(script_name)
    return log_dir


def set_seed(seed):
    """
    Set the seed for all the possible random number generators.
    :param seed:
    :return: None
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.set_random_seed(seed)
    except ImportError as e:
        print("Could not import tensorflow. Skipping tf.set_random_seed")


def reset_execution_environment(logger=default_logger):
    """
    Call this between calls to separate experiments.
    :return:
    """
    try:
        import tensorflow as tf
        tf.reset_default_graph()
    except ImportError as e:
        print("Could not import tensorflow. Skipping tf.reset_default_graph")
    import importlib
    importlib.reload(logger)


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.
    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).
    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")
