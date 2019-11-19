import uuid

filename = str(uuid.uuid4())

import skvideo.io
import numpy as np
import time

from torchvision.utils import save_image
import railrl.torch.pytorch_util as ptu

def add_border(img, imsize, pad_length, pad_color):
    img = img.reshape((-1, imsize, 3))
    img2 = np.ones((img.shape[0] + 2 * pad_length, img.shape[1] + 2 * pad_length, img.shape[2]), dtype=np.uint8) * pad_color
    img2[pad_length:-pad_length, pad_length:-pad_length, :] = img
    return img2

def get_image(*sweeps, imsize, pad_length=1, pad_color=255, two_frames=False):
    img = None
    for sweep in sweeps:
        if sweep is not None:
            if img is None:
                img = sweep.reshape(-1, imsize, imsize).transpose((1, 2, 0))
            else:
                img = np.concatenate((img, sweep.reshape(-1, imsize, imsize).transpose((1, 2, 0))))
    img = np.uint8(255 * img)
    if pad_length > 0:
        img = add_border(img, imsize, pad_length, pad_color)
    return img

def dump_video(
    env,
    policy,
    filename,
    rollout_function,
    qf=None,
    vf=None,
    rows=3,
    columns=6,
    pad_length=0,
    pad_color=255,
    do_timer=True,
    imsize=84,
    epoch=None,
    vis_list=None,
    vis_blacklist=None,
):
    if vis_list is None:
        vis_list = [
            'image_desired_goal',
            'image_observation',
            'reconstr_image_observation',
            'reconstr_image_reproj_observation',
            'image_desired_subgoal',
            'image_desired_subgoal_reproj',
            'image_plt',
            'image_latent_histogram_2d',
            'image_latent_histogram_mu_2d',
            'image_v_latent',
            'image_v',
            'image_v_noisy_state_and_goal',
            'image_v_noisy_state',
            'image_v_noisy_goal',
            'image_rew',
            'image_rew_euclidean',
            'image_rew_mahalanobis',
            'image_rew_logp',
            'image_rew_kl',
            'image_rew_kl_rev',
        ]

    if vis_blacklist is not None:
        vis_list = [x for x in vis_list if x not in vis_blacklist]

    num_channels = 1 if env.grayscale else 3
    frames = []
    N = rows * columns

    subgoal_images = []

    for i in range(N):
        start = time.time()
        path = rollout_function(
            env,
            policy,
            qf=qf,
            vf=vf,
            animated=False,
            epoch=epoch,
            rollout_num=i,
        )

        if 'image_desired_subgoals_reproj' in path['full_observations'][1]:
            image_ob = path['full_observations'][1]['image_observation'].reshape((-1, 3, imsize, imsize))
            if 'image_desired_goal_annotated' in path['full_observations'][1]:
                image_goal = path['full_observations'][1]['image_desired_goal_annotated'].reshape((-1, 3, imsize, imsize))
            else:
                image_goal = path['full_observations'][1]['image_desired_goal'].reshape((-1, 3, imsize, imsize))
            image_sg = path['full_observations'][1]['image_desired_subgoals_reproj']
            image_sg = image_sg.reshape((-1, 3, imsize, imsize))
            image = np.concatenate((image_ob, image_sg, image_goal))
            subgoal_images.append(image)

        mini_frames = []
        for d in path['full_observations'][1:]:
            get_image_kwargs = dict(
                pad_length=pad_length,
                pad_color=pad_color,
                imsize=imsize,
            )
            get_image_sweeps = [d.get(key, None) for key in vis_list]
            img = get_image(
                *get_image_sweeps,
                **get_image_kwargs,
            )

            mini_frames.append(img)
        horizon = len(mini_frames)
        frames += mini_frames
        if do_timer:
            print(i, time.time() - start)

    if len(subgoal_images) != 0:
        from railrl.core import logger
        import os.path as osp
        logdir = logger.get_snapshot_dir()
        filename_subgoals = osp.join(logdir, 'sg_{epoch}.png'.format(epoch=epoch))
        nrow = subgoal_images[0].shape[0]
        subgoal_images = np.concatenate(subgoal_images)
        save_image(
            ptu.FloatTensor(ptu.from_numpy(subgoal_images)),
            filename_subgoals,
            nrow=nrow,
        )

    frames = np.array(frames, dtype=np.uint8).reshape(
        (N, horizon, -1, imsize + 2 * pad_length, num_channels))
    f1 = []
    for k1 in range(columns):
        f2 = []
        for k2 in range(rows):
            k = k1 * rows + k2
            f2.append(frames[k:k+1, :, :, :, :].
                      reshape((horizon, -1, imsize + 2 * pad_length, num_channels)))
        f1.append(np.concatenate(f2, axis=1))
    outputdata = np.concatenate(f1, axis=2)
    skvideo.io.vwrite(filename, outputdata)
    print("Saved video to ", filename)