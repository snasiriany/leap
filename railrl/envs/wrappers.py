import torch
import mujoco_py
import numpy as np
import gym.spaces
import itertools
from gym import Env, spaces
from gym.spaces import Box
from gym.spaces import Discrete
from PIL import Image

from collections import deque
from railrl.core.serializable import Serializable
import mujoco_py
import torch
import cv2


class ProxyEnv(Serializable, Env):
    def __init__(self, wrapped_env):
        Serializable.quick_init(self, locals())
        self._wrapped_env = wrapped_env
        self.action_space = self._wrapped_env.action_space
        self.observation_space = self._wrapped_env.observation_space

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def reset(self, **kwargs):
        return self._wrapped_env.reset(**kwargs)

    def step(self, action):
        return self._wrapped_env.step(action)

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    def log_diagnostics(self, paths, *args, **kwargs):
        if hasattr(self._wrapped_env, 'log_diagnostics'):
            self._wrapped_env.log_diagnostics(paths, *args, **kwargs)

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def terminate(self):
        if hasattr(self.wrapped_env, "terminate"):
            self.wrapped_env.terminate()

    def __getattr__(self, attrname):
        if attrname == '_serializable_initialized':
            return None
        return getattr(self._wrapped_env, attrname)

    def __str__(self):
        return '{}({})'.format(type(self).__name__, self.wrapped_env)


class HistoryEnv(ProxyEnv, Env):
    def __init__(self, wrapped_env, history_len):
        self.quick_init(locals())
        super().__init__(wrapped_env)
        self.history_len = history_len

        high = np.inf * np.ones(self.history_len*self.observation_space.low.size)
        low = -high
        self.observation_space = Box(low=low,
                                     high=high,
                                     )
        self.history = deque(maxlen=self.history_len)

    def step(self, action):
        state, reward, done, info = super().step(action)
        self.history.append(state)
        flattened_history = self._get_history().flatten()
        return flattened_history, reward, done, info

    def reset(self, **kwargs):
        state = super().reset()
        self.history = deque(maxlen=self.history_len)
        self.history.append(state)
        flattened_history = self._get_history().flatten()
        return flattened_history

    def _get_history(self):
        observations = list(self.history)

        obs_count = len(observations)
        for _ in range(self.history_len - obs_count):
            dummy = np.zeros(self._wrapped_env.observation_space.low.size)
            observations.append(dummy)
        return np.c_[observations]


class ImageMujocoEnv(ProxyEnv, Env):
    def __init__(self,
                 wrapped_env,
                 imsize=32,
                 keep_prev=0,
                 init_camera=None,
                 camera_name=None,
                 transpose=False,
                 grayscale=False,
                 normalize=False,
    ):
        self.quick_init(locals())
        super().__init__(wrapped_env)

        self.imsize = imsize
        if grayscale:
            self.image_length = self.imsize * self.imsize
        else:
            self.image_length = 3 * self.imsize * self.imsize
        # This is torch format rather than PIL image
        self.image_shape = (self.imsize, self.imsize)
        # Flattened past image queue
        self.history_length = keep_prev + 1
        self.history = deque(maxlen=self.history_length)
        # init camera
        if init_camera is not None:
            sim = self._wrapped_env.sim
            viewer = mujoco_py.MjRenderContextOffscreen(sim, device_id=-1)
            init_camera(viewer.cam)
            sim.add_render_context(viewer)
        self.camera_name = camera_name # None means default camera
        self.transpose = transpose
        self.grayscale = grayscale
        self.normalize = normalize
        self._render_local = False

        self.observation_space = Box(low=0.0,
                                     high=1.0,
                                     shape=(self.image_length * self.history_length,))

    def step(self, action):
        # image observation get returned as a flattened 1D array
        true_state, reward, done, info = super().step(action)

        observation = self._image_observation()
        self.history.append(observation)
        history = self._get_history().flatten()
        full_obs = self._get_obs(history, true_state)
        return full_obs, reward, done, info

    def reset(self, **kwargs):
        true_state = super().reset(**kwargs)
        self.history = deque(maxlen=self.history_length)

        observation = self._image_observation()
        self.history.append(observation)
        history = self._get_history().flatten()
        full_obs = self._get_obs(history, true_state)
        return full_obs

    def get_image(self):
        """TODO: this should probably consider history"""
        return self._image_observation()

    def _get_obs(self, history_flat, true_state):
        # adds extra information from true_state into to the image observation.
        # Used in ImageWithObsEnv.
        return history_flat

    def _image_observation(self):
        # returns the image as a torch format np array
        image_obs = self._wrapped_env.sim.render(width=self.imsize, height=self.imsize, camera_name=self.camera_name)
        if self._render_local:
            cv2.imshow('env', image_obs)
            cv2.waitKey(1)
        if self.grayscale:
            image_obs = Image.fromarray(image_obs).convert('L')
            image_obs = np.array(image_obs)
        if self.normalize:
            image_obs = image_obs / 255.0
        if self.transpose:
            image_obs = image_obs.transpose()
        return image_obs

    def _get_history(self):
        observations = list(self.history)

        obs_count = len(observations)
        for _ in range(self.history_length - obs_count):
            dummy = np.zeros(self.image_shape)
            observations.append(dummy)
        return np.c_[observations]

    def retrieve_images(self):
        # returns images in unflattened PIL format
        images = []
        for image_obs in self.history:
            pil_image = self.torch_to_pil(torch.Tensor(image_obs))
            images.append(pil_image)
        return images

    def split_obs(self, obs):
        # splits observation into image input and true observation input
        imlength = self.image_length * self.history_length
        obs_length = self.observation_space.low.size
        obs = obs.view(-1, obs_length)
        image_obs = obs.narrow(start=0,
                               length=imlength,
                               dimension=1)
        if obs_length == imlength:
            return image_obs, None

        fc_obs = obs.narrow(start=imlength,
                            length=obs.shape[1] - imlength,
                            dimension=1)
        return image_obs, fc_obs

    def enable_render(self):
        self._render_local = True


class ImageMujocoWithObsEnv(ImageMujocoEnv):
    def __init__(self, env, **kwargs):
        self.quick_init(locals())
        super().__init__(env, **kwargs)
        self.observation_space = Box(low=0.0,
                                     high=1.0,
                                     shape=(self.image_length * self.history_length +
                                            self.wrapped_env.obs_dim,))

    def _get_obs(self, history_flat, true_state):
        return np.concatenate([history_flat,
                               true_state])


class DiscretizeEnv(ProxyEnv, Env):
    def __init__(self, wrapped_env, num_bins):
        self.quick_init(locals())
        super().__init__(wrapped_env)
        low = self.wrapped_env.action_space.low
        high = self.wrapped_env.action_space.high
        action_ranges = [
            np.linspace(low[i], high[i], num_bins)
            for i in range(len(low))
        ]
        self.idx_to_continuous_action = [
            np.array(x) for x in itertools.product(*action_ranges)
        ]
        self.action_space = Discrete(len(self.idx_to_continuous_action))

    def step(self, action):
        continuous_action = self.idx_to_continuous_action[action]
        return super().step(continuous_action)



class NormalizedBoxEnv(ProxyEnv, Serializable):
    """
    Normalize action to in [-1, 1].

    Optionally normalize observations and scale reward.
    """
    def __init__(
            self,
            env,
            reward_scale=1.,
            obs_mean=None,
            obs_std=None,
    ):
        # self._wrapped_env needs to be called first because
        # Serializable.quick_init calls getattr, on this class. And the
        # implementation of getattr (see below) calls self._wrapped_env.
        # Without setting this first, the call to self._wrapped_env would call
        # getattr again (since it's not set yet) and therefore loop forever.
        self._wrapped_env = env
        # Or else serialization gets delegated to the wrapped_env. Serialize
        # this env separately from the wrapped_env.
        self._serializable_initialized = False
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)
        self._should_normalize = not (obs_mean is None and obs_std is None)
        if self._should_normalize:
            if obs_mean is None:
                obs_mean = np.zeros_like(env.observation_space.low)
            else:
                obs_mean = np.array(obs_mean)
            if obs_std is None:
                obs_std = np.ones_like(env.observation_space.low)
            else:
                obs_std = np.array(obs_std)
        self._reward_scale = reward_scale
        self._obs_mean = obs_mean
        self._obs_std = obs_std
        ub = np.ones(self._wrapped_env.action_space.shape)
        self.action_space = Box(-1 * ub, ub)

    def estimate_obs_stats(self, obs_batch, override_values=False):
        if self._obs_mean is not None and not override_values:
            raise Exception("Observation mean and std already set. To "
                            "override, set override_values to True.")
        self._obs_mean = np.mean(obs_batch, axis=0)
        self._obs_std = np.std(obs_batch, axis=0)

    def _apply_normalize_obs(self, obs):
        return (obs - self._obs_mean) / (self._obs_std + 1e-8)

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        # Add these explicitly in case they were modified
        d["_obs_mean"] = self._obs_mean
        d["_obs_std"] = self._obs_std
        d["_reward_scale"] = self._reward_scale
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        self._obs_mean = d["_obs_mean"]
        self._obs_std = d["_obs_std"]
        self._reward_scale = d["_reward_scale"]

    def step(self, action):
        lb = self._wrapped_env.action_space.low
        ub = self._wrapped_env.action_space.high
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)

        wrapped_step = self._wrapped_env.step(scaled_action)
        next_obs, reward, done, info = wrapped_step
        if self._should_normalize:
            next_obs = self._apply_normalize_obs(next_obs)
        return next_obs, reward * self._reward_scale, done, info

    def __str__(self):
        return "Normalized: %s" % self._wrapped_env


"""
Some wrapper codes for rllab.
"""


class ConvertEnvToRllab(ProxyEnv, Serializable):
    """
    rllab sometimes requires the action/observation space to be a specific type
    """
    def __init__(self, env):
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)
        self.action_space = convert_space_to_rllab_space(
            self._wrapped_env.action_space
        )
        self.observation_space = convert_space_to_rllab_space(
            self._wrapped_env.observation_space
        )
        from rllab.envs.env_spec import EnvSpec
        self.spec = EnvSpec(
            observation_space=self.observation_space,
            action_space=self.action_space,
        )

    def __str__(self):
        return "RllabConverted: %s" % self._wrapped_env

    def get_param_values(self):
        if hasattr(self.wrapped_env, "get_param_values"):
            return self.wrapped_env.get_param_values()
        return None


def convert_space_to_rllab_space(space):
    from sandbox.rocky.tf.spaces import Box as TfBox
    from sandbox.rocky.tf.spaces import Discrete as TfDiscrete
    from rllab.spaces.discrete import Discrete as RllabDiscrete
    from rllab.spaces.box import Box as RllabBox
    if isinstance(space, RllabBox) or isinstance(space, RllabDiscrete):
        return space
    if isinstance(space, TfBox) or isinstance(space, gym.spaces.Box):
        return RllabBox(space.low, space.high)
    elif isinstance(space, TfDiscrete) or isinstance(space, gym.spaces.Discrete):
        return RllabDiscrete(space.n)
    raise TypeError()


class ConvertEnvToTf(ProxyEnv, Serializable):
    def __init__(self, env):
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)
        self.action_space = convert_space_to_tf_space(
            self._wrapped_env.action_space
        )
        self.observation_space = convert_space_to_tf_space(
            self._wrapped_env.observation_space
        )
        from rllab.envs.env_spec import EnvSpec
        self.spec = EnvSpec(
            observation_space=self.observation_space,
            action_space=self.action_space,
        )

    def __str__(self):
        return "TfConverted: %s" % self._wrapped_env

    def get_param_values(self):
        if hasattr(self.wrapped_env, "get_param_values"):
            return self.wrapped_env.get_param_values()
        return None


def convert_space_to_tf_space(space):
    from sandbox.rocky.tf.spaces import Box as TfBox
    from sandbox.rocky.tf.spaces import Discrete as TfDiscrete
    from rllab.spaces.discrete import Discrete as RllabDiscrete
    from rllab.spaces.box import Box as RllabBox
    if isinstance(space, TfBox) or isinstance(space, TfDiscrete):
        return space
    if isinstance(space, RllabBox) or isinstance(space, gym.spaces.Box):
        return TfBox(space.low, space.high)
    elif isinstance(space, RllabDiscrete) or isinstance(space, gym.spaces.Discrete):
        return TfDiscrete(space.n)
    raise TypeError()
