
import gym
import gym_super_mario_bros
import numpy as np
from scipy.ndimage import gaussian_filter


def getEnvinfo(env):
  observationShape = env.observation_space.shape
  actionSize = env.action_space.n
  return observationShape, actionSize

class foveatedObservation(gym.ObservationWrapper):
  def __init__(self, env, inner_radius=8, middle_radius=20, middle_blur=1.0, outer_blur=2.0):
    super().__init__(env)
    self.inner_radius = inner_radius
    self.middle_radius = middle_radius
    self.middle_blur = middle_blur
    self.outer_blur = outer_blur
    self.last_x = None
    self.last_y = None

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    if isinstance(info, dict):
      x_pos = info.get("x_pos", None)
      y_pos = info.get("y_pos", None)

      if x_pos is not None:
        self.last_x = int(x_pos % obs.shape[1])
      if y_pos is not None:
        self.last_y = int(obs.shape[0] - (y_pos % obs.shape[0]))

    obs = self.apply_foveation(obs)
    return obs, reward, done, info

  def observation(self, obs):
    return self.apply_foveation(obs)

  def apply_foveation(self, img):
    h, w = img.shape[:2]
    if self.last_x is None:
      center_x = w // 2
    else:
      center_x = int((self.last_x + 6) % w)

    if self.last_y is None:
      center_y = h // 2
    else:
      center_y = int(self.last_y)

    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    middle_blur_frame = gaussian_filter(img, sigma=(self.middle_blur, self.middle_blur, 0))
    outer_blur_frame = gaussian_filter(img, sigma=(self.outer_blur, self.outer_blur, 0))

    inner_mask = distance <= self.inner_radius
    middle_mask = (distance > self.inner_radius) & (distance <= self.middle_radius)
    outer_mask = distance > self.middle_radius

    result = np.zeros_like(img)
    result[inner_mask] = img[inner_mask]
    result[middle_mask] = middle_blur_frame[middle_mask]
    result[outer_mask] = outer_blur_frame[outer_mask]

    return result.astype(np.uint8)
    
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

class envPreproccessing(gym.ObservationWrapper):
  def __init__(self, env):
    super().__init__(env)
    h, w, c = self.observation_space.shape
    self.observation_space = gym.spaces.Box(low=0, high=1, shape=(c ,h, w), dtype=np.float32)

  def observation(self, obs):
    observation = np.transpose(obs, (2, 0, 1))/255.0
    return observation


class envWrapper(gym.Wrapper):
  def __init__(self, env):
    super().__init__(env)

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    return obs, reward, done, info

  def reset(self, seed=None):
    try:
      if seed is not None:
        result = self.env.reset(seed=seed)
      else:
        result = self.env.reset()
    except TypeError:
      result = self.env.reset()

    if isinstance(result, tuple):
      obs, info = result
    else:
      obs = result

    return obs
