
import gym
import gym_super_mario_bros
import numpy as np

def getEnvinfo(env):
  observationShape = env.observation_space.shape
  actionSize = env.action_space.n
  return observationShape, actionSize

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
    # done = terminated or truncated
    return obs, reward, done

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
    # obs, info = self.env.reset(seed=seed)
    return obs

