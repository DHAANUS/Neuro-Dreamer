import attridict
import numpy as np
import torch
class ReplayBuffer(object):
  def __init__(self, config, device, observation_shape, action_size):
    self.config = config
    self.device = device
    self.capacity = int(self.config.capacity)

    self.obsertavtions = np.empty((self.capacity, *observation_shape), dtype=np.float32)
    self.next_observations = np.empty((self.capacity, *observation_shape), dtype=np.float32)

    self.actions = np.empty((self.capacity, action_size), dtype=np.float32)
    self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
    self.dones = np.empty((self.capacity, 1), dtype=np.float32)

    self.bufferindex = 0
    self.full = False

  def __len__(self):
    return self.capacity if self.full else self.bufferindex

  def add(self, observation, action, reward, next_observation, done):
    self.observations[self.bufferindex] = observation
    self.actions[self.bufferindex] = action
    self.rewards[self.bufferindex] = reward
    self.next_observations[self.bufferindex] = next_observation
    self.dones[self.bufferindex] = done

    self.bufferindex = (self.bufferindex+1)%self.capacity
    self.full = self.full or self.bufferindex == 0

  def sample(self, batchSize, sequenceSize):
    validStart = self.bufferindex - sequenceSize + 1
    assert self.full or (batchSize<validStart), "Not Enough Data in the Buffer to Sample bro"
    sampleindex = np.random.randint(0, self.capacity if self.full else validStart, batchSize).reshape(-1,1)
    sequenceLength = np.arange(sequenceSize).reshape(1, -1)

    sampleIndex = (sampleIndex+sequenceLength)%self.capacity

    observations = torch.as_tensor(self.observations[sampleIndex], device=self.device).float()
    next_observations = torch.as_tensor(self.next_observations[sampleIndex], device=self.device).float()
    actions = torch.as_tensor(self.actions[sampleIndex], device=self.device)
    rewards = torch.as_tensor(self.rewards[sampleIndex], device=self.device)
    dones = torch.as_tensor(self.dones[sampleIndex], device=self.device)

    sample = attridict({
        "observations": observations,
        "next_observations": next_observations,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
    })
    return sample
