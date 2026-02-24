import imageio
import torch
import torch.nn as nn
import numpy as np

class EnvironmentInteraction(nn.Module):
  def __init__(self,  core, config):
      super().__init__()
      self.config = config
      self.device = core.device
      self.core = core

      self.observation_shape = core.observation_shape
      self.latentsize = core.latentsize
      self.action_size = core.action_size
      self.recurrentSize = core.recurrentSize

      self.encoder = core.encoder
      self.decoder = core.decoder
      self.recurrentModel = core.recurrentModel
      self.posterior = core.posterior
      self.actor = core.actor

      self.buffer = core.buffer
      # self.totalEpisodes = core.totalEpisodes
      # self.totalEnvSteps = core.totalEnvSteps

  @torch.no_grad()
  def envInteraction(self, env, numEpisodes, seed=None, evaluation=False, savevideo=False, fileName='videos/testvideo',fps=30, macroBlockSize=16):
    scores = []
    for i in range(numEpisodes):
      recurrentState = torch.zeros(1, self.recurrentSize, device=self.device)
      latentState = torch.zeros(1, self.latentsize, device=self.device)
      action = torch.zeros(1, self.action_size).to(self.device) #
      if seed is not None:
        try:
          observation = env.reset(seed = seed + self.totalEpisodes)
        except:
          observation = env.reset()
      else:
        observation = env.reset()
      self._max_x = 0
      # observation = env.reset(seed= (seed + self.totalEpisodes if seed else None))
      encodedObs = self.encoder(torch.from_numpy(observation).float().unsqueeze(0).to(self.device))

      currScore, stepCount, done, frames = 0, 0, False, []
      while not done:
        recurrentState = self.recurrentModel(recurrentState, latentState, action)
        latentState, _ = self.posterior(torch.cat((recurrentState, encodedObs.view(1, -1)), -1))
        fullState = torch.cat((recurrentState, latentState), -1)
        action, _, _, _ = self.actor(fullState,
                            self.config.dreamer.actorModel.tau,
                            training=False)
        actionindex = action.argmax(dim=1).item()

        nextObservation, reward, done , info= env.step(actionindex)
        x = info.get('x_pos', 0)

        if not hasattr(self, '_max_x'):
            self._max_x = 0
        progress = max(0, x - self._max_x)
        self._max_x = max(self._max_x, x)
        shaped_reward = progress * 0.1

        if not evaluation:
          one_hot = np.zeros(self.action_size , dtype=np.float32)
          one_hot[actionindex] = 1.0
          self.buffer.add(observation, one_hot, shaped_reward, nextObservation, done)

        if savevideo and i == 0:
          try:
            frame = env.unwrapped.render(mode="rgb_array")
            targetheight = (frame.shape[0] + macroBlockSize - 1)//macroBlockSize*macroBlockSize
            targetwidth = (frame.shape[1] + macroBlockSize - 1)//macroBlockSize*macroBlockSize
            frames.append(np.pad(frame, ((0, targetheight - frame.shape[0]), (0 , targetwidth - frame.shape[1]),(0,0)), mode='edge'))
          except Exception as e:
            print('Render Skipped', e)

        encodedObs = self.encoder(torch.from_numpy(nextObservation).float().unsqueeze(0).to(self.device))
        observation = nextObservation
        currScore += shaped_reward
        stepCount += 1

        if done:
          scores.append(currScore)
          if not evaluation:
            self.core.totalEpisodes += 1
            self.core.totalEnvSteps += stepCount

          if savevideo and i == 0:
            finalFilename = f'{fileName}_step{stepCount:.0f}_reward_{currScore:.0f}.mp4'
            with imageio.get_writer(finalFilename, fps=fps) as video:
              for frame in frames:
                video.append_data(frame)

          break
    return sum(scores)/numEpisodes if numEpisodes else None

