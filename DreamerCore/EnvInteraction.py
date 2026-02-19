import imageio
import torch
import torch.nn as nn
import numpy as np

class EnvironmentInteraction(nn.Module):
  def __init__(self,  core, config):
      super().__init__()
      self.config = config
      self.device = core.device

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
      self.totalEpisodes = 0
      self.totalEnvSteps = 0

  @torch.no_grad()
  def envInteraction(self, env, numEpisodes, seed=None, evaluation=False, savideo=False, fileName='videos/testvideo',fps=30, macroBlockSize=16):
    scores = []
    for i in range(numEpisodes):
      recurrentState = torch.zeros(1, self.recurrentSize, device=self.device)
      latentState = torch.zeros(1, self.latentsize, device=self.device)
      action = torch.zeros(1, self.action_size).to(self.device) #
      observation = env.reset(seed= (seed + self.totalEpisodes if seed else None))
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

        nextObservation, reward, done = env.step(actionindex)

        if not evaluation:
          self.buffer.add(observation, actionindex, reward, nextObservation, done)
        if savideo and i == 0:
          frame = env.render()
          targetheight = (frame.shape[0] + macroBlockSize - 1)//macroBlockSize*macroBlockSize
          targetwidth = (frame.shape[1] + macroBlockSize - 1)//macroBlockSize*macroBlockSize
          frames.append(np.pad(frame, ((0, targetheight - frame.shape[0]), (0 , targetwidth - frame.shape[1]),(0,0)), mode='edge'))

        encodedObs = self.encoder(torch.from_numpy(nextObservation).float().unsqueeze(0).to(self.device))
        observation = nextObservation
        currScore += reward
        stepCount += 1

        if done:
          scores.append(currScore)
          if not evaluation:
            self.totalEpisodes += 1
            self.totalEnvSteps += stepCount

          if savideo and i == 0:
            finalFilename = f'{fileName} _reward_{currScore:.0f}.mp4'
            with imageio.get_writer(finalFilename, fps=fps) as video:
              for frame in frames:
                video.append_data(frame)
             
          break
    return sum(scores)/numEpisodes if numEpisodes else None

