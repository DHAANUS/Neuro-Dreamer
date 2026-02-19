import os
import torch
import torch.nn as nn
class LoadCheckpoints(nn.Module):
  def __init__(self, core, worldmodel, behaviour, envinter,config):
    super().__init__()
    self.config = config
    self.device = core.device
    self.encoder = core.encoder
    self.decoder = core.decoder
    self.recurrentModel = core.recurrentModel
    self.prior = core.prior
    self.posterior = core.posterior
    self.reward = core.reward
    self.actor = core.actor
    self.critic = core.critic
    self.continueModel = core.continueModel
    self.worldmodelOptimizer = worldmodel.worldmodelOptimizer
    self.actorOptimizer = behaviour.actorOptimizer
    self.criticOptimizer = behaviour.criticOptimizer
    self.envinter = envinter


  def loadCheckPoint(self, checkpointPath):
    if not checkpointPath.endswith('.pth'):
      checkpointPath += '.pth'
    if not os.path.exists(checkpointPath):
      raise ValueError(f'Checkpoint {checkpointPath} does not exist')

    checkpoint = torch.load(checkpointPath, map_location = self.device)
    self.encoder.load_state_dict(checkpoint['encoder'])
    self.decoder.load_state_dict(checkpoint['decoder'])
    self.recurrentModel.load_state_dict(checkpoint['recurrentModel'])
    self.prior.load_state_dict(checkpoint['prior'])
    self.posterior.load_state_dict(checkpoint['posterior'])
    self.reward.load_state_dict(checkpoint['reward'])
    self.actor.load_state_dict(checkpoint['actor'])
    self.critic.load_state_dict(checkpoint['critic'])
    self.worldmodelOptimizer.load_state_dict(checkpoint['worldmodelOptimizer'])
    self.criticOptimizer.load_state_dict(checkpoint['criticOptimizer'])
    self.actorOptimizer.load_state_dict(checkpoint['actorOptimizer'])
    self.envinter.totalEpisodes = checkpoint['totalEpisodes']
    self.envinter.totalEnvSteps = checkpoint['totalEnvSteps']
    self.totalGradientSteps = checkpoint['totalGradientSteps']
    if self.config.dreamer.useContinuationPred:
      self.continueModel.load_state_dict(checkpoint['continueModel'])
