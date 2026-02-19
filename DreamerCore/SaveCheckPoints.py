import torch
import torch.nn as nn
class Savingcheckpoints(nn.Module):
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

  def saveCheckpoint(self, checkpointPath, totalGradientSteps):
    if not checkpointPath.endswith('.pth'):
      checkpointPath += '.pth'

    checkpoint = {
        'encoder': self.encoder.state_dict(),
        'decoder': self.decoder.state_dict(),
        'recurrentModel': self.recurrentModel.state_dict(),
        'prior': self.prior.state_dict(),
        'posterior': self.posterior.state_dict(),
        'reward': self.reward.state_dict(),
        'actor': self.actor.state_dict(),
        'critic': self.critic.state_dict(),
        'worldmodelOptimizer': self.worldmodelOptimizer.state_dict(),
        'criticOptimizer': self.criticOptimizer.state_dict(),
        'actorOptimizer': self.actorOptimizer.state_dict(),
        'totalEpisodes': self.envinter.totalEpisodes,
        'totalEnvSteps': self.envinter.totalEnvSteps,
        'totalGradientSteps': totalGradientSteps
    }
    if self.config.dreamer.useContinuationPred:
      checkpoint['continueModel'] = self.continueModel.state_dict()
    torch.save(checkpoint, checkpointPath)

