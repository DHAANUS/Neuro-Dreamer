
import torch
import torch.nn as nn

from Models.RSSM import Recurrent_model, Prior, Posterior
from Models.Encoder import Encoder
from Models.Decoder import Decoder

from Heads.RewardModel import RewardModel
from Heads.Actor import Actor
from Heads.Critic import Critic
from Heads.ContinueModel import ContinueModel

from Utils.ReplayBuffer import ReplayBuffer

class CentralInitialization(nn.Module):
  def __init__(self, observation_shape, action_size, device, config):
    super().__init__()
    self.config = config
    self.observation_shape = observation_shape
    self.device = device

    self.latentsize = config.dreamer.latentlength*config.dreamer.latentclasses
    self.recurrentSize = config.dreamer.recurrentSize
    self.fullStateSize = self.latentsize+self.recurrentSize
    self.action_size = action_size


    self.encoder = Encoder(observation_shape,
                           config.dreamer.encodedObjSize,
                           config.dreamer.encoder.activation,
                           config).to(self.device)
    self.decoder = Decoder(self.fullStateSize,
                           observation_shape,
                           config.dreamer.decoder.activation).to(self.device)

    self.recurrentModel = Recurrent_Model(action_size,
                                          self.latentsize,
                                          self.recurrentSize,
                                          config).to(self.device)
    self.prior = Prior(self.recurrentSize,
                       config.dreamer.latentlength,
                       config.dreamer.latentclasses,
                       config).to(self.device)
    self.posterior = Posterior(self.recurrentSize + self.config.dreamer.encodedObjSize,
                               config.dreamer.latentlength,
                               config.dreamer.latentclasses,
                               config).to(self.device)

    self.reward = RewardModel(self.fullStateSize,
                              config).to(self.device)
    self.actor = Actor(self.fullStateSize, action_size, config).to(self.device)
    self.critic = Critic(self.fullStateSize, config).to(self.device)
    self.buffer = ReplayBuffer(config.dreamer.buffer, self.device,self.observation_shape, self.action_size)

    if self.config.dreamer.useContinuationPred:
      self.continueModel =  ContinueModel(self.fullStateSize, config).to(self.device)
    else:
      self.continueModel = None
    self.worldmodelParameters = (list(self.encoder.parameters()) +
                                 list(self.decoder.parameters()) +
                                 list(self.recurrentModel.parameters()) +
                                 list(self.prior.parameters()) +
                                 list(self.posterior.parameters()) +
                                 list(self.reward.parameters()))
    if self.config.dreamer.useContinuationPred:
      self.worldmodelParameters += list(self.continueModel.parameters())
    self.totalEpisodes = 0
    self.totalEnvSteps = 0
    self.totalGradientSteps = 0

