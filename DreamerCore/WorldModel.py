import torch
import torch.nn as nn
from torch.distributions import Independent, Normal, OneHotCategoricalStraightThrough, kl_divergence

class WorldModel(nn.Module):
  def __init__(self, core, config):
    super().__init__()
    self.config = config

    self.observation_shape = core.observation_shape
    self.latentsize = core.latentsize
    self.recurrentSize = core.recurrentSize
    self.fullStateSize = core.fullStateSize
    self.action_size = core.action_size
    self.device = core.device

    self.encoder = core.encoder
    self.decoder = core.decoder
    self.recurrentModel = core.recurrentModel
    self.prior = core.prior
    self.posterior = core.posterior
    self.reward = core.reward
    self.continueModel = core.continueModel

    self.worldmodelParameters = core.worldmodelParameters
    self.worldmodelOptimizer = torch.optim.Adam(self.worldmodelParameters, lr=self.config.dreamer.worldmodelLR)

  def train_world(self, data):
    encodedObs = self.encoder(data.observations.view(-1, *self.observation_shape)).view(self.config.dreamer.batchsize, self.config.dreamer.batchlength, -1)
    previousRecurrentState = torch.zeros(self.config.dreamer.batchsize, self.recurrentSize, device=self.device)
    previousLatentState = torch.zeros(self.config.dreamer.batchsize, self.latentsize, device=self.device)

    recurrentStates, priorLogits, posteriors, posteriorLogits = [], [], [], []
    for i in range(1, self.config.dreamer.batchlength):
      recurrentState = self.recurrentModel(previousRecurrentState, previousLatentState, data.actions[:,  i-1])
      _, priorLogit = self.prior(recurrentState)
      posterior, posteriorLogit = self.posterior(torch.cat((recurrentState, encodedObs[:, i]), -1))

      recurrentStates.append(recurrentState)
      priorLogits.append(priorLogit)
      posteriors.append(posterior)
      posteriorLogits.append(posteriorLogit)

      previousRecurrentState = recurrentState.detach()
      previousLatentState = posterior.detach()

    recurrentStates = torch.stack(recurrentStates, dim=1)
    priorLogits = torch.stack(priorLogits, dim=1)
    posteriors = torch.stack(posteriors, dim=1)
    posteriorLogits = torch.stack(posteriorLogits, dim=1)

    fullstates = torch.cat((recurrentStates, posteriors), dim=-1)
    # print(fullstates.shape)
    # reconstructionMean = self.decoder(fullstates.view(-1, self.fullStateSize)).view(self.config.dreamer.batchsize, self.config.dreamer.batchlength-1, *self.observation_shape)
    decoded = self.decoder(fullstates.view(-1, self.fullStateSize))
    # print(decoded.shape)
    # print(data.observations[:, 1:].shape)
    reconstructionMean = decoded.view(
      self.config.dreamer.batchsize,
      self.config.dreamer.batchlength-1,
      *decoded.shape[1:]
    )
    # print("Decoder:", reconstructionMean.shape)
    # print("Target :", data.observations[:, 1:].shape)
    reconstructiondist = Independent(Normal(reconstructionMean, 1), len(self.observation_shape))
    reconstructionLoss = -reconstructiondist.log_prob(data.observations[:, 1:]).mean()

    rewarddist = self.reward(fullstates)
    rewardLoss = -rewarddist.log_prob(data.rewards[:, 1:]).mean()

    priordist = Independent(OneHotCategoricalStraightThrough(logits=priorLogits), 1)
    priordiststopgradient = Independent(OneHotCategoricalStraightThrough(logits=priorLogits.detach()), 1)
    posteriordist = Independent(OneHotCategoricalStraightThrough(logits=posteriorLogits), 1)
    posteriordiststopgradient = Independent(OneHotCategoricalStraightThrough(logits=posteriorLogits.detach()), 1)

    priorLoss = kl_divergence(posteriordiststopgradient, priordist)
    posteriorLoss = kl_divergence(posteriordist, priordiststopgradient)
    freeNats = torch.full_like(priorLoss, self.config.dreamer.freeNats)

    priorLoss = self.config.dreamer.betaPrior * torch.maximum(priorLoss, freeNats)
    posteriorLoss = self.config.dreamer.betaPosterior * torch.maximum(posteriorLoss, freeNats)
    klLoss = (priorLoss + posteriorLoss).mean()

    worldmodelLoss = reconstructionLoss + rewardLoss + klLoss

    if self.config.dreamer.useContinuationPred:
      continueDist = self.continueModel(fullstates)
      bce = nn.BCELoss()
      continueLoss = bce(continueDist.probs, 1-data.dones[:, 1:])
      worldmodelLoss += continueLoss.mean()

    self.worldmodelOptimizer.zero_grad()
    worldmodelLoss.backward()
    nn.utils.clip_grad_norm_(self.worldmodelParameters, self.config.dreamer.gradientClip, norm_type=self.config.dreamer.gradientNormType)
    self.worldmodelOptimizer.step()

    klLossShift = (self.config.dreamer.betaPrior + self.config.dreamer.betaPosterior) *self.config.dreamer.freeNats

    metrics = {
        'worldmodelLoss' : worldmodelLoss.item() - klLossShift,
        'reconstructionLoss' : reconstructionLoss.item(),
        'rewardPredictionLoss' : rewardLoss.item(),
        'klLoss' : klLoss.item() - klLossShift
    }

    return fullstates.view(-1, self.fullStateSize).detach(), metrics

