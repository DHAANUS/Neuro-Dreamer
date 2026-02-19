import torch
import torch.nn as nn
from Utils.Utils import Moments, calc_lambda_values
class BehaviorModel(nn.Module):
  def __init__(self,  core ,config):
    super().__init__()
    self.config = config

    self.observation_shape = core.observation_shape
    self.latentsize = core.latentsize
    self.recurrentSize = core.recurrentSize
    self.fullStateSize = core.fullStateSize
    self.action_size = core.action_size

    self.device = core.device

    self.actor = core.actor
    self.critic = core.critic
    self.recurrentModel = core.recurrentModel
    self.prior = core.prior

    self.reward = core.reward
    self.valueMoments = Moments(self.device)

    self.actorOptimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.dreamer.actorLR)
    self.criticOptimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.dreamer.criticLR)
    if self.config.dreamer.useContinuationPred:
      self.continueModel =  core.continueModel
    else:
      self.continueModel = None
  def train_behaviour(self, fullState):
    recurrentState, latentState = torch.split(fullState, (self.recurrentSize, self.latentsize), -1)
    fullStates, logprobs, entropies = [], [], []
    for _ in range(self.config.dreamer.imaginehorizon):
      action, logprob, entropy, _ = self.actor(fullState.detach(), self.config.dreamer.actorModel.tau, training=True)
      recurrentState = self.recurrentModel(recurrentState, latentState, action)
      latentState, _ = self.prior(recurrentState)

      fullState = torch.cat((recurrentState, latentState),-1)
      fullStates.append(fullState)
      logprobs.append(logprob)
      entropies.append(entropy)

    fullStates = torch.stack(fullStates, dim=1)
    logprobs = torch.stack(logprobs[1:], dim=1)
    entropies = torch.stack(entropies[1:], dim=1)

    predReward = self.reward(fullStates[:, :-1]).mean
    values = self.critic(fullStates).mean
    if self.config.dreamer.useContinuationPred:
      continues = self.continueModel(fullStates).mean
    else:
      continues = torch.full_like(predReward, self.config.dreamer.discount)

    lambdavals = calc_lambda_values(predReward, values, continues, self.config.dreamer.lambda_)

    _, inverseScale = self.valueMoments(lambdavals)
    advantages = (lambdavals - values[:, :-1])/inverseScale

    actorLoss = - torch.mean(advantages.detach() * logprobs + self.config.dreamer.entropyScale * entropies)

    self.actorOptimizer.zero_grad()
    actorLoss.backward()
    nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.dreamer.gradientClip, norm_type=self.config.dreamer.gradientNormType)
    self.actorOptimizer.step()

    valuedist = self.critic(fullStates[:, :-1].detach())
    criticLoss = -torch.mean(valuedist.log_prob(lambdavals.detach()))

    self.criticOptimizer.zero_grad()
    criticLoss.backward()
    nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.dreamer.gradientClip, norm_type=self.config.dreamer.gradientNormType)
    self.criticOptimizer.step()

    metrics = {
        'actorLoss': actorLoss.item(),
        'criticLoss': criticLoss.item(),
        'entropies': entropies.mean().item(),
        'logprobs': logprobs.mean().item(),
        'advantages': advantages.mean().item(),
        'criticValues': values.mean().item()}
    return metrics

