import torch
import torch.nn as nn
from torch.distributions import OneHotCategorical
from torch.nn import functional as F
class Actor(nn.Module):
  def __init__(self,  inputsize,actionspace, config):
    super().__init__()
    self.config = config.dreamer.actorModel
    self.inputsize = inputsize
    self.hidden_size = config.hiddenSize
    self.layer_size = config.hiddenLayers
    self.network = build_nn(
        inputsize,
        self.hidden_size,
        self.layer_size,
        actionspace,
        self.config.activation
    )
  def forward(self, x, tau,training=False):
    logits = self.network(x)
    dist = OneHotCategorical(logits=logits)

    if training:
      action = F.gumbel_softmax(logits,
                                tau=tau,
                                hard=True)
      action_index = action.argmax(dim=-1)
    else:
      action = dist.sample()
      action_index = action.argmax(dim=-1)

    log_prob = dist.log_prob(action_index)
    entropy = dist.entropy()
    return action, log_prob, entropy, logits
