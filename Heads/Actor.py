from math import tau
import torch
import torch.nn as nn
from torch.distributions import OneHotCategorical
from torch.nn import functional as F
class Actor(nn.Module):
  def __init__(self, latent_classes, latent_length, deterministic_size, actionspace, activation, hidden_size, layer_size):
    super().__init__()
    self.latent_size = latent_classes*latent_length
    inputsize = deterministic_size+self.latent_size
    self.network = build_nn(
        inputsize,
        hidden_size,
        layer_size,
        actionspace,
        activation
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
