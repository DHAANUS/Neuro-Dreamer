import torch
import torch.nn as nn
from torch.distributions import Normal
class Critic(nn.Module):
  def __init__(self, inputsize, config):
    super().__init__()
    self.config = config.dreamer.criticModel
    self.inputsize = inputsize
    self.network = build_nn(
        inputsize,
        self.config.hiddenSize,
        self.config.hiddenLayers,
        2,
        activation
    )
  def forward(self,x):
    mean, logstd = self.network(x).chunk(2, dim=-1)
    std = torch.exp(logstd)
    return torch.distributions.Normal(mean.squeeze(-1), std.squeeze(-1))
