import torch
import torch.nn as nn
from torch.distributions import OneHotCategorical
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
  def forward(self, x):
    return OneHotCategorical(logits=self.network(x))
