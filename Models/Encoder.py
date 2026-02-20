
import torch
import torch.nn as nn
from Utils.Utils import get_activation
class Encoder(nn.Module):
  def __init__(self, inputshape, outputsize, activation, config):
    super().__init__()
    self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    channels, height, width = inputshape
    self.outputsize = outputsize
    self.activation = get_activation(config.dreamer.encoder.activation)
    self.network = nn.Sequential(
        nn.Conv2d(channels, config.dreamer.encoder.depth*1, config.dreamer.encoder.kernelSize, config.dreamer.encoder.stride, padding=1),
        self.activation,
        nn.Conv2d(config.dreamer.encoder.depth*1 ,config.dreamer.encoder.depth*2, config.dreamer.encoder.kernelSize, config.dreamer.encoder.stride, padding=1),
        self.activation,
        nn.Conv2d(config.dreamer.encoder.depth*2, config.dreamer.encoder.depth*4, config.dreamer.encoder.kernelSize, config.dreamer.encoder.stride, padding=1),
        self.activation,
        nn.Conv2d(config.dreamer.encoder.depth*4, config.dreamer.encoder.depth*8, config.dreamer.encoder.kernelSize, config.dreamer.encoder.stride, padding=1),
        self.activation,
        nn.Flatten(),
        nn.Linear(config.dreamer.encoder.depth*8*(height//config.dreamer.encoder.stride ** 4)*(width//config.dreamer.encoder.stride ** 4), outputsize),
        self.activation
    )
  def forward(self, x):
    return self.network(x).view(-1, self.outputsize)
