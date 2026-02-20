
import torch
import torch.nn as nn
from Utils.Utils import get_activation
class Decoder(nn.Module):
  def __init__(self, inputsize, outputshape, activation, config):
    super().__init__()
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.channels, self.height, self.width = outputshape
    self.activation = get_activation(activation)
    self.network = nn.Sequential(
        nn.Linear(inputsize, config.dreamer.decoder.depth*32*2*2),
        nn.Unflatten(1, (config.dreamer.decoder.depth*32 , 2, 2)),
        nn.Unflatten(2, (1, 1)),
        nn.ConvTranspose2d(config.dreamer.decoder.depth*32, config.dreamer.decoder.depth*4, config.dreamer.decoder.kernelSize, config.dreamer.decoder.stride),
        self.activation,
        nn.ConvTranspose2d(config.dreamer.decoder.depth*4, config.dreamer.decoder.depth*2, config.dreamer.decoder.kernelSize, config.dreamer.decoder.stride),
        self.activation,
        nn.ConvTranspose2d(config.dreamer.decoder.depth*2, config.dreamer.decoder.depth*1, config.dreamer.decoder.kernelSize+1, config.dreamer.decoder.stride),
        self.activation,
        nn.ConvTranspose2d(config.dreamer.decoder.depth*1, self.channels, config.dreamer.decoder.kernelSize+1, config.dreamer.decoder.stride),
    )
  def forward(self,x):
    return self.network(x)
