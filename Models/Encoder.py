
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
        nn.Conv2d(channels, 16, kernel_size=4, stride=4, padding=1),
        self.activation,
        nn.Conv2d(16,32, kernel_size=4, stride=2, padding=1),
        self.activation,
        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
        self.activation,
        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
        self.activation,
        nn.Flatten(),
        nn.Linear(128*(height//16)*(width//16), outputsize),
        self.activation
    )
  def forward(self, x):
    return self.network(x).view(-1, self.outputsize)
