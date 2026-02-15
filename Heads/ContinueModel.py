class ContinueModel(nn.Module):
  def __init__(self,  inputsize, config):
    super().__init__()
    self.config = config.dreamer.continuationModel
    self.input_size = inputsize
    self.network = build_nn(self.input_size,
                            self.config.hiddenSize,
                            self.config.hiddenLayers,
                            1, 
                            self.config.activation)

  def forward(self, x):
    return torch.distributions.Bernoulli(logits=self.network(x))
