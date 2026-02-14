class RewardModel(nn.Module):
  def __init__(self,inputSize, config):
    super().__init__()
    self.input_size = inputSize
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.network = build_nn(self.inputsize, config.dreamer.rewardModel.hiddenSize,config.dreamer.rewardModel.hiddenLayers ,2, config.dreamer.rewardModel.activation)

  def forward(self, x):
    out = self.network(x)
    mean , logsd = out.chunk(2, dim=-1)
    sd = torch.exp(logsd)
    return torch.distributions.Normal(mean, sd)
