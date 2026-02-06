class RewardModel(nn.Module):
  def __init__(self, inputsize, activation, hidden_size, hidden_dim):
    super().__init__()
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.network = build_network(inputsize, 2 , activation, [hidden_size]*hidden_dim)

  def forward(self, x):
    out = self.network(x)
    mean , logsd = out.chunk(2, dim=-1)
    sd = torch.exp(logsd)
    return torch.distributions.Normal(mean, sd)
