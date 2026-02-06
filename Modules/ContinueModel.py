class ContinueModel(nn.Module):
  def __init__(self, inputsize, activation, hidden_size, hidden_dim):
    super().__init__()
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.network = build_network(inputsize,1, activation, [hidden_size]*hidden_dim)

  def forward(self, x):
    return torch.distributions.Bernoulli(logits=self.network(x))
