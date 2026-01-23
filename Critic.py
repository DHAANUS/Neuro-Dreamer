class Critic(nn.Module):
  def __init__(self, inputsize, activation):
    super().__init__()
    self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    self.network = build_network(
        inputsize,
        2,
        activation,
        16*16
    )
  def forward(self,x):
    mean, logstd = self.network(x).chunk(2, dim=-1)
    std = torch.exp(logstd)
    return torch.distributions.normal(mean.squeezq(-1), std.squeeze(-1))
