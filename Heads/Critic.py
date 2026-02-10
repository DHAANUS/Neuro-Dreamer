class Critic(nn.Module):
  def __init__(self, latent_classes, latent_length, deterministic_size, activation, hidden_size, layer_size):
    super().__init__()
    self.latent_size = latent_classes*latent_length
    inputsize = deterministic_size+self.latent_size
    self.network = build_nn(
        inputsize,
        hidden_size,
        layer_size,
        2,
        activation
    )
  def forward(self,x):
    mean, logstd = self.network(x).chunk(2, dim=-1)
    std = torch.exp(logstd)
    return torch.distributions.Normal(mean.squeeze(-1), std.squeeze(-1))
